#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMS–NM alignment pipeline (10.5 GV) — Reproducible Script
Author: ChatGPT
Date: 2025-10-07

What this script does:
  1) Load AMS daily proton rigidity spectrum (proton.csv)
  2) Compute the geometric-mean rigidity per bin, P_k
  3) Around P_target=10.5 GV, perform a 7-point local log-log weighted linear fit
     to get the local spectral index Gamma_10p5 per day
  4) Define a 3-day reference window from the start of the AMS series to compute
     DeltaGamma_10p5 and DeltaI_10p5 (fractional change of flux at ~10.5 GV)
  5) Load NM hourly count rates across stations, melt to long (time, station, count_rate),
     join cutoff rigidities (station_names_all.csv), compute per-station fractional change
     DeltaI_station relative to each station's first-3-day baseline, then aggregate to daily
  6) Export tidy CSVs and a couple of comparison plots (AMS vs NM daily DeltaI)

Inputs (edit PATHS as needed):
  - /mnt/data/proton.csv
  - /mnt/data/Hourly_Count_Rate_All_Stations_Clean - 副本.csv
  - /mnt/data/station_names_all.csv

Outputs:
  - /mnt/data/ams_10p5_daily_timeseries.csv
  - /mnt/data/nm_hourly_long_fractional.csv
  - /mnt/data/nm_daily_fractional.csv
  - /mnt/data/ams_vs_nm_deltaI_daily.png
  - /mnt/data/sopo_vs_ams_deltaI_daily.png

Python version:
  - Python 3.10+
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User-configurable parameters
# -----------------------------

PATHS = {
    "ams": "/home/zpw/Files/lstmPaper/data/raw_data/ams/proton.csv",
    "nm": "/home/zpw/Files/AMS_NeutronMonitors/NM_v0829/NM_0_preprocess_v250220/Hourly_Count_Rate_All_Stations_Clean - 副本.csv",
    "stations": "/home/zpw/Files/AMS_NeutronMonitors/NM_v0829/NM_0_preprocess_v250220/station_names_all.csv",
}
P_TARGET = 10.5   # GV, following the paper
REF_DAYS_AMS = 3  # 3-day reference for AMS
REF_DAYS_NM  = 3  # 3-day reference per station for NM

OUT = {
    "ams_series": "results/ams_10p5_daily_timeseries.csv",
    "nm_hourly_long": "results/nm_hourly_long_fractional.csv",
    "nm_daily": "results/nm_daily_fractional.csv",
    "plot_ams_vs_nm": "results/ams_vs_nm_deltaI_daily.png",
    "plot_sopo_vs_ams": "results/sopo_vs_ams_deltaI_daily.png",
}

REP_STATIONS = ["SOPO", "JUNG", "APTY", "ATHN"]  # representative stations to plot if available


# -----------------------------
# Helper functions
# -----------------------------

def compute_local_gamma_for_day(group_df: pd.DataFrame,
                                p_target: float,
                                flux_col: str = "flux",
                                flux_sys_err_col: str = "flux_sys_err") -> pd.Series:
    """
    Compute local spectral index Gamma at ~p_target using a weighted 7-point
    log-log linear fit of log(flux) vs log(P). Returns Gamma_10p5 and Flux_10p5.
    """
    g = group_df.sort_values("P_GV").reset_index(drop=True)

    # find index of bin closest to p_target
    idx = (g["P_GV"] - p_target).abs().idxmin()

    # window of 7 points centered on idx if possible
    start = max(0, idx - 3)
    end = min(len(g), idx + 4)
    window = g.iloc[start:end].copy()

    if len(window) < 5:
        return pd.Series({"Gamma_10p5": np.nan, "Flux_10p5": np.nan, "P_10p5": np.nan})

    # Weighted least squares: y = a + b x with y=log(flux), x=log(P) => Gamma=-b
    x = np.log(window["P_GV"].values)
    y = np.log(window[flux_col].values)

    # weights: use relative systematic error if available, else equal weights
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = window[flux_sys_err_col].values / window[flux_col].values
        w = 1.0 / np.where(np.isfinite(rel_err) & (rel_err > 0), rel_err**2, 1.0)

    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    try:
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        b = beta[1]
        gamma = -b
    except np.linalg.LinAlgError:
        gamma = np.nan

    # flux at the bin closest to p_target (not interpolated; follows the paper's approach)
    closest = g.loc[idx]
    return pd.Series({
        "Gamma_10p5": float(gamma),
        "Flux_10p5": float(closest[flux_col]),
        "P_10p5": float(closest["P_GV"]),
    })


def build_ams_series(ams_path: str,
                     p_target: float,
                     ref_days: int) -> pd.DataFrame:
    """
    Load AMS daily proton file and compute daily Gamma_10p5, DeltaGamma_10p5,
    Flux_10p5, and DeltaI_10p5 relative to the first `ref_days` days.
    """
    ams = pd.read_csv(ams_path)
    ams = ams.rename(columns={
        "date YYYY-MM-DD": "date",
        "rigidity_min GV": "Pmin_GV",
        "rigidity_max GV": "Pmax_GV",
        "proton_flux m^-2sr^-1s^-1GV^-1": "flux",
        "proton_flux_error_statistical m^-2sr^-1s^-1GV^-1": "flux_stat_err",
        "proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1": "flux_time_err",
        "proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1": "flux_sys_err",
    })
    ams["date"] = pd.to_datetime(ams["date"], utc=True)
    ams["P_GV"] = np.sqrt(ams["Pmin_GV"] * ams["Pmax_GV"])

    daily = ams.groupby("date").apply(
        lambda grp: compute_local_gamma_for_day(grp, p_target)
    ).reset_index()

    # Reference: first `ref_days` days from the start of the AMS series
    first_day = daily["date"].min().normalize()
    ref_mask = (daily["date"] >= first_day) & (daily["date"] < first_day + pd.Timedelta(days=ref_days))

    Gamma0 = daily.loc[ref_mask, "Gamma_10p5"].mean()
    Flux0 = daily.loc[ref_mask, "Flux_10p5"].mean()

    daily["DeltaGamma_10p5"] = daily["Gamma_10p5"] - Gamma0
    daily["DeltaI_10p5"] = (daily["Flux_10p5"] - Flux0) / Flux0

    return daily[["date", "P_10p5", "Gamma_10p5", "DeltaGamma_10p5", "Flux_10p5", "DeltaI_10p5"]].copy()


def build_nm_series(nm_path: str,
                    stations_path: str,
                    ref_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NM hourly wide table, melt to long, attach station cutoff rigidity,
    compute fractional change per station relative to first `ref_days` days,
    and aggregate to daily means.
    Returns: (nm_hourly_long, nm_daily_agg)
    """
    nm = pd.read_csv(nm_path)
    # Guess/require a time column named 'datetime'
    if "datetime" not in nm.columns:
        # fallback: try to find a plausible time column
        candidates = [c for c in nm.columns if "time" in c.lower() or "date" in c.lower()]
        if not candidates:
            raise ValueError("No datetime-like column found in NM file.")
        time_col = candidates[0]
        nm = nm.rename(columns={time_col: "datetime"})
    nm["datetime"] = pd.to_datetime(nm["datetime"], utc=True, errors="coerce")

    count_cols = [c for c in nm.columns if c.endswith("_count_rate")]
    if not count_cols:
        raise ValueError("No *_count_rate columns found in NM file.")

    # Melt to long
    nm_long = nm[["datetime"] + count_cols].melt(
        id_vars="datetime", var_name="station_col", value_name="count_rate"
    )
    nm_long["station"] = nm_long["station_col"].str.replace("_count_rate", "", regex=False)

    # Attach cutoff
    stations = pd.read_csv(stations_path)
    stations = stations.rename(columns={"station_name": "station", "cutoff_R": "cutoff_R"})
    nm_long = nm_long.merge(stations, on="station", how="left")

    # drop missing rates
    nm_long = nm_long.dropna(subset=["count_rate"])

    # per-station baseline over first `ref_days` days
    first_dt = nm_long["datetime"].min().floor("D")
    baseline_end = first_dt + pd.Timedelta(days=ref_days)
    baseline = nm_long[(nm_long["datetime"] >= first_dt) & (nm_long["datetime"] < baseline_end)]
    baseline_mean = baseline.groupby("station")["count_rate"].mean().rename("baseline_rate").reset_index()

    nm_long = nm_long.merge(baseline_mean, on="station", how="left")
    nm_long["DeltaI_station"] = (nm_long["count_rate"] - nm_long["baseline_rate"]) / nm_long["baseline_rate"]

    # build daily agg
    nm_long["date"] = nm_long["datetime"].dt.floor("D")
    nm_daily = nm_long.groupby(["date", "station", "cutoff_R"], as_index=False).agg(
        DeltaI_station=("DeltaI_station", "mean"),
        count_rate=("count_rate", "mean")
    )
    # Keep key columns in hourly
    nm_hourly_export = nm_long[["datetime", "station", "cutoff_R", "count_rate", "baseline_rate", "DeltaI_station"]].copy()

    return nm_hourly_export, nm_daily


def plot_ams_vs_nm_daily(ams_series: pd.DataFrame,
                         nm_daily: pd.DataFrame,
                         out_path: str,
                         rep_stations: list[str] | None = None) -> None:
    """
    Plot AMS DeltaI(10.5 GV) vs a few representative NM stations' daily DeltaI.
    """
    df = nm_daily.copy()
    ams = ams_series.copy()
    ams["date"] = pd.to_datetime(ams["date"]).dt.floor("D")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")

    # pick available stations
    if rep_stations is None:
        rep_stations = []
    available = sorted(set(df["station"]).intersection(rep_stations))

    plt.figure(figsize=(12, 4.5))
    for st in available:
        sub = df[df["station"] == st]
        merged = sub.merge(ams[["date", "DeltaI_10p5"]], on="date", how="inner")
        if merged.empty:
            continue
        plt.plot(merged["date"], merged["DeltaI_station"], label=f"{st} ΔI (station)")
    # Add AMS line (deduplicate by date)
    plt.plot(ams["date"].drop_duplicates(), ams.drop_duplicates("date")["DeltaI_10p5"], label="AMS ΔI @ 10.5 GV")
    plt.title("Daily relative change: AMS (10.5 GV) vs representative NM stations")
    plt.xlabel("Date (UTC)")
    plt.ylabel("ΔI (fraction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sopo_vs_ams_daily(ams_series: pd.DataFrame,
                           nm_daily: pd.DataFrame,
                           out_path: str) -> None:
    """
    Plot SOPO (South Pole) daily DeltaI vs AMS DeltaI(10.5 GV).
    """
    ams = ams_series.copy()
    ams["date"] = pd.to_datetime(ams["date"]).dt.floor("D")

    sopo = nm_daily[nm_daily["station"] == "SOPO"].copy()
    if sopo.empty:
        # No SOPO in the dataset; skip plotting
        return
    sopo["date"] = pd.to_datetime(sopo["date"]).dt.floor("D")

    merged = sopo.merge(ams[["date", "DeltaI_10p5"]], on="date", how="inner")
    if merged.empty:
        return

    plt.figure(figsize=(12, 4.5))
    plt.plot(merged["date"], merged["DeltaI_station"], label="SOPO ΔI (station)")
    plt.plot(merged["date"], merged["DeltaI_10p5"], label="AMS ΔI @ 10.5 GV")
    plt.title("Daily relative change: SOPO vs AMS (10.5 GV)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("ΔI (fraction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    # 1) AMS
    print("[AMS] Loading and computing daily 10.5 GV series ...")
    ams_series = build_ams_series(PATHS["ams"], P_TARGET, REF_DAYS_AMS)
    ams_series.to_csv(OUT["ams_series"], index=False)
    print(f"[AMS] Saved: {OUT['ams_series']}  (rows={len(ams_series)})")

    # 2) NM
    print("[NM] Loading and computing hourly/daily fractional changes ...")
    nm_hourly_long, nm_daily = build_nm_series(PATHS["nm"], PATHS["stations"], REF_DAYS_NM)
    nm_hourly_long.to_csv(OUT["nm_hourly_long"], index=False)
    nm_daily.to_csv(OUT["nm_daily"], index=False)
    print(f"[NM] Saved hourly (long): {OUT['nm_hourly_long']} (rows={len(nm_hourly_long)})")
    print(f"[NM] Saved daily aggregate: {OUT['nm_daily']} (rows={len(nm_daily)})")

    # 3) Plots
    print("[PLOT] Building comparison plots ...")
    plot_ams_vs_nm_daily(ams_series, nm_daily, OUT["plot_ams_vs_nm"], REP_STATIONS)
    print(f"[PLOT] Saved: {OUT['plot_ams_vs_nm']}")
    plot_sopo_vs_ams_daily(ams_series, nm_daily, OUT["plot_sopo_vs_ams"])
    print(f"[PLOT] Saved: {OUT['plot_sopo_vs_ams']}")

    print("[DONE] Pipeline finished successfully.")


if __name__ == "__main__":
    main()
