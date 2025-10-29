#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFA-lite (NM-only, 18 stations) — Isotropic-Only Prototype
==========================================================

Purpose
-------
Replicate (as closely as possible with limited inputs) the paper's idea of
recovering Forbush Decrease (FD) spectral behavior at a reference rigidity
P_r = 10.5 GV using only neutron monitor (NM) hourly data from ~18 stations.

What it does
------------
1) Loads NM hourly count rates and station geomagnetic cutoff rigidities.
2) Selects 18 stations with the best temporal coverage.
3) For each station, builds a fractional change series ΔR relative to a reference
   baseline (default: the first 3 days of the entire span; optionally per-event).
4) Assigns an effective primary rigidity P_eff to each station from cutoff_R via a
   simple mapping (heuristic, tunable).
5) For each hour t, fits across stations the relation
         log(1 + ΔR_d(t)) = a0(t) + γ0(t) * log(P_eff_d / P_r),
   where d indexes stations. This recovers two hourly parameters:
     - A0(t) = exp(a0(t)) - 1  → an estimate of ΔI at P_r
     - γ0(t)                    → the rigidity slope of the isotropic term
6) Converts γ0(t) into an approximate spectral-index change at P_r via
         ΔΓ_est(t) = [A0(t) * γ0(t)] / [1 + A0(t)]
   (following the paper's derivative definition for log[1 + ΔI(P,t)] at P = P_r).
7) Produces hourly series and 24 h rolling averages, and (optionally) overlays
   with AMS ΔΓ@10.5 GV daily values if available.

Important limitations (read this!)
----------------------------------
- This is an **isotropic-only** prototype. It does **not** solve for 1st/2nd-order
  directional anisotropy as the full GFA does (no asymptotic directions or yield
  integration over solid angle). That requires additional metadata and tracing.
- The mapping cutoff_R → P_eff is **heuristic**. We default to
      P_eff = 11.3 + k_slope * max(0, cutoff_R), with k_slope = 0.7
  so South Pole (very low cutoff) is near 11.3 GV (as in the paper), and higher
  cutoff stations get larger P_eff. You can tune k_slope.
- Reference baseline is global (first 3 days of the entire span). The paper
  uses a 3-day reference per event window for their plots. You can switch to
  per-window baselines if you provide an event-window CSV.

Inputs
------
- NM hourly counts (wide table):
    
    /mnt/data/Hourly_Count_Rate_All_Stations_Clean - 副本.csv

  Expected columns: a datetime column (named 'datetime' or similar) and one
  column per station named like '<STATION>_count_rate'.

- Station metadata (at least station code and cutoff):
    
    /mnt/data/station_names_all.csv

  Expected columns: 'station_name', 'cutoff_R' (GV).

- (Optional) AMS daily series already computed at 10.5 GV:
    
    /mnt/data/ams_10p5_daily_timeseries.csv

  Expected columns: 'date', 'DeltaGamma_10p5'.

Outputs
-------
- Hourly GFA-lite estimates:
    /mnt/data/nm_gfa_lite_hourly.csv

  Columns: time_utc, DeltaI_Pr, Gamma_eff, DeltaGamma_Pr, (24 h smoothed versions),
           and n_stations used per hour.

- Plots:
    /mnt/data/nm_gfa_lite_deltaI_hourly.png
    /mnt/data/nm_gfa_lite_deltaGamma_hourly.png
    /mnt/data/nm_gfa_lite_vs_ams_deltaGamma.png   (if AMS file is present)

How to run
----------
  python gfa_lite_nm18.py

You can tweak the config section below to change file paths and parameters.

"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

NM_PATH = "/home/zpw/Files/AMS_NeutronMonitors/NM_v0829/NM_0_preprocess_v250220/Hourly_Count_Rate_All_Stations_Clean - 副本.csv"  # NM hourly counts (wide)
ST_META_PATH = "/home/zpw/Files/AMS_NeutronMonitors/NM_v0829/NM_0_preprocess_v250220/station_names_all.csv"                         # station_name, cutoff_R
AMS_SERIES_PATH = "/home/zpw/Files/lstmPaper/data/raw_data/ams/proton.csv"             # optional (from prior step)

# Outputs
OUT_HOURLY = "results/nm_gfa_lite_hourly.csv"
OUT_PLOT_DIA = "results/nm_gfa_lite_deltaI_hourly.png"
OUT_PLOT_GAM = "results/nm_gfa_lite_deltaGamma_hourly.png"
OUT_PLOT_OVERLAY = "results/nm_gfa_lite_vs_ams_deltaGamma.png"

# Key parameters
P_r = 10.5             # GV (paper's reference rigidity)
REF_DAYS = 3           # reference baseline length (global). Use per-window for paper-like plots
N_STATIONS = 18        # select this many stations by coverage
K_SLOPE = 0.7          # cutoff_R → P_eff mapping slope: P_eff = 11.3 + K_SLOPE * max(0, cutoff_R)
MIN_STATIONS_HOUR = 8  # minimum number of stations to solve a given hour robustly
ROLL_HOURS = 24        # rolling mean window for display

# -----------------------------
# Utilities
# -----------------------------

def _find_datetime_column(df: pd.DataFrame) -> str:
    if "datetime" in df.columns:
        return "datetime"
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    raise ValueError("No datetime-like column found in NM file.")


def _station_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_count_rate")]


def select_top_stations_by_coverage(nm_wide: pd.DataFrame, stations_meta: pd.DataFrame, n: int) -> List[str]:
    count_cols = _station_columns(nm_wide)
    coverage = {c.replace("_count_rate", ""): nm_wide[c].notna().sum() for c in count_cols}
    # keep only those that exist in meta (have cutoff)
    allowed = set(stations_meta["station_name"].astype(str))
    candidates = [s for s in sorted(coverage, key=lambda k: coverage[k], reverse=True) if s in allowed]
    return candidates[:n]


def melt_nm_long(nm_wide: pd.DataFrame, station_list: List[str], time_col: str) -> pd.DataFrame:
    use_cols = [time_col] + [f"{st}_count_rate" for st in station_list]
    nm_sel = nm_wide[use_cols].copy()
    nm_long = nm_sel.melt(id_vars=time_col, var_name="station_col", value_name="count_rate")
    nm_long["station"] = nm_long["station_col"].str.replace("_count_rate", "", regex=False)
    nm_long = nm_long.rename(columns={time_col: "datetime"})
    nm_long["datetime"] = pd.to_datetime(nm_long["datetime"], utc=True, errors="coerce")
    return nm_long


def add_cutoff_and_baseline(nm_long: pd.DataFrame, stations_meta: pd.DataFrame, ref_days: int) -> pd.DataFrame:
    meta = stations_meta.rename(columns={"station_name": "station", "cutoff_R": "cutoff_R"}).copy()
    nm_long = nm_long.merge(meta, on="station", how="left")

    first_dt = nm_long["datetime"].min().floor("D")
    baseline_end = first_dt + pd.Timedelta(days=ref_days)
    base = nm_long[(nm_long["datetime"] >= first_dt) & (nm_long["datetime"] < baseline_end)]
    base_mean = base.groupby("station")["count_rate"].mean().rename("baseline_rate").reset_index()
    nm_long = nm_long.merge(base_mean, on="station", how="left")

    nm_long["DeltaR"] = (nm_long["count_rate"] - nm_long["baseline_rate"]) / nm_long["baseline_rate"]
    return nm_long


def assign_peff_from_cutoff(nm_long: pd.DataFrame, k_slope: float, base_Pr_lowcut: float = 11.3) -> pd.DataFrame:
    # Heuristic mapping: P_eff = 11.3 + k * max(0, cutoff_R)
    nm_long["P_eff_GV"] = base_Pr_lowcut + k_slope * np.clip(nm_long["cutoff_R"].values, 0.0, None)
    return nm_long


def fit_hour_group(group: pd.DataFrame, P_r: float, min_stations: int) -> pd.Series:
    df = group.dropna(subset=["DeltaR", "P_eff_GV"]).copy()
    df = df[df["DeltaR"] > -0.9]  # avoid log negative
    if df["station"].nunique() < min_stations:
        return pd.Series({
            "A0": np.nan,
            "gamma0": np.nan,
            "DeltaI_Pr": np.nan,
            "Gamma_eff": np.nan,
            "n_stations": int(df["station"].nunique()),
        })

    x = np.log(df["P_eff_GV"].values / P_r)
    y = np.log(1.0 + df["DeltaR"].values)
    X = np.vstack([np.ones_like(x), x]).T

    # First OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    mask = np.abs(resid) <= 3.0 * mad

    if mask.sum() >= min_stations:
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)

    a0, g0 = float(beta[0]), float(beta[1])
    A0 = math.exp(a0) - 1.0
    gamma_eff = (A0 * g0) / (1.0 + A0)  # d/dlogP ln[1+ΔI] at P_r

    return pd.Series({
        "A0": A0,
        "gamma0": g0,
        "DeltaI_Pr": A0,
        "Gamma_eff": gamma_eff,
        "n_stations": int(df["station"].nunique()),
    })


def build_hourly_series(nm_long: pd.DataFrame, P_r: float, min_st: int, roll_hours: int, ref_days: int) -> pd.DataFrame:
    hourly = nm_long.groupby(nm_long["datetime"]).apply(lambda g: fit_hour_group(g, P_r, min_st)).reset_index().rename(columns={"datetime": "time_utc"})

    # Baseline for Gamma_eff (first ref_days)
    first_hour = hourly["time_utc"].min().floor("D")
    mask0 = (hourly["time_utc"] >= first_hour) & (hourly["time_utc"] < first_hour + pd.Timedelta(days=ref_days))
    Gamma0 = hourly.loc[mask0, "Gamma_eff"].mean()
    hourly["DeltaGamma_Pr"] = hourly["Gamma_eff"] - Gamma0

    # Rolling means (centered)
    hourly = hourly.sort_values("time_utc").reset_index(drop=True)
    hourly["DeltaI_Pr_24h"] = hourly["DeltaI_Pr"].rolling(window=roll_hours, min_periods=roll_hours//2, center=True).mean()
    hourly["DeltaGamma_Pr_24h"] = hourly["DeltaGamma_Pr"].rolling(window=roll_hours, min_periods=roll_hours//2, center=True).mean()
    return hourly


def plot_series(hourly: pd.DataFrame, out_dI: str, out_dG: str) -> None:
    plt.figure(figsize=(12, 4.5))
    plt.plot(hourly["time_utc"], hourly["DeltaI_Pr"], label="ΔI(Pr=10.5 GV) hourly", alpha=0.5)
    plt.plot(hourly["time_utc"], hourly["DeltaI_Pr_24h"], label="ΔI(Pr) 24h mean", linewidth=2)
    plt.title("GFA-lite (NM-only, 18 stations): ΔI at 10.5 GV")
    plt.xlabel("Time (UTC)")
    plt.ylabel("ΔI (fraction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dI)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(hourly["time_utc"], hourly["DeltaGamma_Pr"], label="ΔΓ(Pr) hourly", alpha=0.5)
    plt.plot(hourly["time_utc"], hourly["DeltaGamma_Pr_24h"], label="ΔΓ(Pr) 24h mean", linewidth=2)
    plt.title("GFA-lite (NM-only, 18 stations): ΔΓ at 10.5 GV (approx.)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("ΔΓ (approx)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dG)
    plt.close()


def overlay_with_ams(hourly: pd.DataFrame, ams_series_path: str, out_path: str) -> bool:
    if not os.path.exists(ams_series_path):
        return False
    ams = pd.read_csv(ams_series_path)
    if "date" not in ams.columns or "DeltaGamma_10p5" not in ams.columns:
        return False
    ams["date"] = pd.to_datetime(ams["date"], utc=True)
    ams = ams[["date", "DeltaGamma_10p5"]].dropna()

    nm_daily = hourly.copy()
    nm_daily["date"] = nm_daily["time_utc"].dt.floor("D")
    nm_daily = nm_daily.groupby("date", as_index=False).agg(NM_DeltaGamma_daily=("DeltaGamma_Pr", "mean"))

    merged = ams.merge(nm_daily, on="date", how="inner")
    if merged.empty:
        return False

    plt.figure(figsize=(12, 4.5))
    plt.plot(merged["date"], merged["NM_DeltaGamma_daily"], label="NM GFA-lite ΔΓ (daily mean)", linewidth=2)
    plt.scatter(merged["date"], merged["DeltaGamma_10p5"], label="AMS ΔΓ @10.5 GV (daily)", s=14)
    plt.title("ΔΓ@10.5 GV: NM GFA-lite (daily mean) vs AMS (daily)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("ΔΓ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


# -----------------------------
# Event-window baseline + AMS overlay helpers
# -----------------------------

def build_ams_series_if_missing(ams_raw_path: str, out_series_path: str, p_target: float, ref_days: int) -> Optional[pd.DataFrame]:
    """Build AMS daily ΔΓ@10.5 GV if a precomputed series is missing.
    Returns the loaded/built DataFrame.
    """
    if os.path.exists(out_series_path):
        try:
            df = pd.read_csv(out_series_path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            return df
        except Exception:
            pass
    if not os.path.exists(ams_raw_path):
        print(f"[INFO] AMS raw not found: {ams_raw_path}; AMS overlay will be skipped.")
        return None

    # Minimal builder (mirrors the earlier AMS step)
    ams = pd.read_csv(ams_raw_path)
    ams = ams.rename(columns={
        "date YYYY-MM-DD": "date",
        "rigidity_min GV": "Pmin_GV",
        "rigidity_max GV": "Pmax_GV",
        "proton_flux m^-2sr^-1s^-1GV^-1": "flux",
        "proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1": "flux_sys_err",
    })
    ams["date"] = pd.to_datetime(ams["date"], utc=True, errors="coerce")
    ams["P_GV"] = np.sqrt(ams["Pmin_GV"] * ams["Pmax_GV"])

    def _local_gamma(grp: pd.DataFrame, p_target: float) -> pd.Series:
        g = grp.sort_values("P_GV").reset_index(drop=True)
        idx = (g["P_GV"] - p_target).abs().idxmin()
        start = max(0, idx - 3); end = min(len(g), idx + 4)
        w = g.iloc[start:end].copy()
        if len(w) < 5:
            return pd.Series({"Gamma_10p5": np.nan, "Flux_10p5": np.nan, "P_10p5": np.nan})
        x = np.log(w["P_GV"].values); y = np.log(w["flux"].values)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = w["flux_sys_err"].values / w["flux"].values
            ww = 1.0 / np.where(np.isfinite(rel) & (rel > 0), rel**2, 1.0)
        X = np.vstack([np.ones_like(x), x]).T
        try:
            beta = np.linalg.inv(X.T @ (ww[:,None]*X)) @ (X.T @ (ww * y))
            b = beta[1]
            gamma = -b
        except np.linalg.LinAlgError:
            gamma = np.nan
        closest = g.loc[idx]
        return pd.Series({
            "Gamma_10p5": float(gamma),
            "Flux_10p5": float(closest["flux"]),
            "P_10p5": float(closest["P_GV"]),
        })

    daily = ams.groupby("date").apply(lambda grp: _local_gamma(grp, p_target)).reset_index()
    first_day = daily["date"].min().normalize()
    mask = (daily["date"] >= first_day) & (daily["date"] < first_day + pd.Timedelta(days=ref_days))
    Gamma0 = daily.loc[mask, "Gamma_10p5"].mean()
    Flux0  = daily.loc[mask, "Flux_10p5"].mean()
    daily["DeltaGamma_10p5"] = daily["Gamma_10p5"] - Gamma0
    daily["DeltaI_10p5"]     = (daily["Flux_10p5"] - Flux0) / Flux0

    daily.to_csv(out_series_path, index=False)
    return daily


def make_event_plot(hourly: pd.DataFrame,
                    ams_daily: Optional[pd.DataFrame],
                    win_start: pd.Timestamp,
                    win_end: pd.Timestamp,
                    out_path: str,
                    roll_hours: int = 24) -> None:
    """Apply a per-window baseline (first 3 days of the window) and overlay AMS points."""
    ref_start = win_start
    ref_end   = win_start + pd.Timedelta(days=3)

    ref_seg = hourly[(hourly["time_utc"] >= ref_start) & (hourly["time_utc"] < ref_end)]
    base_gamma = ref_seg["Gamma_eff"].mean()

    seg = hourly[(hourly["time_utc"] >= win_start) & (hourly["time_utc"] <= win_end)].copy()
    seg["DeltaGamma_Pr_window"] = seg["Gamma_eff"] - base_gamma
    seg["DeltaGamma_Pr_window_24h"] = seg["DeltaGamma_Pr_window"].rolling(window=roll_hours, min_periods=roll_hours//2, center=True).mean()

    plt.figure(figsize=(10.5, 4.2))
    plt.plot(seg["time_utc"], seg["DeltaGamma_Pr_window"], label="GFA-lite ΔΓ(Pr) hourly", alpha=0.45)
    plt.plot(seg["time_utc"], seg["DeltaGamma_Pr_window_24h"], label="GFA-lite ΔΓ(Pr) 24h mean", linewidth=2.2)

    if ams_daily is not None and {"date","DeltaGamma_10p5"}.issubset(ams_daily.columns):
        ams_win = ams_daily[(ams_daily["date"] >= win_start) & (ams_daily["date"] <= win_end)]
        if not ams_win.empty:
            plt.scatter(ams_win["date"], ams_win["DeltaGamma_10p5"], s=18, label="AMS ΔΓ @ 10.5 GV (daily)", zorder=3)

    plt.title("ΔΓ at 10.5 GV — Event window 2015-03-11 to 2015-04-07 (baseline: first 3 days)")
    plt.xlabel("Time (UTC)"); plt.ylabel("ΔΓ")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    # 1) NM GFA-lite hourly (reuse existing functions above)
    nm_wide = pd.read_csv(NM_PATH)
    st_meta = pd.read_csv(ST_META_PATH)

    time_col = _find_datetime_column(nm_wide)
    stations = select_top_stations_by_coverage(nm_wide, st_meta, N_STATIONS)
    nm_long = melt_nm_long(nm_wide, stations, time_col)
    nm_long = add_cutoff_and_baseline(nm_long, st_meta, REF_DAYS)
    nm_long = assign_peff_from_cutoff(nm_long, K_SLOPE, base_Pr_lowcut=11.3)

    hourly = build_hourly_series(nm_long, P_r=P_r, min_st=MIN_STATIONS_HOUR, roll_hours=ROLL_HOURS, ref_days=REF_DAYS)

    # 2) Build or load AMS daily series
    ams_daily = build_ams_series_if_missing(
        ams_raw_path=AMS_SERIES_PATH,
        out_series_path="results/ams_10p5_daily_timeseries.csv",
        p_target=P_r,
        ref_days=3,
    )

    # 3) Make the requested event plot for 2015-03-11 .. 2015-04-07
    win_start = pd.Timestamp("2015-03-11T00:00:00Z")
    win_end   = pd.Timestamp("2015-04-07T23:59:59Z")
    out_path  = "results/nm_gfa_lite_event_2015Mar_deltaGamma.png"

    make_event_plot(hourly, ams_daily, win_start, win_end, out_path, roll_hours=ROLL_HOURS)
    print(f"[OK] Saved event plot: {out_path}")


if __name__ == "__main__":
    main()

