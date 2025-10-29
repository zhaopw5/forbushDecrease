#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Very simple script:
Read an hourly flux CSV (wide format), add a 'date' column,
and convert it to a long-format table with columns:
date, rigidity_min, rigidity_max, flux, error_bar

Edit the variables in the "USER SETTINGS" section and run.
"""

import numpy as np
import pandas as pd

# =========================
# ===== USER SETTINGS =====
# =========================
FLUX_FILE = "STL_2011-01-01-2024-07-31_hourly_FDs_pred_ams.csv"   # input hourly flux CSV
ERROR_FILE = "maximum_error_data.csv"  # or None if you don't want error_bar
OUT_CSV = "flux_long.csv"              # output CSV path

START_DATETIME = "2011-01-01 00:00:00" # first timestamp of your hourly data

# Optional time window; set to None to disable
TIME_MIN = None  # e.g., "2015-03-10 00:00:00" or None
TIME_MAX = None  # e.g., "2015-03-24 23:00:00" or None

# Rigidity bin edges (GV) — same as your plotting script
RIG_BIN_EDGES = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,
                 3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,
                 8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

# =========================
# ======= MAIN CODE =======
# =========================

# 1) Read flux file (wide format)
flux_df = pd.read_csv(FLUX_FILE, header=0)
n_rows, n_cols = flux_df.shape

# 2) Use the first 30 columns as flux bins
expected_bins = len(RIG_BIN_EDGES) - 1  # 30
if n_cols < expected_bins:
    raise ValueError(f"Flux file has {n_cols} columns, needs at least {expected_bins}.")
flux_bins = flux_df.iloc[:, :expected_bins].copy()

# 3) Build the hourly 'date' column
date_index = pd.date_range(start=pd.Timestamp(START_DATETIME), periods=n_rows, freq="H")
flux_bins.insert(0, "date", date_index.astype("datetime64[ns]"))

# 4) Optional window filter
if TIME_MIN is not None:
    tmin = pd.Timestamp(TIME_MIN)
    flux_bins = flux_bins[flux_bins["date"] >= tmin]
if TIME_MAX is not None:
    tmax = pd.Timestamp(TIME_MAX)
    flux_bins = flux_bins[flux_bins["date"] <= tmax]

# 5) Rigidity arrays
rig_min = np.array(RIG_BIN_EDGES[:-1], dtype=float)
rig_max = np.array(RIG_BIN_EDGES[1:], dtype=float)

# 6) Load relative error per bin (assume column is exactly 'std_dev')
if ERROR_FILE is not None:
    err_df = pd.read_csv(ERROR_FILE)
    if "std_dev" not in err_df.columns:
        raise ValueError(f"'std_dev' column not found in {ERROR_FILE}.")
    rel_err = err_df["std_dev"].to_numpy().astype(float)
    if rel_err.shape[0] != expected_bins:
        raise ValueError(f"Length of std_dev = {rel_err.shape[0]} != {expected_bins}.")
else:
    rel_err = np.full(expected_bins, np.nan, dtype=float)

# 7) Build long-format table
long_parts = []
for i in range(expected_bins):
    flux_values = flux_bins.iloc[:, i + 1].to_numpy(dtype=float)  # +1: col 0 is 'date'
    part = pd.DataFrame({
        "date": flux_bins["date"].values,
        "rigidity_min": rig_min[i],
        "rigidity_max": rig_max[i],
        "flux": flux_values,
    })
    part["error_bar"] = flux_values * rel_err[i]
    long_parts.append(part)

long_df = pd.concat(long_parts, axis=0, ignore_index=True)

# 8) Sort and keep required column order
long_df.sort_values(by=["date", "rigidity_min"], inplace=True, kind="mergesort")
long_df = long_df[["date", "rigidity_min", "rigidity_max", "flux", "error_bar"]]

# 9) Save
long_df.to_csv(OUT_CSV, index=False)
print(f"Saved long-format table with {len(long_df):,} rows to: {OUT_CSV}")
