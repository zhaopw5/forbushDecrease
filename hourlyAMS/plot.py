#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot flux ± error_bar for a specified rigidity bin from a long-format CSV.

Input CSV must have columns:
  date, rigidity_min, rigidity_max, flux, error_bar

The script:
  1) filters rows by (rigidity_min, rigidity_max)
  2) optionally clips by a time window
  3) optionally normalizes by a baseline (mean/median/first)
  4) optionally resamples (e.g., daily) to reduce visual clutter
  5) plots flux as a line and ±error as a shaded band

Edit the USER SETTINGS section and run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# ===== USER SETTINGS =====
# =========================
INPUT_CSV = "flux_long.csv"     # produced by your previous script
RIGIDITY_MIN = 1.51             # GV
RIGIDITY_MAX = 1.71             # GV

# Optional time window; set to None to disable
TIME_MIN = "2015-03-10 00:00:00"                 # e.g., "2015-03-10 00:00:00" or None
TIME_MAX = "2015-03-24 23:00:00"                 # e.g., "2015-03-24 23:00:00" or None

# Normalization options
NORMALIZE = False               # True/False
NORM_BASELINE = "mean"          # "mean" | "median" | "first"

# Optional resample to reduce points (None to disable). Examples: "D" (daily), "W" (weekly).
RESAMPLE = None                 # e.g., "D" or None

# Output
OUT_DIR = "plots_flux_error"
OUT_NAME = f"flux_error_{RIGIDITY_MIN}-{RIGIDITY_MAX}GV"

# =========================
# ======= MAIN CODE =======
# =========================

# 1) Load data
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

# 2) Filter by rigidity bin (use np.isclose for safety)
tol = 1e-8
mask_bin = np.isclose(df["rigidity_min"], RIGIDITY_MIN, atol=tol) & \
           np.isclose(df["rigidity_max"], RIGIDITY_MAX, atol=tol)
sub = df.loc[mask_bin].copy()

if sub.empty:
    raise ValueError(f"No rows found for bin [{RIGIDITY_MIN}, {RIGIDITY_MAX}] GV in {INPUT_CSV}.")

# 3) Optional time window
if TIME_MIN is not None:
    sub = sub[sub["date"] >= pd.Timestamp(TIME_MIN)]
if TIME_MAX is not None:
    sub = sub[sub["date"] <= pd.Timestamp(TIME_MAX)]

if sub.empty:
    raise ValueError("No data after applying the time window. Adjust TIME_MIN/TIME_MAX.")

# Ensure sorted by time
sub.sort_values("date", inplace=True)

# 4) Optional resample (simple mean; you can change to other aggregations if needed)
if RESAMPLE is not None:
    sub = (sub
           .set_index("date")
           .resample(RESAMPLE)
           .mean(numeric_only=True)
           .reset_index())
    # After resampling, we lost the original rigidity columns; we can add them back for clarity
    sub["rigidity_min"] = RIGIDITY_MIN
    sub["rigidity_max"] = RIGIDITY_MAX

# 5) Normalization (applies to both flux and error_bar)
if NORMALIZE:
    if NORM_BASELINE == "mean":
        baseline = sub["flux"].mean()
    elif NORM_BASELINE == "median":
        baseline = sub["flux"].median()
    elif NORM_BASELINE == "first":
        baseline = sub["flux"].iloc[0]
    else:
        raise ValueError(f"Unsupported NORM_BASELINE: {NORM_BASELINE}")
    if baseline == 0 or not np.isfinite(baseline):
        raise ValueError("Invalid baseline for normalization.")
    sub["flux"] = sub["flux"] / baseline
    sub["error_bar"] = sub["error_bar"] / baseline
    y_label = "Normalized Flux"
else:
    y_label = r"Proton Flux (m$^{-2}$ sr$^{-1}$ s$^{-1}$ GV$^{-1}$)"

# 6) Compute upper/lower bounds
upper = sub["flux"] + sub["error_bar"]
lower = sub["flux"] - sub["error_bar"]

# 7) Plot
plt.figure(figsize=(12, 6))
plt.errorbar(sub["date"], sub["flux"], yerr=sub["error_bar"], fmt='o', linewidth=1.2, label="Flux")
# plt.fill_between(sub["date"], lower, upper, alpha=0.25, label="± error_bar")

title = f"Flux ± error_bar  [{RIGIDITY_MIN}-{RIGIDITY_MAX}] GV"
plt.title(title)
plt.xlabel("Date")
plt.ylabel(y_label)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

# 8) Save
os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, OUT_NAME + (f"_{RESAMPLE}" if RESAMPLE else "") + ( "_norm" if NORMALIZE else "" ) + ".png")
pdf_path = os.path.join(OUT_DIR, OUT_NAME + (f"_{RESAMPLE}" if RESAMPLE else "") + ( "_norm" if NORMALIZE else "" ) + ".pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.savefig(pdf_path)
print(f"Saved:\n  {png_path}\n  {pdf_path}")
