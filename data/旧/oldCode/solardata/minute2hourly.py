#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make an hourly-averaged OMNI file from the minute-level CSV.

Input CSV must include a 'datetime' column (created by your previous script).
This script:
  - parses 'datetime' and sets it as index
  - computes hourly mean for all numeric columns
  - adds per-hour sample counts and a simple "valid_ratio" (count / 60)
  - saves to OUTPUT_HOURLY_CSV

Edit the USER SETTINGS section and run.
"""

import pandas as pd
import numpy as np

# =========================
# ===== USER SETTINGS =====
# =========================
INPUT_CSV = "omni_min2015.csv"          # your minute-level CSV
OUTPUT_HOURLY_CSV = "omni_min2015_hourly.csv"

# Optional: a representative column to count valid samples.
# If None, we'll count all rows per hour (df.resample('H').size()).
REPRESENTATIVE_COL = None  # e.g., "V" or None

# =========================
# ======= MAIN CODE =======
# =========================

# 1) Load the minute-level CSV, parse datetime
df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])

# 2) Set datetime as index and sort
df = df.set_index("datetime").sort_index()

# 3) Select numeric columns to average (non-numeric like IDs are excluded)
num_cols = df.select_dtypes(include=[np.number]).columns

# 4) Hourly mean for numeric columns
hourly_mean = df[num_cols].resample("H").mean()

# 5) Per-hour sample counts
if REPRESENTATIVE_COL is not None and REPRESENTATIVE_COL in df.columns:
    # Count non-NaN entries of the chosen column per hour
    hourly_count = df[REPRESENTATIVE_COL].resample("H").count()
else:
    # Count total rows per hour (regardless of NaNs in specific columns)
    hourly_count = df.resample("H").size()

hourly = hourly_mean.copy()
hourly["n_samples"] = hourly_count

# 6) A simple valid ratio: how many minute rows we saw in that hour out of 60
#    (Note: the first/last hour of your file may be partial; that's normal.)
hourly["valid_ratio"] = hourly["n_samples"] / 60.0

# 7) Save
hourly.to_csv(OUTPUT_HOURLY_CSV, float_format="%.6g")
print(f"Saved hourly file to: {OUTPUT_HOURLY_CSV}")
print("Columns included:", ", ".join(hourly.columns))
