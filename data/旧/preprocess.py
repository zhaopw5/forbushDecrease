# -*- coding: utf-8 -*-
"""
Preprocess AMS hourly proton flux (flux_long.csv) and OMNI (omni2_2011_2025.csv)
for the FD case: 2015-06-19 ~ 2015-07-07.

Key fixes vs previous version:
- DO NOT cut AMS at [t_start, t_end] before baseline.
- Always load AMS with a pre-onset buffer (>= 48h), then compute baseline in [t_start-48h, t_start).
- Propagate AMS errors: dI_err = error_bar / baseline_mean (per rigidity bin).
- Join ΔI and its error into output; keep OMNI cleaned (3σ clip + ≤2h interpolate).
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- paths ----------
AMS_PATH  = "ams/flux_long.csv"
OMNI_PATH = "omni/omni2_2011_2025.csv"
OUT_CSV   = "merged_FD_20150619_20150707.csv"

# ---------- event window ----------
t_start = pd.Timestamp("2015-06-19 00:00:00")
t_end   = pd.Timestamp("2015-07-07 23:00:00")
PRE_HOURS = 48  # baseline buffer

# ---------- load ----------
# AMS
ams = pd.read_csv(AMS_PATH, parse_dates=["date"])
# OMNI
omni = pd.read_csv(OMNI_PATH)
omni["datetime"] = pd.to_datetime(omni["datetime"], format="%Y/%m/%d/%H", errors="coerce")

# ---------- select windows ----------
# AMS: keep [t_start - PRE_HOURS, t_end]  (so baseline exists!)
ams_win  = ams[(ams["date"] >= t_start - timedelta(hours=PRE_HOURS)) & (ams["date"] <= t_end)].copy()
# OMNI: only [t_start, t_end] for alignment (baseline只用于AMS)
omni_win = omni[(omni["datetime"] >= t_start) & (omni["datetime"] <= t_end)].copy()

# ---------- OMNI cleaning: 3σ clip + ≤2h interpolate ----------
num_cols = omni_win.select_dtypes(include=[np.number]).columns
for col in num_cols:
    s = omni_win[col]
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    outlier = (s < mu - 3*sd) | (s > mu + 3*sd)
    omni_win.loc[outlier, col] = np.nan
# ≤2h线性插值；两端也补≤2h的缺口
omni_win[num_cols] = omni_win[num_cols].interpolate(limit=2, limit_direction="both")

# ---------- AMS: relative intensity ΔI per rigidity bin with error ----------
dfs = []
for (rmin, rmax), g in ams_win.groupby(["rigidity_min", "rigidity_max"]):
    g = g.sort_values("date").set_index("date")
    # 保证小时轴连续
    g = g.resample("1H").interpolate(limit_direction="both")
    # baseline: [t_start-48h, t_start)
    base_mask = (g.index < t_start) & (g.index >= t_start - timedelta(hours=PRE_HOURS))
    baseline_mean = g.loc[base_mask, "flux"].mean()
    # 如果基线为空（极端情况），退化为窗口首24h的均值
    if not np.isfinite(baseline_mean) or baseline_mean == 0:
        baseline_mean = g.iloc[:24]["flux"].mean()

    g["I_rel"] = g["flux"] / baseline_mean
    g["dI"]    = g["I_rel"] - 1.0
    # 误差传播（相对误差）：err_rel = error_bar / baseline
    if "error_bar" in g.columns:
        g["dI_err"] = g["error_bar"] / baseline_mean
    else:
        g["dI_err"] = np.nan

    g["rigidity_bin"] = f"{rmin:.2f}-{rmax:.2f} GV"
    dfs.append(g[["I_rel", "dI", "dI_err", "rigidity_bin"]].reset_index())

ams_rel = pd.concat(dfs, ignore_index=True)

# 只保留分析窗口 [t_start, t_end]
ams_rel = ams_rel[(ams_rel["date"] >= t_start) & (ams_rel["date"] <= t_end)].copy()

# 宽表：ΔI 与 ΔI_err
dI_wide     = ams_rel.pivot(index="date", columns="rigidity_bin", values="dI").sort_index()
dIerr_wide  = ams_rel.pivot(index="date", columns="rigidity_bin", values="dI_err").sort_index()
dI_wide.index.name = "datetime"

# ---------- align & merge ----------
merged = omni_win.set_index("datetime").join(dI_wide, how="left")

# 把误差也拼接，后缀 _err
for col in dI_wide.columns:
    merged[col + "_err"] = dIerr_wide[col].reindex(merged.index)

# ---------- save ----------
merged.to_csv(OUT_CSV, index=True)
print(f"[OK] saved {OUT_CSV}  shape={merged.shape}")
