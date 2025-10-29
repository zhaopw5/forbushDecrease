# -*- coding: utf-8 -*-
"""
Preprocess AMS flux_long.csv + OMNI2 omni2_2011_2025.csv
for the 2015-06-19 ~ 2015-07-07 Forbush Decrease event.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===============================
# 1. 读入 AMS & OMNI 数据
# ===============================

flux = pd.read_csv("ams/flux_long.csv", parse_dates=["date"])
omni = pd.read_csv("omni/omni2_2011_2025.csv", parse_dates=["datetime"])

# ===============================
# 2. 筛选事件时间窗
# ===============================
t_start = pd.Timestamp("2015-06-19 00:00:00")
t_end   = pd.Timestamp("2015-07-07 23:00:00")

flux = flux[(flux["date"] >= t_start) & (flux["date"] <= t_end)].copy()
omni = omni[(omni["datetime"] >= t_start) & (omni["datetime"] <= t_end)].copy()

# ===============================
# 3. 太阳风数据清洗 (3σ 裁剪 + 插值)
# ===============================
numeric_cols = omni.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    series = omni[col]
    mu, sigma = series.mean(skipna=True), series.std(skipna=True)
    mask = (series < mu - 3*sigma) | (series > mu + 3*sigma)
    omni.loc[mask, col] = np.nan
# 短缺口 (≤2h) 插值
omni[numeric_cols] = omni[numeric_cols].interpolate(limit=2, limit_direction="both")

# ===============================
# 4. AMS 数据处理：计算相对通量 ΔI
# ===============================
# 分组按 rigidity_min, rigidity_max
dfs = []
for (rmin, rmax), group in flux.groupby(["rigidity_min", "rigidity_max"]):
    group = group.sort_values("date").set_index("date")
    # 用线性插值确保小时连续
    group = group.resample("1H").interpolate()
    # 计算相对通量：以事件前48小时为基线
    baseline_window = (group.index < t_start) & (group.index >= t_start - timedelta(hours=48))
    baseline_mean = group.loc[baseline_window, "flux"].mean()
    group["I_rel"] = group["flux"] / baseline_mean
    group["dI"] = group["I_rel"] - 1
    group["rigidity_bin"] = f"{rmin:.2f}-{rmax:.2f} GV"
    dfs.append(group.reset_index())

ams_rel = pd.concat(dfs, ignore_index=True)

# ===============================
# 5. AMS 与 OMNI 对齐
# ===============================
# 先展开 AMS 为宽表（每刚度一列）
ams_pivot = ams_rel.pivot(index="date", columns="rigidity_bin", values="dI")
ams_pivot = ams_pivot.reindex(omni["datetime"])  # 对齐时间轴
ams_pivot.index.name = "datetime"

# 合并
merged = pd.concat([omni.set_index("datetime"), ams_pivot], axis=1)

# ===============================
# 6. 保存结果
# ===============================
merged.to_csv("merged_FD_20150619_20150707.csv")
print(f"Saved merged_FD_20150619_20150707.csv with shape {merged.shape}")
