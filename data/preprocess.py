# -*- coding: utf-8 -*-
"""
预处理 AMS 和 OMNI 数据（分开保存）
FD 事件: 2015-06-19 ~ 2015-07-07

修改点：
1. AMS 和 OMNI 分别保存为独立文件
2. AMS 列名去除空格（刚度区间）
3. 保存 I, I_err, I_rel, I_rel_err, dI, dI_err（每个刚度6列）
4. 打印每个刚度区间的 baseline_mean 用于验证
5. OMNI 不做异常值处理，保持原始数据
6. 修复 DataFrame 碎片化警告
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- 路径配置 ----------
AMS_PATH  = "ams/flux_long.csv"
OMNI_PATH = "omni/omni2_2011_2025.csv"

# ---------- 事件窗口 ----------
t_start = pd.Timestamp("2021-11-01 00:00:00")
t_end   = pd.Timestamp("2021-11-08 23:00:00")
PRE_HOURS = 48  # 基线缓冲时长

# ---------- 创建输出文件夹 ----------
import os
event_date = t_start.strftime("%Y%m%d")
output_dir = f"FD_{event_date}"
os.makedirs(output_dir, exist_ok=True)

OUT_AMS   = os.path.join(output_dir, "ams_processed.csv")
OUT_OMNI  = os.path.join(output_dir, "omni_processed.csv")

print(f"事件窗口: {t_start} 至 {t_end}")
print(f"基线窗口: {t_start - timedelta(hours=PRE_HOURS)} 至 {t_start}")
print("=" * 60)

# ---------- 加载数据 ----------
ams = pd.read_csv(AMS_PATH, parse_dates=["date"])
omni = pd.read_csv(OMNI_PATH)
omni["datetime"] = pd.to_datetime(omni["datetime"], format="%Y/%m/%d/%H", errors="coerce")

# ---------- AMS 数据处理 ----------
# 选择窗口：[t_start - PRE_HOURS, t_end]
ams_win = ams[(ams["date"] >= t_start - timedelta(hours=PRE_HOURS)) & (ams["date"] <= t_end)].copy()

# 打印选择的日期范围
print(f"起点：{t_start - timedelta(hours=PRE_HOURS)} ")
print(f"终点：{t_end} ")
print("-" * 60)
print("\n【AMS 数据选择窗口】")
print("-" * 60)
print(f"首个数据点: {ams_win['date'].min()}")
print(f"末个数据点: {ams_win['date'].max()}")
print(f"总数据点数: {len(ams_win['date'].unique())}")


# 处理每个刚度区间
dfs = []
print("\n【AMS 基线验证】")
print("-" * 60)

for (rmin, rmax), g in ams_win.groupby(["rigidity_min", "rigidity_max"]):
    g = g.sort_values("date").set_index("date")
    
    # 确保小时轴连续
    g = g.resample("1h").interpolate(limit_direction="both")
    
    # 计算基线: [t_start-48h, t_start)
    base_mask = (g.index < t_start) & (g.index >= t_start - timedelta(hours=PRE_HOURS))
    baseline_mean = g.loc[base_mask, "flux"].mean()
    
    # 容错：如果基线为空，使用窗口首24小时
    if not np.isfinite(baseline_mean) or baseline_mean == 0:
        baseline_mean = g.iloc[:24]["flux"].mean()
        print(f"⚠️  刚度 {rmin:.2f}-{rmax:.2f} GV: 基线数据不足，使用前24小时均值")
    
    # 打印基线均值（用于验证）
    print(f"刚度 {rmin:.2f}-{rmax:.2f} GV: baseline_mean = {baseline_mean:.4e}")
    
    # 计算相对强度
    g["I_rel"] = g["flux"] / baseline_mean
    g["dI"] = g["I_rel"] - 1.0
    
    # 误差传播
    if "error_bar" in g.columns:
        g["I_err"] = g["error_bar"]
        g["I_rel_err"] = g["error_bar"] / baseline_mean
        g["dI_err"] = g["error_bar"] / baseline_mean
    else:
        g["I_err"] = np.nan
        g["I_rel_err"] = np.nan
        g["dI_err"] = np.nan
    
    # 刚度列名（无空格）
    rigidity_label = f"{rmin:.2f}-{rmax:.2f}GV"
    g["rigidity_bin"] = rigidity_label
    
    dfs.append(g[["flux", "I_err", "I_rel", "I_rel_err", "dI", "dI_err", "rigidity_bin"]].reset_index())

ams_rel = pd.concat(dfs, ignore_index=True)

# 只保留分析窗口 [t_start, t_end]
ams_rel = ams_rel[(ams_rel["date"] >= t_start) & (ams_rel["date"] <= t_end)].copy()

# 转换为宽表格式
I_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="flux")
I_err_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="I_err")
I_rel_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="I_rel")
I_rel_err_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="I_rel_err")
dI_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="dI")
dI_err_wide = ams_rel.pivot(index="date", columns="rigidity_bin", values="dI_err")

# 使用 pd.concat 一次性合并所有列（避免碎片化警告）
all_dfs = []
for col in sorted(I_wide.columns):
    temp_df = pd.DataFrame({
        f"I_{col}": I_wide[col],
        f"I_{col}_err": I_err_wide[col],
        f"I_rel_{col}": I_rel_wide[col],
        f"I_rel_{col}_err": I_rel_err_wide[col],
        f"dI_{col}": dI_wide[col],
        f"dI_{col}_err": dI_err_wide[col]
    })
    all_dfs.append(temp_df)

ams_output = pd.concat(all_dfs, axis=1)
ams_output.index.name = "datetime"

# 保存 AMS
ams_output.to_csv(OUT_AMS, index=True)
print(f"\n[✓] AMS 数据已保存: {OUT_AMS}")
print(f"    形状: {ams_output.shape}")
print(f"    每个刚度区间包含6列: I, I_err, I_rel, I_rel_err, dI, dI_err")

# ---------- OMNI 数据处理 ----------
# 选择窗口: [t_start, t_end]，不做异常值处理
omni_win = omni[(omni["datetime"] >= t_start) & (omni["datetime"] <= t_end)].copy()

print("\n【OMNI 数据】")
print("-" * 60)
print("保持原始数据，未进行异常值处理")

# 保存 OMNI
omni_win.to_csv(OUT_OMNI, index=False)
print(f"\n[✓] OMNI 数据已保存: {OUT_OMNI}")
print(f"    形状: {omni_win.shape}")

print("\n" + "=" * 60)
print("预处理完成！")
print(f"  输出文件夹: {output_dir}/")
print(f"    - ams_processed.csv  (每个刚度6列)")
print(f"    - omni_processed.csv (原始数据)")