# -*- coding: utf-8 -*-
"""
flux_t_rig.py

功能：
1. 绘制 flux~time 图（选定3个代表性刚度 + 6个OMNI参数，含误差棒）
2. 绘制 flux~rigidity 图（每日能谱，每条曲线带 flux 误差和 rigidity 误差）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# ====================== 用户参数 ======================
AMS_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20150619/ams_processed.csv"
OMNI_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20150619/omni_processed.csv"
OUT_DIR = "fig_flux"

# AMS 选用的三个代表性刚度区间（对应通量列名 I_***）
SELECTED_RIGS = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

# 选用的 OMNI 参数（来自你的筛选结果）
OMNI_COLS = ["Sunspot_R", "Vsw", "F10_7", "Dst", "alpha_to_proton", "pf_gt10MeV"]

# ====================== 读入与预处理 ======================
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 读 AMS ----
ams = pd.read_csv(AMS_PATH, parse_dates=["datetime"]).set_index("datetime")

# ---- 读 OMNI ----
omni = pd.read_csv(OMNI_PATH, parse_dates=["datetime"]).set_index("datetime")

# ---- 时间对齐 ----
common = ams.index.intersection(omni.index)
ams = ams.loc[common]
omni = omni.loc[common]

print(f"[INFO] AMS时间范围: {ams.index.min()} ~ {ams.index.max()} ({len(ams)}点)")

# ====================== 1. flux~time 图 ======================

nrows = len(SELECTED_RIGS) + len(OMNI_COLS)
fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 2*nrows), sharex=True)
fig.suptitle("AMS Proton Flux & OMNI Parameters vs Time", fontsize=14, y=0.93)

# ---- (1) AMS flux 部分 ----
for i, rig in enumerate(SELECTED_RIGS):
    flux_col = f"I_{rig}"
    err_col  = f"I_{rig}_err"
    if flux_col not in ams.columns or err_col not in ams.columns:
        print(f"[WARN] 缺少 {rig} 的通量或误差列，跳过。")
        continue

    ax = axes[i]
    ax.errorbar(
        ams.index, ams[flux_col], yerr=ams[err_col],
        fmt="o", markersize=2, elinewidth=0.8, capsize=2,
        label=f"{rig}"
    )
    ax.set_ylabel(f"{rig}\nFlux", fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

# ---- (2) OMNI 部分 ----
for j, col in enumerate(OMNI_COLS):
    ax = axes[len(SELECTED_RIGS) + j]
    if col not in omni.columns:
        print(f"[WARN] 缺少 {col} 列。")
        continue
    ax.plot(omni.index, omni[col], lw=1.0, label=col)
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

# ---- x轴格式 ----
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axes[-1].set_xlabel("Date (UTC)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(OUT_DIR, "flux_vs_time.png")
plt.savefig(out1, dpi=300)
plt.close()
print(f"[OK] 保存 flux~time 图：{out1}")

# ====================== 2. flux~rigidity 图 ======================

# ---- 取物理通量列（不含相对量/误差/ΔI）----
flux_cols = [
    c for c in ams.columns
    if c.startswith("I_") and ("rel" not in c) and ("dI" not in c) and ("err" not in c)
]
# 对应误差列
err_cols = [c + "_err" for c in flux_cols]

# ---- 计算刚度中心与误差 ----
rigidity_centers = []
rigidity_errs = []
for c in flux_cols:
    part = c.replace("I_", "").replace("GV", "")
    rmin, rmax = map(float, part.split("-"))
    rigidity_centers.append((rmin + rmax) / 2)
    rigidity_errs.append((rmax - rmin) / 2)
rigidity_centers = np.array(rigidity_centers)
rigidity_errs = np.array(rigidity_errs)

# ---- 按天平均 ----
ams_daily = ams.resample("1D").mean()

plt.figure(figsize=(8,5))
for date, row in ams_daily.iterrows():
    flux = row[flux_cols].values
    flux_err = row[err_cols].values if all(c in row.index for c in err_cols) else np.zeros_like(flux)
    if np.isnan(flux).all():
        continue
    plt.errorbar(
        rigidity_centers, flux, xerr=rigidity_errs, yerr=flux_err,
        fmt="-o", lw=1, markersize=3, capsize=2,
        label=date.strftime("%m-%d")
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rigidity [GV]")
plt.ylabel("Flux")
plt.title("Daily Proton Flux Spectra (with Rigidity & Flux Errors)")
plt.legend(ncol=3, fontsize=8, frameon=False)
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "flux_vs_rigidity.png")
plt.savefig(out2, dpi=300)
plt.close()
print(f"[OK] 保存 flux~rigidity 图：{out2}")

print(f"\n✅ 全部绘图完成，结果保存在 {OUT_DIR}/")
