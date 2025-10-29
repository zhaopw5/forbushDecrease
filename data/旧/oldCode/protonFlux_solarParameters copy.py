#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-panel plot:
  Row 1: Cosmic-ray flux (selected rigidity bin) with ±error band
  Row 2: Field magnitude average Bz_gse (nT)
  Row 3: Solar wind flow speed V (km/s)

Inputs:
  - OMNI hourly CSV (with 'datetime', 'Bz_gse', 'V'), e.g., omni_min2015_hourly.csv
  - Long-format flux CSV (date, rigidity_min, rigidity_max, flux, error_bar), e.g., flux_long.csv

Edit USER SETTINGS and run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =========================
# ===== USER SETTINGS =====
# =========================
OMNI_HOURLY_CSV = "/home/zpw/Files/forbushDecrease/solardata/omni_min2015_hourly.csv"
FLUX_LONG_CSV   = "/home/zpw/Files/forbushDecrease/hourlyAMS/flux_long.csv"

# Choose rigidity bin to plot
RIGIDITY_MIN = 1.51   # GV
RIGIDITY_MAX = 1.71   # GV

# Optional time window (set None to disable)
TIME_MIN = "2015-03-10 00:00:00"       # e.g., "2015-03-10 00:00:00" or None
TIME_MAX = "2015-03-24 23:00:00"       # e.g., "2015-03-24 23:00:00" or None

# Optional resampling for clearer curves: None, "D" (daily), "W" (weekly)
RESAMPLE = None       # e.g., "D" or None

# Output
OUT_DIR  = "plots"
OUT_NAME = f"flux_Bz_V_{RIGIDITY_MIN}-{RIGIDITY_MAX}GV"

# =========================
# ======= MAIN CODE =======
# =========================

# 1) Load OMNI hourly with Bz_gse and V
omni = pd.read_csv(OMNI_HOURLY_CSV, parse_dates=["datetime"])
need_cols = {"Bz_gse", "V"}
missing = need_cols - set(omni.columns)
if missing:
    raise ValueError(f"Columns {missing} not found in {OMNI_HOURLY_CSV}.")
omni = omni[["datetime", "Bz_gse", "V"]].dropna(subset=["datetime"]).copy()
omni = omni.set_index("datetime").sort_index()

# 2) Load cosmic-ray flux (long format) and select bin
flux_df = pd.read_csv(FLUX_LONG_CSV, parse_dates=["date"])
tol = 1e-8
mask_bin = (np.isclose(flux_df["rigidity_min"], RIGIDITY_MIN, atol=tol) &
            np.isclose(flux_df["rigidity_max"], RIGIDITY_MAX, atol=tol))
sub = flux_df.loc[mask_bin, ["date", "flux", "error_bar"]].copy()
if sub.empty:
    raise ValueError(f"No rows for bin [{RIGIDITY_MIN}, {RIGIDITY_MAX}] GV in {FLUX_LONG_CSV}.")
sub = sub.set_index("date").sort_index()

# 3) Optional time window
if TIME_MIN is not None:
    tmin = pd.Timestamp(TIME_MIN)
    omni = omni[omni.index >= tmin]
    sub  = sub[sub.index >= tmin]
if TIME_MAX is not None:
    tmax = pd.Timestamp(TIME_MAX)
    omni = omni[omni.index <= tmax]
    sub  = sub[sub.index <= tmax]

if omni.empty or sub.empty:
    raise ValueError("No data to plot after applying the time window.")

# 4) Optional resampling (mean)
if RESAMPLE is not None:
    omni_plot = omni.resample(RESAMPLE).mean()
    sub_plot  = sub.resample(RESAMPLE).mean(numeric_only=True)
else:
    omni_plot = omni.copy()
    sub_plot  = sub.copy()

# 5) Optionally align time index by inner join (ensures shared x ticks match nicely)
#    If you prefer not to align (keep original timestamps), comment out next two lines.
aligned = omni_plot.join(sub_plot, how="inner")
if aligned.empty:
    # fallback to independent axes if no overlap after join
    aligned = None

# 6) Prepare data series to plot
if aligned is not None:
    # Use aligned data
    time_idx = aligned.index
    flux_y   = aligned["flux"]
    err_y    = aligned["error_bar"] if "error_bar" in aligned.columns else pd.Series(np.nan, index=time_idx)
    B_y      = aligned["Bz_gse"]
    V_y      = aligned["V"]
else:
    # Use original (un-aligned); plots still work, just x ticks may differ slightly
    time_idx = sub_plot.index
    flux_y   = sub_plot["flux"]
    err_y    = sub_plot.get("error_bar", pd.Series(np.nan, index=time_idx))
    B_y      = omni_plot["Bz_gse"]
    V_y      = omni_plot["V"]

upper = flux_y + err_y
lower = flux_y - err_y


# 设置绘图字体大小：
plt.rcParams.update({
    'font.size': 20,            # 所有字体的默认大小
    'axes.labelsize': 22,       # 坐标轴标签
    'axes.titlesize': 24,       # 子图标题
    'legend.fontsize': 18,      # 图例
    'xtick.labelsize': 20,      # x轴刻度
    'ytick.labelsize': 20,      # y轴刻度
    'figure.titlesize': 26      # 总标题
})

def format_date(dt):
    """将 matplotlib date number 转换为 'Month Day\\nYear' 格式"""
    return mdates.num2date(dt).strftime("%B %d\n%Y")

# 7) Plot: 3 rows, shared x-axis
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 12))
# 子图间隙设置为0
plt.subplots_adjust(hspace=0)


# Row 1: flux ± error
axes[0].errorbar(flux_y.index, flux_y.values, yerr=err_y.values, linewidth=1.3, label=f" [{RIGIDITY_MIN}-{RIGIDITY_MAX}] GV")
# axes[0].fill_between(flux_y.index, lower.values, upper.values, alpha=0.25, label="Flux ± error")
axes[0].set_ylabel(r"Proton Flux")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(loc="best")

# Row 2: Bz_gse
axes[1].scatter(B_y.index, B_y.values, linewidth=1.2, s=3, label="Bz_gse (Field magnitude)")
axes[1].set_ylabel("Bz_gse (nT)")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend(loc="best")

# Row 3: V
axes[2].scatter(V_y.index, V_y.values, linewidth=1.2, s=3, label="V (Flow speed)", color="tab:red")
axes[2].set_ylabel("V (km/s)")
# axes[2].set_xlabel("Date")
axes[2].set_xticklabels([format_date(d) for d in axes[2].get_xticks()], rotation=0)
axes[2].grid(True, linestyle="--", alpha=0.4)
axes[2].legend(loc="best")

# title = f"Proton Flux  [{RIGIDITY_MIN}-{RIGIDITY_MAX}] GV  vs  Bz_gse & V"
# if RESAMPLE:
#     title += f"  (Resample={RESAMPLE})"
# fig.suptitle(title, y=0.98, fontsize=13)

# plt.tight_layout(rect=[0, 0, 1, 0.97])

# 8) Save
os.makedirs(OUT_DIR, exist_ok=True)
suffix = f"_{RESAMPLE}" if RESAMPLE else ""
png_path = os.path.join(OUT_DIR, OUT_NAME + suffix + ".png")
pdf_path = os.path.join(OUT_DIR, OUT_NAME + suffix + ".pdf")
plt.savefig(png_path, dpi=200)
plt.savefig(pdf_path)
print(f"Saved:\n  {png_path}\n  {pdf_path}")
