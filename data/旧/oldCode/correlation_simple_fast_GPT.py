# -*- coding: utf-8 -*-
"""
Compute correlation between AMS ΔI and OMNI parameters.
Visualize as heatmap and bar ranking.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- paths ----------
ams_file = "FD_20150619/ams_processed.csv"
omni_file = "FD_20150619/omni_processed.csv"

# ---------- read ----------
ams = pd.read_csv(ams_file, parse_dates=["datetime"]).set_index("datetime")
omni = pd.read_csv(omni_file, parse_dates=["datetime"]).set_index("datetime")

# 只选研究期内共有的时间
common = ams.index.intersection(omni.index)
ams = ams.loc[common]
omni = omni.loc[common]

# ---------- pick key OMNI columns ----------
omni_cols = [
    # 行星际磁场（IMF）
    "B_avg_abs", "B_vec_mag", "Bx_gse", "By_gse", "Bz_gse", "sigma_B",
    # 太阳风参数
    "Vsw", "Np", "Tp", "P_dyn",
    # 地磁指数
    "Dst", "Kp", "ap_index", "AE",
    # 太阳活动
    "F10_7", "Sunspot_R",
    # OMNI Proton fluxes
    # "pf_gt1MeV", "pf_gt2MeV", "pf_gt4MeV", 
    "pf_gt10MeV", "pf_gt30MeV", "pf_gt60MeV", "pf_flag",
    # 其他
    "Ey_mVpm", "MA_alfven", #高 Ey、低 MA 表示湍流增强与传播慢化；影响 τ↑【Wibberenz 1998】
    "alpha_to_proton", "Mach_ms"
]
omni_sel = omni[omni_cols]

# ---------- AMS columns ----------
# rig_cols = [c for c in ams.columns if "GV" in c]  # AMS ΔI 列
rig_cols = [
    "1.00-1.16GV",   # 低刚度
    "2.67-2.97GV",   # 中低刚度
    "5.37-5.90GV",   # 中刚度
    "10.10-11.00GV", # 中高刚度
    "22.80-33.50GV", # 高刚度
]

# ---------- 去趋势 (滑动均值减去) ----------
window = 12  # 小时
ams_detrend = ams# - ams.rolling(window, center=True, min_periods=3).mean()
omni_detrend = omni_sel# - omni_sel.rolling(window, center=True, min_periods=3).mean()

# ---------- 计算相关矩阵 ----------
corr_matrix = pd.DataFrame(index=rig_cols, columns=omni_cols, dtype=float)

for rcol0 in rig_cols:
    rcol = f"dI_{rcol0}"
    for ocol in omni_cols:
        # Pearson相关
        valid = ams_detrend[rcol].notna() & omni_detrend[ocol].notna()
        if valid.sum() > 10:
            corr = ams_detrend.loc[valid, rcol].corr(omni_detrend.loc[valid, ocol])
        else:
            corr = np.nan
        corr_matrix.loc[rcol0, ocol] = corr

# ---------- 可视化: 热图 ----------
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix.T.astype(float), cmap="coolwarm", center=0,
            annot=True, fmt=".2f", cbar_kws={"label": "Pearson r"})
plt.title("Correlation between AMS ΔI and OMNI Parameters")
plt.xlabel("AMS Rigidity (GV)")
plt.ylabel("OMNI Parameters")
plt.tight_layout()
plt.savefig("FD_20150619/correlation_heatmap_GPT.png", dpi=300)

# ---------- 平均相关性排名 ----------
mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
plt.figure(figsize=(7,4))
sns.barplot(x=mean_corr.values, y=mean_corr.index, orient="h", color="steelblue")
plt.title("Mean |Correlation| across all rigidities")
plt.xlabel("Mean |r|")
plt.tight_layout()
plt.savefig("FD_20150619/correlation_ranking_GPT.png", dpi=300)

print("[OK] Generated correlation_heatmap_GPT.png and correlation_ranking_GPT.png")
