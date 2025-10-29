# -*- coding: utf-8 -*-
"""
分析 OMNI 参数与 AMS 质子通量的相关性

功能:
1. 计算 Pearson 和 Spearman 相关系数
2. 热图可视化
3. 识别最强相关参数
4. 考虑时间滞后效应 (0-24*n小时)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 配置 ----------
EVENT_DIR = "FD_20150619"
AMS_FILE = f"{EVENT_DIR}/ams_processed.csv"
OMNI_FILE = f"{EVENT_DIR}/omni_processed.csv"

# 选择要分析的OMNI参数（排除非物理量）
OMNI_PARAMS = [
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

# 选择代表性刚度区间（避免太多列）
# RIGIDITY_BINS = [
#     "1.00-1.16GV",   # 低刚度
#     "2.67-2.97GV",   # 中低刚度
#     "5.37-5.90GV",   # 中刚度
#     "10.10-11.00GV", # 中高刚度
#     "22.80-33.50GV", # 高刚度
# ]
RIGIDITY_BINS = [
    "1.00-1.16GV",
    "1.16-1.33GV",
    "1.33-1.51GV",
    "1.51-1.71GV",
    "1.71-1.92GV",
    "1.92-2.15GV",
    "2.15-2.40GV",
    "2.40-2.67GV",
    "2.67-2.97GV",
    "2.97-3.29GV",
    "3.29-3.64GV",
    "3.64-4.02GV",
    "4.02-4.43GV",
    "4.43-4.88GV",
    "4.88-5.37GV",
    "5.37-5.90GV",
    "5.90-6.47GV",
    "6.47-7.09GV",
    "7.09-7.76GV",
    "7.76-8.48GV",
    "8.48-9.26GV",
    "9.26-10.10GV",
    "10.10-11.00GV",
    "11.00-13.00GV",
    "13.00-16.60GV",
    "16.60-22.80GV",
    "22.80-33.50GV",
    "33.50-48.50GV",
    "48.50-69.70GV",
    "69.70-100.00GV"
]

MAX_LAG_HOURS = 24*20  # 最大时间滞后

print("=" * 70)
print("AMS-OMNI 相关性分析")
print("=" * 70)

# ---------- 加载数据 ----------
ams = pd.read_csv(AMS_FILE, parse_dates=["datetime"])
omni = pd.read_csv(OMNI_FILE, parse_dates=["datetime"])

# 合并数据
df = pd.merge(ams, omni, on="datetime", how="inner")
print(f"\n数据点数: {len(df)}")
print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

# ---------- 函数：计算相关性 ----------
def compute_correlation(x, y, method='pearson'):
    """计算相关系数，处理缺失值"""
    # 去除包含NaN的行
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:  # 至少需要10个有效点
        return np.nan, np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    if method == 'pearson':
        corr, pval = pearsonr(x_clean, y_clean)
    else:
        corr, pval = spearmanr(x_clean, y_clean)
    
    return corr, pval

# ---------- 1. 零滞后相关性分析 ----------
print("\n" + "=" * 70)
print("【零滞后相关性分析】")
print("=" * 70)

# 构建相关性矩阵
corr_matrix = pd.DataFrame(index=OMNI_PARAMS, columns=RIGIDITY_BINS)
pval_matrix = pd.DataFrame(index=OMNI_PARAMS, columns=RIGIDITY_BINS)

for omni_param in OMNI_PARAMS:
    if omni_param not in df.columns:
        continue
    
    for rigidity in RIGIDITY_BINS:
        ams_col = f"dI_{rigidity}"
        if ams_col not in df.columns:
            continue
        
        corr, pval = compute_correlation(
            df[omni_param].values,
            df[ams_col].values,
            method='pearson'
        )
        
        corr_matrix.loc[omni_param, rigidity] = corr
        pval_matrix.loc[omni_param, rigidity] = pval

# 转换为数值类型
corr_matrix = corr_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

# 找出最强相关的参数
print("\n最强相关参数 (|r| > 0.5, p < 0.01):")
for rigidity in RIGIDITY_BINS:
    significant = corr_matrix[rigidity][
        (corr_matrix[rigidity].abs() > 0.5) & 
        (pval_matrix[rigidity] < 0.01)
    ].sort_values(key=abs, ascending=False)
    
    if len(significant) > 0:
        print(f"\n  {rigidity}:")
        for param, corr in significant.items():
            pval = pval_matrix.loc[param, rigidity]
            print(f"    {param:20s}: r={corr:6.3f}, p={pval:.2e}")

# ---------- 2. 热图可视化 ----------
fig, ax = plt.subplots(figsize=(20, 8))

# Pearson相关系数热图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Pearson r', 'pad': 0.01})
ax.set_title('OMNI vs AMS dI (Pearson)', fontsize=14, pad=20)
# ax.set_xlabel('Rigidity', fontsize=12)
ax.set_ylabel('OMNI Parameter', fontsize=12)

# # 显著性标记 (p < 0.01用**, p < 0.05用*)
# for i, omni_param in enumerate(OMNI_PARAMS):
#     for j, rigidity in enumerate(RIGIDITY_BINS):
#         pval = pval_matrix.loc[omni_param, rigidity]
#         if pval < 0.01:
#             ax.text(j+0.5, i+0.7, '**', ha='center', va='center',
#                         color='black', fontsize=10, fontweight='bold')
#         elif pval < 0.05:
#             ax.text(j+0.5, i+0.7, '*', ha='center', va='center',
#                     color='black', fontsize=10)

plt.tight_layout()
plt.savefig(f"{EVENT_DIR}/correlation_heatmap_pearson.png", dpi=300)#, bbox_inches='tight')
plt.close(fig)
print(f"\n[✓] 热图已保存: {EVENT_DIR}/correlation_heatmap_pearson.png")


# Spearman相关系数（非线性关系）
spearman_matrix = corr_matrix.copy()
for omni_param in OMNI_PARAMS:
    if omni_param not in df.columns:
        continue
    for rigidity in RIGIDITY_BINS:
        ams_col = f"dI_{rigidity}"
        if ams_col not in df.columns:
            continue
        corr, _ = compute_correlation(
            df[omni_param].values,
            df[ams_col].values,
            method='spearman'
        )
        spearman_matrix.loc[omni_param, rigidity] = corr

spearman_matrix = spearman_matrix.astype(float)


fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(spearman_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Spearman rho', 'pad': 0.01})
ax.set_title('OMNI vs AMS dI (Spearman)', fontsize=14, pad=20)
ax.set_xlabel('Rigidity', fontsize=12)
ax.set_ylabel('OMNI Parameter', fontsize=12)

plt.tight_layout()
plt.savefig(f"{EVENT_DIR}/correlation_heatmap_spearman.png", dpi=300)#, bbox_inches='tight')
plt.close(fig)
print(f"\n[✓] 热图已保存: {EVENT_DIR}/correlation_heatmap_spearman.png")

# ---------- 3. 时间滞后分析 ----------
print("\n" + "=" * 70)
print("【时间滞后分析】")
print("=" * 70)

# 选择3个最相关的OMNI参数进行滞后分析
top_params = corr_matrix.abs().mean(axis=1).sort_values(ascending=False).head(5).index.tolist()
print(f"分析参数: {top_params}")

# 选择一个代表性刚度
target_rigidity = "5.37-5.90GV"
ams_col = f"dI_{target_rigidity}"

fig, axes = plt.subplots(1, len(top_params), figsize=(15, 4))
if len(top_params) == 1:
    axes = [axes]

for idx, omni_param in enumerate(top_params):
    lags = []
    corrs = []
    
    for lag in range(-MAX_LAG_HOURS, MAX_LAG_HOURS+1):
        # 正滞后：OMNI领先AMS
        if lag >= 0:
            omni_shifted = df[omni_param].values[:-lag] if lag > 0 else df[omni_param].values
            ams_shifted = df[ams_col].values[lag:]
        else:
            omni_shifted = df[omni_param].values[-lag:]
            ams_shifted = df[ams_col].values[:lag]
        
        corr, _ = compute_correlation(omni_shifted, ams_shifted)
        lags.append(lag)
        corrs.append(corr)
    
    # 绘图
    axes[idx].plot(lags, corrs, 'o-', linewidth=2, markersize=4)
    axes[idx].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[idx].axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero lag')
    
    # 标记最大相关
    max_idx = np.nanargmax(np.abs(corrs))
    max_lag = lags[max_idx]
    max_corr = corrs[max_idx]
    axes[idx].plot(max_lag, max_corr, 'r*', markersize=15, 
                   label=f'Max: lag={max_lag}h, r={max_corr:.3f}')
    
    axes[idx].set_xlabel('Lag (hours)', fontsize=11)
    axes[idx].set_ylabel('Correlation', fontsize=11)
    axes[idx].set_title(f'{omni_param} vs {target_rigidity}', fontsize=12)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{EVENT_DIR}/lag_correlation.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[✓] 滞后分析图已保存: {EVENT_DIR}/lag_correlation.png")

# ---------- 4. 保存结果 ----------
# 保存相关系数表
corr_matrix.to_csv(f"{EVENT_DIR}/correlation_matrix_pearson.csv")
spearman_matrix.to_csv(f"{EVENT_DIR}/correlation_matrix_spearman.csv")
pval_matrix.to_csv(f"{EVENT_DIR}/pvalue_matrix.csv")

print("\n" + "=" * 70)
print("分析完成！")
print(f"  输出文件:")
print(f"    - correlation_heatmap.png (热图)")
print(f"    - lag_correlation.png (滞后分析)")
print(f"    - correlation_matrix_pearson.csv")
print(f"    - correlation_matrix_spearman.csv")
print(f"    - pvalue_matrix.csv")
print("=" * 70)

plt.show()