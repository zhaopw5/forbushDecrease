# -*- coding: utf-8 -*-
"""
筛选在至少部分刚度下显著相关的 OMNI 参数

筛选条件:
- 至少 MIN_RATIO 比例的刚度通道中满足：|Spearman ρ| > RHO_THRESHOLD 且 p < P_THRESHOLD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ---------- 配置 ----------
EVENT_DIR = "FD_20150619"
AMS_FILE = f"{EVENT_DIR}/ams_processed.csv"
OMNI_FILE = f"{EVENT_DIR}/omni_processed.csv"

OMNI_PARAMS = [
    "B_avg_abs", "B_vec_mag", "Bx_gse", "By_gse", "Bz_gse", "sigma_B",
    "Vsw", "Np", "Tp", "P_dyn",
    "Dst", "Kp", "ap_index", "AE",
    "F10_7", "Sunspot_R",
    "pf_gt10MeV", "pf_gt30MeV", "pf_gt60MeV", "pf_flag",
    "Ey_mVpm", "MA_alfven",
    "alpha_to_proton", "Mach_ms"
]

ALL_RIGIDITY_BINS = [
    "1.00-1.16GV", "1.16-1.33GV", "1.33-1.51GV", "1.51-1.71GV",
    "1.71-1.92GV", "1.92-2.15GV", "2.15-2.40GV", "2.40-2.67GV",
    "2.67-2.97GV", "2.97-3.29GV", "3.29-3.64GV", "3.64-4.02GV",
    "4.02-4.43GV", "4.43-4.88GV", "4.88-5.37GV", "5.37-5.90GV",
    "5.90-6.47GV", "6.47-7.09GV", "7.09-7.76GV", "7.76-8.48GV",
    "8.48-9.26GV", "9.26-10.10GV", "10.10-11.00GV", "11.00-13.00GV",
    "13.00-16.60GV", "16.60-22.80GV", "22.80-33.50GV", "33.50-48.50GV",
    "48.50-69.70GV", "69.70-100.00GV"
]

# 筛选阈值
RHO_THRESHOLD = 0.5
P_THRESHOLD = 0.01
MIN_RATIO = 0.5  # 至少满足条件的刚度比例

print("=" * 70)
print("筛选强相关 OMNI 参数 (Spearman)")
print("=" * 70)
print(f"筛选条件:")
print(f"  - 至少 {MIN_RATIO:.0%} 的刚度满足 |ρ| > {RHO_THRESHOLD} 且 p < {P_THRESHOLD}")
print("=" * 70)

# ---------- 加载数据 ----------
ams = pd.read_csv(AMS_FILE, parse_dates=["datetime"])
omni = pd.read_csv(OMNI_FILE, parse_dates=["datetime"])
df = pd.merge(ams, omni, on="datetime", how="inner")

print(f"\n数据点数: {len(df)}")

# ---------- 计算 Spearman 相关系数 ----------
def compute_spearman(x, y):
    """计算 Spearman 相关系数"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return np.nan, np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    rho, pval = spearmanr(x_clean, y_clean)
    return rho, pval

# 构建相关性矩阵
print("\n计算 Spearman 相关系数...")
corr_matrix = pd.DataFrame(index=OMNI_PARAMS, columns=ALL_RIGIDITY_BINS)
pval_matrix = pd.DataFrame(index=OMNI_PARAMS, columns=ALL_RIGIDITY_BINS)

for omni_param in OMNI_PARAMS:
    if omni_param not in df.columns:
        continue
    
    for rigidity in ALL_RIGIDITY_BINS:
        ams_col = f"dI_{rigidity}"
        if ams_col not in df.columns:
            continue
        
        rho, pval = compute_spearman(
            df[omni_param].values,
            df[ams_col].values
        )
        
        corr_matrix.loc[omni_param, rigidity] = rho
        pval_matrix.loc[omni_param, rigidity] = pval

corr_matrix = corr_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

# ---------- 基于“至少 MIN_RATIO 刚度满足”的主筛选 ----------
print("\n筛选满足条件的 OMNI 参数...")

qualified_params = []

for omni_param in OMNI_PARAMS:
    if omni_param not in corr_matrix.index:
        continue
    
    rho_values = corr_matrix.loc[omni_param]
    pval_values = pval_matrix.loc[omni_param]
    
    valid_mask = ~rho_values.isna()
    rho_valid = rho_values[valid_mask]
    pval_valid = pval_values[valid_mask]
    if len(rho_valid) == 0:
        continue
    
    strong_mask = (rho_valid.abs() > RHO_THRESHOLD) & (pval_valid < P_THRESHOLD)
    satisfy_ratio = strong_mask.sum() / len(rho_valid)
    if satisfy_ratio >= MIN_RATIO:  # 至少 MIN_RATIO 的刚度满足条件
        qualified_params.append({
            'parameter': omni_param,
            'satisfy_ratio': satisfy_ratio,
            'n_satisfy': int(strong_mask.sum()),
            'n_total': int(len(rho_valid)),
            'mean_abs_rho': rho_valid[strong_mask].abs().mean() if strong_mask.sum() > 0 else 0,
            'max_pval_satisfied': pval_valid[strong_mask].max() if strong_mask.sum() > 0 else np.nan
        })

# ---------- 显示结果 ----------
print("\n" + "=" * 70)
print(f"【筛选结果】共找到 {len(qualified_params)} 个满足条件的参数")
print("=" * 70)

if len(qualified_params) == 0:
    print("\n⚠️ 没有找到满足“至少 "
          f"{MIN_RATIO:.0%} 刚度 |ρ|>{RHO_THRESHOLD} 且 p<{P_THRESHOLD}”的参数")
    print("\n建议:")
    print("  1. 放宽 |ρ| 阈值或降低刚度比例 (如 40%)")
    print("  2. 检查数据质量")
    print("  3. 考虑分刚度范围分析 (低/中/高刚度)")
else:
    qualified_df = pd.DataFrame(qualified_params)
    qualified_df = qualified_df.sort_values(['satisfy_ratio', 'mean_abs_rho'], ascending=False)
    
    print("\n按满足比例与平均|ρ|排序:")
    print("-" * 70)
    for _, row in qualified_df.iterrows():
        print(f"{row['parameter']:20s}: {row['n_satisfy']:2d}/{row['n_total']:2d} "
              f"({row['satisfy_ratio']:5.1%})  平均|ρ|={row['mean_abs_rho']:.3f}")
    
    # 保存结果
    qualified_df.to_csv(f"{EVENT_DIR}/qualified_omni_params.csv", index=False)
    print(f"\n[✓] 筛选结果已保存: {EVENT_DIR}/qualified_omni_params.csv")
    
    # ---------- 可视化：合格参数的热图 ----------
    print("\n生成合格参数热图...")
    qualified_param_names = qualified_df['parameter'].tolist()
    fig, ax = plt.subplots(figsize=(20, max(4, len(qualified_param_names)*0.3)))
    corr_subset = corr_matrix.loc[qualified_param_names]
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Spearman ρ'})
    ax.set_title(
        f'合格 OMNI 参数 (≥{MIN_RATIO:.0%} 刚度 |ρ|>{RHO_THRESHOLD}, p<{P_THRESHOLD})',
        fontsize=14, pad=20
    )
    ax.set_xlabel('Rigidity', fontsize=12)
    ax.set_ylabel('OMNI Parameter', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{EVENT_DIR}/qualified_params_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"[✓] 热图已保存: {EVENT_DIR}/qualified_params_heatmap.png")

# ---------- 放宽条件的统计（同样基于“至少 MIN_RATIO 刚度满足”） ----------
print("\n" + "=" * 70)
print("【放宽条件的统计】")
print("=" * 70)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
relaxed_stats = []

for threshold in thresholds:
    count = 0
    for omni_param in OMNI_PARAMS:
        if omni_param not in corr_matrix.index:
            continue
        
        rho_values = corr_matrix.loc[omni_param]
        pval_values = pval_matrix.loc[omni_param]
        
        valid_mask = ~rho_values.isna()
        rho_valid = rho_values[valid_mask]
        pval_valid = pval_values[valid_mask]
        if len(rho_valid) == 0:
            continue
        
        strong_mask = (rho_valid.abs() > threshold) & (pval_valid < P_THRESHOLD)
        satisfy_ratio = strong_mask.sum() / len(rho_valid)
        if satisfy_ratio >= MIN_RATIO:
            count += 1
    
    relaxed_stats.append({'threshold': threshold, 'count': count})

print(f"\n在至少 {MIN_RATIO:.0%} 刚度满足条件下，不同 |ρ| 阈值对应的参数数量:")
for stat in relaxed_stats:
    print(f"  |ρ| > {stat['threshold']}: {stat['count']} 个参数")

# ---------- 结束 ----------
print("\n" + "=" * 70)
print("筛选完成！")
print("=" * 70)