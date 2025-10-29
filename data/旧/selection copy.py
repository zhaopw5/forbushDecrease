# -*- coding: utf-8 -*-
"""
筛选在所有刚度下都显著相关的 OMNI 参数

筛选条件:
- 在所有刚度通道中，|Spearman ρ| > 0.5
- 在所有刚度通道中，p < 0.01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

print("=" * 70)
print("筛选强相关 OMNI 参数 (Spearman)")
print("=" * 70)
print(f"筛选条件:")
print(f"  - 在所有刚度下 |ρ| > {RHO_THRESHOLD}")
print(f"  - 在所有刚度下 p < {P_THRESHOLD}")
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

# ---------- 筛选强相关参数 ----------
print("\n筛选满足条件的 OMNI 参数...")

qualified_params = []

for omni_param in OMNI_PARAMS:
    if omni_param not in corr_matrix.index:
        continue
    
    # 获取该参数在所有刚度下的相关系数和p值
    rho_values = corr_matrix.loc[omni_param]
    pval_values = pval_matrix.loc[omni_param]
    
    # 去除NaN值
    valid_mask = ~rho_values.isna()
    rho_valid = rho_values[valid_mask]
    pval_valid = pval_values[valid_mask]
    
    if len(rho_valid) == 0:
        continue
    
    # 检查是否所有刚度都满足条件
    abs_rho_valid = rho_valid.abs()
    all_strong = (abs_rho_valid > RHO_THRESHOLD).all()
    all_significant = (pval_valid < P_THRESHOLD).all()
    
    if all_strong and all_significant:
        qualified_params.append({
            'parameter': omni_param,
            'mean_rho': rho_valid.mean(),
            'min_rho': rho_valid.min(),
            'max_rho': rho_valid.max(),
            'mean_abs_rho': abs_rho_valid.mean(),
            'min_abs_rho': abs_rho_valid.min(),
            'max_abs_rho': abs_rho_valid.max(),
            'max_pval': pval_valid.max(),
            'n_rigidity': len(rho_valid)
        })

# ---------- 显示结果 ----------
print("\n" + "=" * 70)
print(f"【筛选结果】共找到 {len(qualified_params)} 个满足条件的参数")
print("=" * 70)

if len(qualified_params) == 0:
    print("\n⚠️ 没有找到在所有刚度下都满足 |ρ|>0.5 且 p<0.01 的参数")
    print("\n建议:")
    print("  1. 放宽条件: 例如 |ρ|>0.3 或允许部分刚度不满足")
    print("  2. 检查数据质量")
    print("  3. 考虑分刚度范围分析 (低/中/高刚度)")
else:
    # 按平均相关系数排序
    qualified_df = pd.DataFrame(qualified_params)
    qualified_df = qualified_df.sort_values('mean_abs_rho', ascending=False)
    
    print("\n按平均|ρ|排序:")
    print("-" * 70)
    for idx, row in qualified_df.iterrows():
        print(f"\n{row['parameter']}:")
        print(f"  平均 ρ = {row['mean_rho']:6.3f}")
        print(f"  范围: [{row['min_rho']:6.3f}, {row['max_rho']:6.3f}]")
        print(f"  平均|ρ| = {row['mean_abs_rho']:6.3f}")
        print(f"  最大 p值 = {row['max_pval']:.2e}")
        print(f"  有效刚度数 = {row['n_rigidity']}")
    
    # 保存结果
    qualified_df.to_csv(f"{EVENT_DIR}/qualified_omni_params.csv", index=False)
    print(f"\n[✓] 筛选结果已保存: {EVENT_DIR}/qualified_omni_params.csv")
    
    # ---------- 可视化：合格参数的热图 ----------
    print("\n生成合格参数热图...")
    
    if len(qualified_params) > 0:
        qualified_param_names = qualified_df['parameter'].tolist()
        
        fig, ax = plt.subplots(figsize=(20, max(4, len(qualified_param_names)*0.3)))
        
        # 只显示合格的参数
        corr_subset = corr_matrix.loc[qualified_param_names]
        
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    cbar_kws={'label': 'Spearman ρ'})
        
        ax.set_title(f'合格 OMNI 参数 (所有刚度 |ρ|>{RHO_THRESHOLD}, p<{P_THRESHOLD})', 
                     fontsize=14, pad=20)
        ax.set_xlabel('Rigidity', fontsize=12)
        ax.set_ylabel('OMNI Parameter', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{EVENT_DIR}/qualified_params_heatmap.png", dpi=300)
        plt.close(fig)
        print(f"[✓] 热图已保存: {EVENT_DIR}/qualified_params_heatmap.png")

# ---------- 放宽条件的统计 ----------
print("\n" + "=" * 70)
print("【放宽条件的统计】")
print("=" * 70)

# 统计满足不同阈值的参数数量
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
        
        all_strong = (rho_valid.abs() > threshold).all()
        all_significant = (pval_valid < P_THRESHOLD).all()
        
        if all_strong and all_significant:
            count += 1
    
    relaxed_stats.append({'threshold': threshold, 'count': count})

print("\n不同阈值下满足条件的参数数量:")
for stat in relaxed_stats:
    print(f"  |ρ| > {stat['threshold']}: {stat['count']} 个参数")

# ---------- 部分满足条件的统计 ----------
print("\n" + "=" * 70)
print("【部分刚度满足条件的参数】")
print("=" * 70)

partial_qualified = []

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
    
    # 计算满足条件的刚度比例
    strong_mask = (rho_valid.abs() > RHO_THRESHOLD) & (pval_valid < P_THRESHOLD)
    satisfy_ratio = strong_mask.sum() / len(rho_valid)
    
    if satisfy_ratio >= 0.5:  # 至少50%的刚度满足条件
        partial_qualified.append({
            'parameter': omni_param,
            'satisfy_ratio': satisfy_ratio,
            'n_satisfy': strong_mask.sum(),
            'n_total': len(rho_valid),
            'mean_abs_rho': rho_valid[strong_mask].abs().mean() if strong_mask.sum() > 0 else 0
        })

if len(partial_qualified) > 0:
    partial_df = pd.DataFrame(partial_qualified)
    partial_df = partial_df.sort_values('satisfy_ratio', ascending=False)
    
    print(f"\n至少50%刚度满足条件的参数 (共 {len(partial_qualified)} 个):")
    print("-" * 70)
    for idx, row in partial_df.iterrows():
        print(f"{row['parameter']:20s}: {row['n_satisfy']:2d}/{row['n_total']:2d} "
              f"({row['satisfy_ratio']:5.1%})  平均|ρ|={row['mean_abs_rho']:.3f}")
    
    partial_df.to_csv(f"{EVENT_DIR}/partial_qualified_omni_params.csv", index=False)
    print(f"\n[✓] 部分满足条件的参数已保存: {EVENT_DIR}/partial_qualified_omni_params.csv")

print("\n" + "=" * 70)
print("筛选完成！")
print("=" * 70)