# -*- coding: utf-8 -*-
"""
改进版：先检验数据分布，再选择合适的相关性分析方法

新增功能:
1. 正态性检验 (Shapiro-Wilk / Anderson-Darling)
2. 线性关系可视化
3. 异常值检测
4. 根据检验结果推荐使用 Pearson 或 Spearman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, anderson

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 配置 ----------
EVENT_DIR = "FD_20150619"
AMS_FILE = f"{EVENT_DIR}/ams_processed.csv"
OMNI_FILE = f"{EVENT_DIR}/omni_processed.csv"

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

# 选择代表性刚度进行详细分析
SAMPLE_RIGIDITY = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

print("=" * 70)
print("数据分布检验与相关性分析")
print("=" * 70)

# ---------- 加载数据 ----------
ams = pd.read_csv(AMS_FILE, parse_dates=["datetime"])
omni = pd.read_csv(OMNI_FILE, parse_dates=["datetime"])
df = pd.merge(ams, omni, on="datetime", how="inner")

print(f"\n数据点数: {len(df)}")
print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

# ---------- 1. 正态性检验 ----------
def normality_test(data, name):
    """
    正态性检验
    返回: (is_normal, test_results)
    """
    # 去除NaN
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) < 20:
        return False, "样本量不足"
    
    # Shapiro-Wilk 检验 (样本量 < 5000)
    if len(data_clean) < 5000:
        stat, p_value = shapiro(data_clean)
        test_name = "Shapiro-Wilk"
    else:
        # Kolmogorov-Smirnov 检验 (大样本)
        stat, p_value = stats.kstest(data_clean, 'norm')
        test_name = "Kolmogorov-Smirnov"
    
    is_normal = p_value > 0.05  # α = 0.05
    
    result = {
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'n_samples': len(data_clean)
    }
    
    return is_normal, result

# ---------- 2. 异常值检测 ----------
def detect_outliers(data, method='iqr'):
    """
    异常值检测
    method: 'iqr' (四分位距) 或 'zscore' (Z分数)
    """
    data_clean = data[~np.isnan(data)]
    
    if method == 'iqr':
        Q1 = np.percentile(data_clean, 25)
        Q3 = np.percentile(data_clean, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (data_clean < lower) | (data_clean > upper)
    else:  # zscore
        z_scores = np.abs(stats.zscore(data_clean))
        outliers = z_scores > 3
    
    outlier_ratio = outliers.sum() / len(data_clean)
    
    return {
        'n_outliers': outliers.sum(),
        'outlier_ratio': outlier_ratio,
        'has_many_outliers': outlier_ratio > 0.1  # >10%认为异常值较多
    }

# ---------- 3. 检验报告 ----------
print("\n" + "=" * 70)
print("【数据分布检验报告】")
print("=" * 70)

distribution_report = []

# 检验 OMNI 参数
print("\n--- OMNI 参数 ---")
for param in OMNI_PARAMS[:5]:  # 先检验前5个参数
    if param not in df.columns:
        continue
    
    data = df[param].values
    is_normal, norm_result = normality_test(data, param)
    outlier_info = detect_outliers(data)
    
    distribution_report.append({
        'variable': param,
        'type': 'OMNI',
        'is_normal': is_normal,
        'p_value': norm_result['p_value'],
        'outlier_ratio': outlier_info['outlier_ratio'],
        'recommended_method': 'Pearson' if is_normal and not outlier_info['has_many_outliers'] else 'Spearman'
    })
    
    print(f"\n{param}:")
    print(f"  正态性: {'✓ 正态分布' if is_normal else '✗ 非正态分布'} (p={norm_result['p_value']:.4f})")
    print(f"  异常值: {outlier_info['n_outliers']} 个 ({outlier_info['outlier_ratio']:.1%})")
    print(f"  推荐方法: {distribution_report[-1]['recommended_method']}")

# 检验 AMS 刚度通道
print("\n--- AMS 刚度通道 ---")
for rigidity in SAMPLE_RIGIDITY:
    ams_col = f"dI_{rigidity}"
    if ams_col not in df.columns:
        continue
    
    data = df[ams_col].values
    is_normal, norm_result = normality_test(data, ams_col)
    outlier_info = detect_outliers(data)
    
    distribution_report.append({
        'variable': rigidity,
        'type': 'AMS',
        'is_normal': is_normal,
        'p_value': norm_result['p_value'],
        'outlier_ratio': outlier_info['outlier_ratio'],
        'recommended_method': 'Pearson' if is_normal and not outlier_info['has_many_outliers'] else 'Spearman'
    })
    
    print(f"\n{rigidity}:")
    print(f"  正态性: {'✓ 正态分布' if is_normal else '✗ 非正态分布'} (p={norm_result['p_value']:.4f})")
    print(f"  异常值: {outlier_info['n_outliers']} 个 ({outlier_info['outlier_ratio']:.1%})")
    print(f"  推荐方法: {distribution_report[-1]['recommended_method']}")

# ---------- 4. 可视化：分布检查 ----------
print("\n生成分布可视化图...")

# 选择2个OMNI参数 和 2个AMS刚度进行详细可视化
sample_omni = ['Vsw', 'Dst']
sample_ams = ['dI_1.00-1.16GV', 'dI_5.37-5.90GV']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# OMNI参数分布
for idx, param in enumerate(sample_omni):
    if param not in df.columns:
        continue
    
    data = df[param].dropna()
    
    # 直方图 + 正态拟合曲线
    ax1 = axes[idx, 0]
    ax1.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax1.set_title(f'{param} - Histogram')
    ax1.set_xlabel(param)
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q图
    ax2 = axes[idx, 1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'{param} - Q-Q Plot')
    ax2.grid(True, alpha=0.3)

# AMS刚度分布
for idx, ams_col in enumerate(sample_ams):
    if ams_col not in df.columns:
        continue
    
    data = df[ams_col].dropna()
    
    # 直方图
    ax1 = axes[idx, 2]
    ax1.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax1.set_title(f'{ams_col} - Histogram')
    ax1.set_xlabel('dI')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q图
    ax2 = axes[idx, 3]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'{ams_col} - Q-Q Plot')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{EVENT_DIR}/distribution_check.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[✓] 分布检验图已保存: {EVENT_DIR}/distribution_check.png")

# ---------- 5. 散点图矩阵：检查线性关系 ----------
print("\n生成散点图矩阵...")

# 选择3个OMNI参数 vs 1个AMS刚度
sample_omni_scatter = ['Vsw', 'Dst', 'B_vec_mag']
target_ams = 'dI_5.37-5.90GV'

if target_ams in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, omni_param in enumerate(sample_omni_scatter):
        if omni_param not in df.columns:
            continue
        
        ax = axes[idx]
        
        # 散点图
        x = df[omni_param].values
        y = df[target_ams].values
        
        # 去除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        ax.scatter(x_clean, y_clean, alpha=0.5, s=20)
        
        # 添加拟合线（线性回归）
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_fit, p(x_fit), 'r--', linewidth=2, label='Linear fit')
        
        # 计算相关系数
        r_pearson, _ = pearsonr(x_clean, y_clean)
        r_spearman, _ = spearmanr(x_clean, y_clean)
        
        ax.set_xlabel(omni_param, fontsize=11)
        ax.set_ylabel(target_ams, fontsize=11)
        ax.set_title(f'{omni_param} vs {target_ams.split("_")[1]}\n'
                     f'Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}',
                     fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{EVENT_DIR}/scatter_linearity_check.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] 散点图已保存: {EVENT_DIR}/scatter_linearity_check.png")

# ---------- 6. 保存检验报告 ----------
report_df = pd.DataFrame(distribution_report)
report_df.to_csv(f"{EVENT_DIR}/distribution_report.csv", index=False)
print(f"[✓] 检验报告已保存: {EVENT_DIR}/distribution_report.csv")

# ---------- 7. 总结建议 ----------
print("\n" + "=" * 70)
print("【分析建议】")
print("=" * 70)

n_normal = sum([r['is_normal'] for r in distribution_report])
n_total = len(distribution_report)

print(f"\n检验了 {n_total} 个变量:")
print(f"  - {n_normal} 个服从正态分布 ({n_normal/n_total:.1%})")
print(f"  - {n_total - n_normal} 个不服从正态分布 ({(n_total-n_normal)/n_total:.1%})")

spearman_count = sum([r['recommended_method'] == 'Spearman' for r in distribution_report])
print(f"\n推荐使用:")
print(f"  - Spearman: {spearman_count} 个变量")
print(f"  - Pearson: {n_total - spearman_count} 个变量")

if spearman_count / n_total > 0.5:
    print("\n⚠️ 超过50%的变量不满足正态性假设")
    print("   建议: 优先使用 Spearman 相关系数进行分析")
else:
    print("\n✓ 大部分变量满足正态性假设")
    print("  建议: 可以使用 Pearson 相关系数，但也计算 Spearman 作为对比")

print("\n" + "=" * 70)
print("分布检验完成！")
print("=" * 70)