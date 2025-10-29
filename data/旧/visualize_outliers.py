# -*- coding: utf-8 -*-
"""
可视化 OMNI 数据的异常值标记情况
用于人工判断是否需要剔除异常值

异常值用红色叉号 (x) 标记
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------- 配置 ----------
OMNI_PATH = "omni/omni2_2011_2025.csv"
t_start = pd.Timestamp("2015-06-19 00:00:00")
t_end   = pd.Timestamp("2015-07-07 23:00:00")

# ---------- 加载数据 ----------
omni = pd.read_csv(OMNI_PATH)
omni["datetime"] = pd.to_datetime(omni["datetime"], format="%Y/%m/%d/%H", errors="coerce")
omni_win = omni[(omni["datetime"] >= t_start) & (omni["datetime"] <= t_end)].copy()

# ---------- 标记异常值 ----------
num_cols = omni_win.select_dtypes(include=[np.number]).columns

outlier_info = {}
for col in num_cols:
    s = omni_win[col].copy()
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    outlier_mask = (s < mu - 3*sd) | (s > mu + 3*sd)
    outlier_info[col] = {
        'data': s,
        'outlier_mask': outlier_mask,
        'mu': mu,
        'sd': sd,
        'n_outliers': outlier_mask.sum()
    }

# ---------- 绘图 ----------
# 只绘制有异常值的列
cols_with_outliers = [col for col, info in outlier_info.items() if info['n_outliers'] > 0]

if len(cols_with_outliers) == 0:
    print("未检测到任何异常值！")
else:
    n_cols = len(cols_with_outliers)
    n_rows = (n_cols + 1) // 2  # 每行2个子图
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, col in enumerate(cols_with_outliers):
        ax = axes[idx]
        info = outlier_info[col]
        
        # 绘制所有数据点
        ax.plot(omni_win["datetime"], info['data'], 
                'o-', markersize=3, linewidth=0.8, alpha=0.7, label='正常数据')
        
        # 标记异常值（红色叉）
        outlier_times = omni_win.loc[info['outlier_mask'], "datetime"]
        outlier_values = info['data'][info['outlier_mask']]
        ax.plot(outlier_times, outlier_values, 
                'rx', markersize=10, markeredgewidth=2, label=f"异常值 (n={info['n_outliers']})")
        
        # 绘制 3σ 边界
        ax.axhline(info['mu'] + 3*info['sd'], color='orange', linestyle='--', 
                   linewidth=1, alpha=0.6, label='μ+3σ')
        ax.axhline(info['mu'] - 3*info['sd'], color='orange', linestyle='--', 
                   linewidth=1, alpha=0.6, label='μ-3σ')
        ax.axhline(info['mu'], color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_title(f"{col}\n(μ={info['mu']:.2f}, σ={info['sd']:.2f})", fontsize=10)
        ax.set_xlabel("时间")
        ax.set_ylabel(col)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for idx in range(len(cols_with_outliers), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("omni_outliers_visualization.png", dpi=150, bbox_inches='tight')
    print(f"[✓] 异常值可视化已保存: omni_outliers_visualization.png")
    print(f"    检测到异常值的变量: {len(cols_with_outliers)} 个")
    print("\n异常值统计:")
    for col in cols_with_outliers:
        print(f"  - {col}: {outlier_info[col]['n_outliers']} 个异常值")
    
    plt.show()

# ---------- 所有变量概览（无异常值的也显示） ----------
print("\n生成所有OMNI变量的完整概览图...")

n_all = len(num_cols)
n_rows_all = (n_all + 2) // 3  # 每行3个子图

fig2, axes2 = plt.subplots(n_rows_all, 3, figsize=(18, 3.5*n_rows_all))
axes2 = axes2.flatten()

for idx, col in enumerate(num_cols):
    ax = axes2[idx]
    info = outlier_info[col]
    
    # 绘制数据
    ax.plot(omni_win["datetime"], info['data'], 
            'o-', markersize=2, linewidth=0.6, alpha=0.7, color='steelblue')
    
    # 如果有异常值则标记
    if info['n_outliers'] > 0:
        outlier_times = omni_win.loc[info['outlier_mask'], "datetime"]
        outlier_values = info['data'][info['outlier_mask']]
        ax.plot(outlier_times, outlier_values, 
                'rx', markersize=8, markeredgewidth=1.5)
    
    ax.set_title(f"{col} (异常: {info['n_outliers']})", fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.2)

# 隐藏多余子图
for idx in range(len(num_cols), len(axes2)):
    axes2[idx].axis('off')

plt.tight_layout()
plt.savefig("omni_all_variables_overview.png", dpi=150, bbox_inches='tight')
print(f"[✓] 完整概览图已保存: omni_all_variables_overview.png")
plt.show()