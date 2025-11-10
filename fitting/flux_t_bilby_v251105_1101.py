# -*- coding: utf-8 -*-
"""

功能：
使用bilby框架对Forbush Decrease事件中通量随时间的双FRED叠加轮廓进行贝叶斯拟合。
修复内容：
1. 修复双FRED叠加模型逻辑（正确处理基线）
2. 改进Prior设置（合理的参数范围）
3. 自动检测两个下降事件的起始时间
4. 增强参数有效性检查
5. 更好的错误处理和日志

参考：
- bilby: https://github.com/bilby-dev/bilby
- PyGRB: https://github.com/JamesPaynter/PyGRB
"""

import pandas as pd
import numpy as np
import os
import sys
import bilby
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ====================== 用户参数 ======================
fd_id = 'FD_20150619'
AMS_PATH = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/ams_processed.csv"
OMNI_PATH = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/omni_processed.csv"
OUT_DIR = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/fig_flux_bilby"

# 选用的三个代表性刚度区间
SELECTED_RIGS = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

# bilby采样器设置
SAMPLER = "dynesty"  # 可选: 'nestle', 'dynesty', 'pymultinest', 'emcee'
N_LIVE = 500  # 采样点数
OUTDIR_BILBY = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/bilby_results"

MIN_FLOAT = 1e-10

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUTDIR_BILBY, exist_ok=True)


# ====================== 单FRED脉冲函数 ======================

def fred_pulse(t, baseline, A, Delta, tau, xi):
    """
    单个FRED脉冲函数
    
    Flux(t) = baseline - A * exp[-xi * (tau/(t-Delta) + (t-Delta)/tau - 2)]
    
    参数：
    - baseline: 基线通量
    - A: 脉冲幅度（正值表示下降）
    - Delta: 脉冲起始时间
    - tau: 特征恢复时间
    - xi: 不对称参数
    """
    t = np.atleast_1d(t)
    dt = t - Delta
    
    # 当 t <= Delta 时，返回基线
    result = np.full_like(dt, baseline, dtype=float)
    
    # 对于 t > Delta 的部分进行计算
    valid = dt > MIN_FLOAT
    
    if np.any(valid):
        dtv = dt[valid]
        tau_safe = max(tau, MIN_FLOAT)
        dtv_safe = np.maximum(dtv, MIN_FLOAT)
        
        exponent = -xi * (tau_safe / dtv_safe + dtv_safe / tau_safe - 2.0)
        # 防止溢出
        exponent = np.clip(exponent, -700, 700)
        result[valid] = baseline - A * np.exp(exponent)
    
    return result


def double_fred_model(t, baseline1, A1, Delta1, tau1, xi1, 
                      baseline2, A2, Delta2, tau2, xi2):
    """
    ✅ 修复版本：双FRED脉冲模型
    
    正确的叠加逻辑：
    - 两个脉冲共享一个全局基线（baseline1）
    - 每个脉冲只计算其下降部分（A * decay_function）
    - 最终通量 = baseline + 衰减1 + 衰减2
    
    参数：
    - baseline1, A1, Delta1, tau1, xi1: 第一个FRED脉冲的参数
    - baseline2, A2, Delta2, tau2, xi2: 第二个FRED脉冲的参数
    
    返回：两个脉冲的正确叠加通量
    """
    t = np.atleast_1d(t)
    dt1 = t - Delta1
    dt2 = t - Delta2
    
    # 从全局基线开始
    result = np.full_like(t, baseline1, dtype=float)
    
    # 第一个脉冲的贡献
    valid1 = dt1 > MIN_FLOAT
    if np.any(valid1):
        dtv1 = dt1[valid1]
        tau1_safe = max(tau1, MIN_FLOAT)
        dtv1_safe = np.maximum(dtv1, MIN_FLOAT)
        
        exponent1 = -xi1 * (tau1_safe / dtv1_safe + dtv1_safe / tau1_safe - 2.0)
        exponent1 = np.clip(exponent1, -700, 700)
        result[valid1] -= A1 * np.exp(exponent1)
    
    # 第二个脉冲的贡献
    valid2 = dt2 > MIN_FLOAT
    if np.any(valid2):
        dtv2 = dt2[valid2]
        tau2_safe = max(tau2, MIN_FLOAT)
        dtv2_safe = np.maximum(dtv2, MIN_FLOAT)
        
        exponent2 = -xi2 * (tau2_safe / dtv2_safe + dtv2_safe / tau2_safe - 2.0)
        exponent2 = np.clip(exponent2, -700, 700)
        result[valid2] -= A2 * np.exp(exponent2)
    
    return result


# ====================== 参数自动初始化函数 ======================

def auto_detect_fred_times(times, fluxes):
    """
    ✅ 自动检测两个FRED脉冲的起始时间
    
    使用导数和局部极值检测两个下降段
    """
    # 计算一阶导数（下降率）
    flux_derivative = np.diff(fluxes) / np.diff(times)
    
    # 找到最负的导数（最快的下降）
    most_negative_indices = np.argsort(flux_derivative)[:5]  # 前5个最负值
    
    if len(most_negative_indices) < 2:
        # 回退方案：使用简单的全局最小值
        argmin_idx = np.argmin(fluxes)
        Delta1 = times[max(0, argmin_idx - 10)]
        Delta2 = times[argmin_idx]
        print(f"[DEBUG] 使用回退方案：Delta1={Delta1:.2f}h, Delta2={Delta2:.2f}h")
        return Delta1, Delta2
    
    # 获取前两个最快下降点的时间
    deltas = sorted([times[idx] for idx in most_negative_indices[:2]])
    Delta1, Delta2 = deltas[0], deltas[1]
    
    print(f"[DEBUG] 自动检测到两个下降起点: Delta1={Delta1:.2f}h, Delta2={Delta2:.2f}h")
    return Delta1, Delta2


def validate_parameters(params):
    """
    ✅ 检查参数的物理合理性
    """
    # 幅度必须为正
    if params['A1'] <= 0 or params['A2'] <= 0:
        return False, "幅度必须为正"
    
    # 时间常数必须为正
    if params['tau1'] <= 0 or params['tau2'] <= 0:
        return False, "时间常数必须为正"
    
    # 不对称参数在合理范围内
    if params['xi1'] <= 0 or params['xi1'] > 10:
        return False, "xi1超出范围"
    if params['xi2'] <= 0 or params['xi2'] > 10:
        return False, "xi2超出范围"
    
    # 第一个脉冲应该早于第二个脉冲
    if params['Delta1'] >= params['Delta2']:
        return False, "Delta1应该早于Delta2"
    
    # 基线应该为正（通量为正）
    if params['baseline1'] <= 0 or params['baseline2'] <= 0:
        return False, "基线应该为正"
    
    return True, "参数有效"


# ====================== 数据读取与预处理 ======================

print("[INFO] 读取数据...")
try:
    ams = pd.read_csv(AMS_PATH, parse_dates=["datetime"]).set_index("datetime")
    omni = pd.read_csv(OMNI_PATH, parse_dates=["datetime"]).set_index("datetime")
except FileNotFoundError as e:
    print(f"[ERROR] 数据文件不存在: {e}")
    sys.exit(1)

# 时间对齐
common = ams.index.intersection(omni.index)
ams = ams.loc[common]
omni = omni.loc[common]

print(f"[INFO] AMS时间范围: {ams.index.min()} ~ {ams.index.max()} ({len(ams)}点)")

# 时间转换为相对小时
t0 = ams.index[0]
ams['hours'] = [(t - t0).total_seconds() / 3600 for t in ams.index]

# ====================== MAIN: 双FRED拟合流程 ======================

for rig in SELECTED_RIGS:
    flux_col = f"I_{rig}"
    err_col = f"I_{rig}_err"
    
    if flux_col not in ams.columns or err_col not in ams.columns:
        print(f"[WARN] 缺少 {rig} 的数据，跳过")
        continue
    
    print(f"\n{'='*60}")
    print(f"处理刚度: {rig}")
    print(f"{'='*60}")
    
    # 准备数据
    times = ams['hours'].values
    fluxes = ams[flux_col].values
    flux_errs = ams[err_col].values
    
    # 过滤无效数据
    valid = ~(np.isnan(fluxes) | np.isnan(flux_errs))
    times = times[valid]
    fluxes = fluxes[valid]
    flux_errs = flux_errs[valid]
    
    n = len(times)
    if n < 20:
        print(f"[WARN] 数据点过少 ({n} < 20)，跳过")
        continue
    
    print(f"[INFO] 有效数据点数: {n}")
    print(f"[INFO] 时间范围: {times[0]:.2f} - {times[-1]:.2f} 小时")
    print(f"[INFO] 通量范围: {fluxes.min():.3e} - {fluxes.max():.3e}")
    print(f"[INFO] 平均误差: {flux_errs.mean():.3e}")
    
    # --------- ✅ 改进的初始参数估计 ---------
    flux_max = np.max(fluxes)
    flux_min = np.min(fluxes)
    flux_mean = np.mean(fluxes)
    time_span = times[-1] - times[0]
    
    # 自动检测两个FRED脉冲的起始时间
    Delta1_init, Delta2_init = auto_detect_fred_times(times, fluxes)
    
    print(f"[INFO] 初始参数估计完成")
    
    # --------- ✅ 改进的Prior设置 ---------
    priors = bilby.core.prior.PriorDict()
    
    # 第一个FRED脉冲的prior
    priors['baseline1'] = bilby.core.prior.Uniform(
        name='baseline1', 
        minimum=flux_min * 0.5, 
        maximum=flux_max * 1.5,
        latex_label=r'$b_1$'
    )
    priors['A1'] = bilby.core.prior.Uniform(
        name='A1', 
        minimum=0.01, 
        maximum=(flux_max - flux_min) * 2,  # ✅ 减少上界
        latex_label=r'$A_1$'
    )
    priors['Delta1'] = bilby.core.prior.Uniform(
        name='Delta1', 
        minimum=times[0],              # ✅ 允许从时间序列开始
        maximum=times[-1] - 50,        # ✅ 给第二个脉冲留出空间
        latex_label=r'$\Delta_1$'
    )
    priors['tau1'] = bilby.core.prior.Uniform(
        name='tau1', 
        minimum=0.5, 
        maximum=time_span * 2,
        latex_label=r'$\tau_1$'
    )
    priors['xi1'] = bilby.core.prior.LogUniform(  # ✅ 改用LogUniform
        name='xi1', 
        minimum=0.1, 
        maximum=5.0,
        latex_label=r'$\xi_1$'
    )
    
    # 第二个FRED脉冲的prior
    priors['baseline2'] = bilby.core.prior.Uniform(
        name='baseline2', 
        minimum=flux_min * 0.5, 
        maximum=flux_max * 1.5,
        latex_label=r'$b_2$'
    )
    priors['A2'] = bilby.core.prior.Uniform(
        name='A2', 
        minimum=0.01, 
        maximum=(flux_max - flux_min) * 2,  # ✅ 减少上界
        latex_label=r'$A_2$'
    )
    priors['Delta2'] = bilby.core.prior.Uniform(
        name='Delta2', 
        minimum=times[0] + 20,         # ✅ 确保Delta2 > Delta1
        maximum=times[-1],
        latex_label=r'$\Delta_2$'
    )
    priors['tau2'] = bilby.core.prior.Uniform(
        name='tau2', 
        minimum=0.5, 
        maximum=time_span * 2,
        latex_label=r'$\tau_2$'
    )
    priors['xi2'] = bilby.core.prior.LogUniform(  # ✅ 改用LogUniform
        name='xi2', 
        minimum=0.1, 
        maximum=5.0,
        latex_label=r'$\xi_2$'
    )
    
    print(f"[INFO] Prior范围已设置（共10个参数）")
    
    # --------- 创建Likelihood ---------
    likelihood = bilby.core.likelihood.GaussianLikelihood(
        x=times,
        y=fluxes,
        func=double_fred_model,
        sigma=flux_errs
    )
    
    print(f"[INFO] Likelihood已创建 (GaussianLikelihood)")
    
    # --------- 运行bilby采样 ---------
    result_label = f"double_fred_{rig.replace('.', 'p').replace('-', '_')}"
    
    print(f"[INFO] 开始bilby采样 (采样器: {SAMPLER}, live points: {N_LIVE})...")
    print(f"[INFO] 这可能需要几分钟...")
    
    try:
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=SAMPLER,
            nlive=N_LIVE,
            outdir=OUTDIR_BILBY,
            label=result_label,
            save=True,
            resume=False,
            verbose=False,
            walks=50,           # ✅ dynesty参数：采样步数
            dlogz=0.1,         # ✅ 停止条件
            sample='rwalk',    # ✅ 采样方法
            bound='live',      # ✅ 边界
        )
    except Exception as e:
        print(f"[ERROR] 采样失败: {e}")
        continue
    
    print(f"[OK] 采样完成")
    print(f"[INFO] 后验样本数: {len(result.posterior['baseline1'])}")
    
    # --------- 提取结果 ---------
    print(f"\n[RESULT] 双FRED拟合参数:")
    print(f"{'参数':<15} {'中位值':<15} {'下界(16%)':<15} {'上界(84%)':<15}")
    print("-" * 60)
    
    # 仅提取模型参数
    model_params = ['baseline1', 'A1', 'Delta1', 'tau1', 'xi1',
                    'baseline2', 'A2', 'Delta2', 'tau2', 'xi2']
    
    for name in model_params:
        med = result.posterior[name].median()
        low = result.posterior[name].quantile(0.16)
        high = result.posterior[name].quantile(0.84)
        print(f"{name:<15} {med:<15.3e} {low:<15.3e} {high:<15.3e}")
    
    # --------- 绘制拟合结果 ---------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 数据与拟合曲线
    ax = axes[0, 0]
    ax.errorbar(times, fluxes, yerr=flux_errs, fmt='o', 
                markersize=4, color='black', alpha=0.6, label='Data')
    
    # ✅ 改进的后验曲线绘制
    n_posterior = len(result.posterior['baseline1'])
    n_samples = min(500, n_posterior)
    
    print(f"[DEBUG] 绘制 {n_samples} 条后验曲线（总后验样本数: {n_posterior}）")
    
    posteriors_dict = {name: result.posterior[name].values 
                       for name in model_params}
    sample_indices = np.random.choice(n_posterior, 
                                      size=n_samples, 
                                      replace=False)
    
    times_fine = np.linspace(times[0], times[-1], 200)
    n_valid_curves = 0
    
    for idx in sample_indices:
        params = {name: posteriors_dict[name][idx] 
                 for name in model_params}
        
        # ✅ 参数有效性检查
        is_valid, msg = validate_parameters(params)
        if not is_valid:
            continue
        
        try:
            y_sample = double_fred_model(times_fine, **params)
            if np.all(np.isfinite(y_sample)):
                ax.plot(times_fine, y_sample, color='red', alpha=0.02)
                n_valid_curves += 1
        except Exception as e:
            continue
    
    print(f"[INFO] 成功绘制 {n_valid_curves} 条有效的后验曲线")
    
    # 绘制中位值曲线
    params_median = {name: result.posterior[name].median() 
                    for name in model_params}
    y_median = double_fred_model(times_fine, **params_median)
    ax.plot(times_fine, y_median, color='red', linewidth=2.5, 
            label='Median fit', zorder=10)
    
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title(f'Double FRED Fit: {rig}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 残差
    ax = axes[0, 1]
    y_fit = double_fred_model(times, **params_median)
    residuals = fluxes - y_fit
    ax.errorbar(times, residuals, yerr=flux_errs, fmt='o', 
                markersize=4, color='blue', alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 两个单独的FRED脉冲
    ax = axes[1, 0]
    pulse1 = fred_pulse(times_fine, 
                       result.posterior['baseline1'].median(),
                       result.posterior['A1'].median(),
                       result.posterior['Delta1'].median(),
                       result.posterior['tau1'].median(),
                       result.posterior['xi1'].median())
    pulse2 = fred_pulse(times_fine,
                       result.posterior['baseline2'].median(),
                       result.posterior['A2'].median(),
                       result.posterior['Delta2'].median(),
                       result.posterior['tau2'].median(),
                       result.posterior['xi2'].median())
    
    ax.plot(times_fine, pulse1, 'o-', color='blue', linewidth=2, 
            markersize=1, label='FRED pulse 1', alpha=0.7)
    ax.plot(times_fine, pulse2, 's-', color='green', linewidth=2, 
            markersize=1, label='FRED pulse 2', alpha=0.7)
    ax.plot(times_fine, y_median, 'r-', linewidth=2.5, 
            label='Combined (actual)', alpha=0.8)
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title('Component FRED Pulses', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. 统计信息文本框
    ax = axes[1, 1]
    ax.axis('off')
    
    chi2_residual = np.sum((residuals / flux_errs) ** 2)
    ndf = len(times) - len(model_params)
    
    info_text = f"""
Double FRED Fitting Results (v3 Fixed)
{'='*45}

Rigidity: {rig}
Data points: {n}
Time range: {times[0]:.2f} - {times[-1]:.2f} h
Flux range: {fluxes.min():.3e} - {fluxes.max():.3e}

Sampler: {SAMPLER}
Live points: {N_LIVE}
Posterior samples: {n_posterior}
Valid posterior curves: {n_valid_curves}

χ²/dof = {chi2_residual:.2f} / {ndf} = {chi2_residual/ndf:.2f}
Evidence (log): {result.log_evidence:.2f} ± {result.log_evidence_err:.2f}

PULSE 1:
baseline₁ = {result.posterior['baseline1'].median():.3e}
A₁ = {result.posterior['A1'].median():.3e}
Δ₁ = {result.posterior['Delta1'].median():.2f} ± {result.posterior['Delta1'].std():.2f} h
τ₁ = {result.posterior['tau1'].median():.2f} ± {result.posterior['tau1'].std():.2f} h
ξ₁ = {result.posterior['xi1'].median():.2f} ± {result.posterior['xi1'].std():.2f}

PULSE 2:
baseline₂ = {result.posterior['baseline2'].median():.3e}
A₂ = {result.posterior['A2'].median():.3e}
Δ₂ = {result.posterior['Delta2'].median():.2f} ± {result.posterior['Delta2'].std():.2f} h
τ₂ = {result.posterior['tau2'].median():.2f} ± {result.posterior['tau2'].std():.2f} h
ξ₂ = {result.posterior['xi2'].median():.2f} ± {result.posterior['xi2'].std():.2f}
    """
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=8.5, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_file = os.path.join(OUT_DIR, f"double_fred_fit_{rig.replace('.', 'p')}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[OK] 图像保存: {plot_file}")
    plt.close()
    
    # --------- 绘制后验分布（Corner图） ---------
    print(f"[INFO] 生成后验分布图...")
    try:
        result.plot_corner(
            filename=os.path.join(OUTDIR_BILBY, 
                                 f'{result_label}_corner.png'),
            parameters=model_params
        )
        print(f"[OK] 后验分布图已保存")
    except Exception as e:
        print(f"[WARN] 后验分布图生成失败: {e}")

print(f"\n✅ 全部双FRED拟合完成，结果保存在:")
print(f"   - 拟合图: {OUT_DIR}/")
print(f"   - bilby结果: {OUTDIR_BILBY}/")