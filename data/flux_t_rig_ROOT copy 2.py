# -*- coding: utf-8 -*-
"""
flux_t_rig_ROOT.py

功能：
1. 用ROOT绘制 flux~time 图，并用负FRED函数拟合
2. 用ROOT绘制 flux~rigidity 图，分别用幂函数和指数函数拟合能谱
"""

import pandas as pd
import numpy as np
import os
import ROOT
from ROOT import TCanvas, TGraphErrors, TF1, TMultiGraph, TLegend, gStyle, gROOT
from datetime import datetime

# ====================== 用户参数 ======================
AMS_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20150619/ams_processed.csv"
OMNI_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20150619/omni_processed.csv"
OUT_DIR = "fig_flux_ROOT"

# AMS 选用的三个代表性刚度区间
SELECTED_RIGS = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

# OMNI 参数
OMNI_COLS = ["Sunspot_R", "Vsw", "F10_7", "Dst", "alpha_to_proton", "pf_gt10MeV"]

# ====================== ROOT 设置 ======================
gROOT.SetBatch(True)  # 批处理模式
gStyle.SetOptFit(1111)  # 显示拟合参数
gStyle.SetOptStat(0)    # 不显示统计框

os.makedirs(OUT_DIR, exist_ok=True)

# ====================== 读入数据 ======================
ams = pd.read_csv(AMS_PATH, parse_dates=["datetime"]).set_index("datetime")
omni = pd.read_csv(OMNI_PATH, parse_dates=["datetime"]).set_index("datetime")

# 时间对齐
common = ams.index.intersection(omni.index)
ams = ams.loc[common]
omni = omni.loc[common]

print(f"[INFO] AMS时间范围: {ams.index.min()} ~ {ams.index.max()} ({len(ams)}点)")

# 时间转换为相对小时（方便拟合）
t0 = ams.index[0]
ams['hours'] = [(t - t0).total_seconds() / 3600 for t in ams.index]

# ====================== 1. flux~time 图 + FRED拟合 ======================

# 负FRED函数定义（用于Forbush下降）
# 基线 - FRED脉冲 = 描述从平稳到下降再恢复
MIN_FLOAT = 1e-10

def negative_fred_func(x, par):
    """
    负FRED函数（用于Forbush下降）
    Flux(t) = baseline - A * exp[-xi * (tau/(t-Delta) + (t-Delta)/tau - 2)]
    
    par[0] = baseline (基线通量)
    par[1] = A (下降幅度)
    par[2] = Delta (下降开始时间)
    par[3] = tau (特征时间)
    par[4] = xi (不对称参数)
    """
    t = x[0]
    baseline = par[0]
    A = par[1]
    Delta = par[2]
    tau = par[3]
    xi = par[4]
    
    dt = t - Delta
    
    # 当 t <= Delta 时，返回基线
    if dt <= 0:
        return baseline
    
    # 避免除零
    if tau < MIN_FLOAT:
        tau = MIN_FLOAT
    if dt < MIN_FLOAT:
        dt = MIN_FLOAT
    
    # FRED公式
    exponent = -xi * (tau / dt + dt / tau - 2.0)
    fred_pulse = A * ROOT.TMath.Exp(exponent)
    
    # 负FRED：基线减去脉冲（形成下降）
    return baseline - fred_pulse

# 为每个刚度创建 flux~time 图并拟合
for rig in SELECTED_RIGS:
    flux_col = f"I_{rig}"
    err_col = f"I_{rig}_err"
    
    if flux_col not in ams.columns or err_col not in ams.columns:
        print(f"[WARN] 缺少 {rig} 的数据，跳过")
        continue
    
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
    if n == 0:
        continue
    
    # 创建TGraphErrors
    gr = TGraphErrors(n)
    for i in range(n):
        gr.SetPoint(i, times[i], fluxes[i])
        gr.SetPointError(i, 0, flux_errs[i])
    
    # 创建画布
    c1 = TCanvas(f"c_time_{rig}", f"Flux vs Time: {rig}", 1200, 600)
    
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(0.5)
    gr.SetMarkerColor(ROOT.kBlue)
    gr.SetLineColor(ROOT.kBlue)
    gr.SetTitle(f"Proton Flux vs Time: {rig};Time [hours];Flux")
    gr.Draw("AP")
    
    # 负FRED拟合（描述Forbush下降）
    # 初始参数估计
    flux_max = np.max(fluxes)  # 事件前基线
    flux_min = np.min(fluxes)  # 下降最低点
    baseline_init = flux_max
    A_init = flux_max - flux_min  # 下降幅度
    Delta_init = times[np.argmin(fluxes)] - (times[-1] - times[0]) / 10  # 稍早于最低点
    tau_init = (times[-1] - times[0]) / 10  # 特征时间
    xi_init = 1.0
    
    fred_fit = TF1(f"fred_{rig}", negative_fred_func, times[0], times[-1], 5)
    fred_fit.SetParameters(baseline_init, A_init, Delta_init, tau_init, xi_init)
    fred_fit.SetParNames("baseline", "A", "Delta", "tau", "xi")
    fred_fit.SetLineColor(ROOT.kRed)
    fred_fit.SetLineWidth(2)
    
    # 设置合理的参数范围
    fred_fit.SetParLimits(0, flux_min * 0.8, flux_max * 1.2)  # baseline
    fred_fit.SetParLimits(1, 0, (flux_max - flux_min) * 2)     # A > 0
    fred_fit.SetParLimits(2, times[0], times[-1])              # Delta
    fred_fit.SetParLimits(3, 0.1, (times[-1] - times[0]))      # tau > 0
    fred_fit.SetParLimits(4, 0.01, 10)                          # xi > 0
    
    gr.Fit(fred_fit, "R")
    
    c1.Update()
    c1.SaveAs(os.path.join(OUT_DIR, f"flux_time_{rig.replace('.', 'p')}.png"))
    print(f"[OK] 保存 {rig} 的 flux~time 拟合图")
    
    # 输出拟合参数
    print(f"  拟合参数: baseline={fred_fit.GetParameter(0):.3e}, "
          f"A={fred_fit.GetParameter(1):.3e}, "
          f"Delta={fred_fit.GetParameter(2):.2f}h, "
          f"tau={fred_fit.GetParameter(3):.2f}h, "
          f"xi={fred_fit.GetParameter(4):.2f}")

# ====================== 2. flux~rigidity 图 + 指数拟合 ======================

# 获取通量列
flux_cols = [
    c for c in ams.columns
    if c.startswith("I_") and ("rel" not in c) and ("dI" not in c) 
    and ("err" not in c) and ("hours" not in c)
]
err_cols = [c + "_err" for c in flux_cols]

# 计算刚度中心和误差
rigidity_centers = []
rigidity_errs = []
for c in flux_cols:
    part = c.replace("I_", "").replace("GV", "")
    rmin, rmax = map(float, part.split("-"))
    rigidity_centers.append((rmin + rmax) / 2)
    rigidity_errs.append((rmax - rmin) / 2)

rigidity_centers = np.array(rigidity_centers)
rigidity_errs = np.array(rigidity_errs)

print(f"[INFO] 刚度范围: {rigidity_centers.min():.2f} - {rigidity_centers.max():.2f} GV")
print(f"[INFO] 共有 {len(rigidity_centers)} 个刚度区间")

# 选择几个代表性时刻绘制能谱
n_times = len(ams)
time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]

# 创建画布 - 只用指数拟合
c2 = TCanvas("c_spectrum", "Flux Spectra (Exponential Fit)", 1000, 700)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetGrid()

mg_exp = TMultiGraph()
mg_exp.SetTitle("Proton Flux Spectra (Exponential Fit);Rigidity [GV];Flux")

colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]
legend_exp = TLegend(0.15, 0.15, 0.45, 0.40)
legend_exp.SetTextSize(0.03)

for idx, time_idx in enumerate(time_indices):
    row = ams.iloc[time_idx]
    timestamp = ams.index[time_idx]
    
    flux_vals = row[flux_cols].values
    flux_errs_vals = row[err_cols].values if all(c in ams.columns for c in err_cols) else np.zeros_like(flux_vals)
    
    valid = ~(np.isnan(flux_vals) | np.isnan(flux_errs_vals) | (flux_vals <= 0))
    if np.sum(valid) < 3:
        continue
    
    rig_valid = rigidity_centers[valid]
    flux_valid = flux_vals[valid]
    flux_err_valid = flux_errs_vals[valid]
    rig_err_valid = rigidity_errs[valid]
    
    print(f"\n[INFO] 时刻 {timestamp}: {len(rig_valid)} 个有效数据点")
    print(f"       刚度范围: {rig_valid.min():.2f} - {rig_valid.max():.2f} GV")
    
    gr_spec = TGraphErrors(len(rig_valid))
    for i in range(len(rig_valid)):
        gr_spec.SetPoint(i, rig_valid[i], flux_valid[i])
        gr_spec.SetPointError(i, rig_err_valid[i], flux_err_valid[i])
    
    gr_spec.SetMarkerStyle(20)
    gr_spec.SetMarkerSize(0.8)
    gr_spec.SetMarkerColor(colors[idx % len(colors)])
    gr_spec.SetLineColor(colors[idx % len(colors)])
    
    # 指数函数拟合: flux = A * exp(-B * R)
    # 设置拟合范围覆盖所有数据
    fit_min = rig_valid.min()
    fit_max = rig_valid.max()
    
    exp_fit = TF1(f"exp_{idx}", "[0]*TMath::Exp(-[1]*x)", fit_min, fit_max)
    
    # 改进初始值估计 - 使用对数线性回归
    log_flux = np.log(flux_valid)
    # log(flux) = log(A) - B*R
    # 使用最小二乘估计
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(rig_valid, log_flux, 1)  # [intercept, slope]
    A_init = np.exp(coef[0])
    B_init = -coef[1]
    
    print(f"       初始估计: A={A_init:.3e}, B={B_init:.4f}")
    
    exp_fit.SetParameters(A_init, B_init)
    exp_fit.SetParNames("A", "B")
    
    # 设置参数范围
    exp_fit.SetParLimits(0, A_init * 0.1, A_init * 10)    # A
    exp_fit.SetParLimits(1, B_init * 0.1, B_init * 10)    # B > 0
    
    exp_fit.SetLineColor(colors[idx % len(colors)])
    exp_fit.SetLineWidth(2)
    exp_fit.SetLineStyle(2)  # 虚线
    
    # 执行拟合 - 使用更好的拟合选项
    fit_result = gr_spec.Fit(exp_fit, "RSMQ+")  # S=保存结果, M=改进, Q=安静, +=添加到列表
    
    chi2 = exp_fit.GetChisquare()
    ndf = exp_fit.GetNDF()
    print(f"       拟合: A={exp_fit.GetParameter(0):.3e}, B={exp_fit.GetParameter(1):.3f}")
    print(f"       Chi2/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")
    
    mg_exp.Add(gr_spec, "P")
    legend_exp.AddEntry(gr_spec, timestamp.strftime("%m-%d %H:%M"), "lp")

mg_exp.Draw("A")
mg_exp.GetXaxis().SetRangeUser(rigidity_centers.min() * 0.8, rigidity_centers.max() * 1.2)
legend_exp.Draw()

c2.Update()
c2.SaveAs(os.path.join(OUT_DIR, "flux_spectrum_exponential.png"))
print(f"[OK] 保存能谱拟合图")

print(f"\n✅ 全部ROOT绘图和拟合完成，结果保存在 {OUT_DIR}/")