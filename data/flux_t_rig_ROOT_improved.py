# -*- coding: utf-8 -*-
"""
flux_t_rig_ROOT_improved.py

改进功能：
1. 同时使用单FRED和双FRED拟合 flux~time（红色和蓝色线）
2. 拟合普通的 flux vs R，但画图时显示 flux*R^2.7
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

# ====================== FRED函数定义 ======================
MIN_FLOAT = 1e-10

def single_fred_func(x, par):
    """
    单FRED函数（FRED1）
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
    
    if dt <= 0:
        return baseline
    
    if tau < MIN_FLOAT:
        tau = MIN_FLOAT
    if dt < MIN_FLOAT:
        dt = MIN_FLOAT
    
    exponent = -xi * (tau / dt + dt / tau - 2.0)
    fred_pulse = A * ROOT.TMath.Exp(exponent)
    
    return baseline - fred_pulse


def double_fred_func(x, par):
    """
    双FRED函数（FRED2）- 两个福布什下降叠加
    Flux(t) = baseline - FRED1 - FRED2
    
    par[0] = baseline
    par[1] = A1 (第一次下降幅度)
    par[2] = Delta1 (第一次下降开始时间)
    par[3] = tau1 (第一次特征时间)
    par[4] = xi1 (第一次不对称参数)
    par[5] = A2 (第二次下降幅度)
    par[6] = Delta2 (第二次下降开始时间)
    par[7] = tau2 (第二次特征时间)
    par[8] = xi2 (第二次不对称参数)
    """
    t = x[0]
    baseline = par[0]
    
    # 第一个FRED
    A1 = par[1]
    Delta1 = par[2]
    tau1 = par[3]
    xi1 = par[4]
    
    dt1 = t - Delta1
    fred1 = 0.0
    if dt1 > 0:
        tau1 = max(tau1, MIN_FLOAT)
        dt1 = max(dt1, MIN_FLOAT)
        exponent1 = -xi1 * (tau1 / dt1 + dt1 / tau1 - 2.0)
        fred1 = A1 * ROOT.TMath.Exp(exponent1)
    
    # 第二个FRED
    A2 = par[5]
    Delta2 = par[6]
    tau2 = par[7]
    xi2 = par[8]
    
    dt2 = t - Delta2
    fred2 = 0.0
    if dt2 > 0:
        tau2 = max(tau2, MIN_FLOAT)
        dt2 = max(dt2, MIN_FLOAT)
        exponent2 = -xi2 * (tau2 / dt2 + dt2 / tau2 - 2.0)
        fred2 = A2 * ROOT.TMath.Exp(exponent2)
    
    return baseline - fred1 - fred2


# ====================== 1. flux~time 图 + 单/双FRED拟合 ======================

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
    gr.SetMarkerColor(ROOT.kBlack)
    gr.SetLineColor(ROOT.kBlack)
    gr.SetTitle(f"Proton Flux vs Time: {rig};Time [hours];Flux")
    gr.Draw("AP")
    
    # ========== 单FRED拟合（FRED1，红色线）==========
    flux_max = np.max(fluxes)
    flux_min = np.min(fluxes)
    baseline_init = flux_max
    A_init = flux_max - flux_min
    Delta_init = times[np.argmin(fluxes)] - (times[-1] - times[0]) / 10
    tau_init = (times[-1] - times[0]) / 10
    xi_init = 1.0
    
    fred1_fit = TF1(f"fred1_{rig}", single_fred_func, times[0], times[-1], 5)
    fred1_fit.SetParameters(baseline_init, A_init, Delta_init, tau_init, xi_init)
    fred1_fit.SetParNames("baseline", "A", "Delta", "tau", "xi")
    fred1_fit.SetLineColor(ROOT.kRed)
    fred1_fit.SetLineWidth(2)
    fred1_fit.SetLineStyle(1)
    
    fred1_fit.SetParLimits(0, flux_min * 0.8, flux_max * 1.2)
    fred1_fit.SetParLimits(1, 0, (flux_max - flux_min) * 2)
    fred1_fit.SetParLimits(2, times[0], times[-1])
    fred1_fit.SetParLimits(3, 0.1, (times[-1] - times[0]))
    fred1_fit.SetParLimits(4, 0.01, 10)
    
    gr.Fit(fred1_fit, "RQ+")  # Q: 安静模式，+: 不清除之前的拟合
    
    # ========== 双FRED拟合（FRED2，蓝色线）==========
    # 策略：强制两个FRED在时间和幅度上都分开
    time_span = times[-1] - times[0]
    
    # 找到最低点作为参考
    min_idx = np.argmin(fluxes)
    min_time = times[min_idx]
    
    # 分析通量曲线，寻找可能的两个下降区域
    # 方法：对通量做平滑后找局部极小值
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(fluxes, sigma=max(1, len(fluxes)//20))
    
    # 找局部极小值
    local_mins = []
    for i in range(1, len(smoothed)-1):
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            local_mins.append(i)
    
    # 根据找到的极小值设置初值
    if len(local_mins) >= 2:
        # 选择最明显的两个
        sorted_mins = sorted(local_mins, key=lambda i: smoothed[i])[:2]
        sorted_mins.sort()  # 按时间排序
        Delta1_init = times[sorted_mins[0]]
        Delta2_init = times[sorted_mins[1]]
        print(f"  检测到两个极小值位置: {Delta1_init:.1f}h, {Delta2_init:.1f}h")
    else:
        # 没找到明显的两个极小值，手动设置
        Delta1_init = times[0] + time_span * 0.3
        Delta2_init = times[0] + time_span * 0.7
        print(f"  未检测到两个明显极小值，使用默认位置: {Delta1_init:.1f}h, {Delta2_init:.1f}h")
    
    # 确保Delta2 > Delta1 + 最小间隔
    min_separation = time_span * 0.2
    if Delta2_init < Delta1_init + min_separation:
        Delta2_init = Delta1_init + min_separation
    
    A1_init = (flux_max - flux_min) * 0.3
    A2_init = (flux_max - flux_min) * 0.3
    tau1_init = time_span / 15
    tau2_init = time_span / 15
    
    fred2_fit = TF1(f"fred2_{rig}", double_fred_func, times[0], times[-1], 9)
    fred2_fit.SetParameters(baseline_init, A1_init, Delta1_init, tau1_init, xi_init,
                            A2_init, Delta2_init, tau2_init, xi_init)
    fred2_fit.SetParNames("baseline", "A1", "Delta1", "tau1", "xi1",
                          "A2", "Delta2", "tau2", "xi2")
    fred2_fit.SetLineColor(ROOT.kBlue)
    fred2_fit.SetLineWidth(2)
    fred2_fit.SetLineStyle(2)  # 虚线
    
    # 参数限制 - 强制两个FRED分开
    fred2_fit.SetParLimits(0, flux_min * 0.8, flux_max * 1.2)                    # baseline
    fred2_fit.SetParLimits(1, 0, (flux_max - flux_min) * 1.0)                    # A1
    fred2_fit.SetParLimits(2, times[0], Delta2_init - min_separation)            # Delta1 必须早于Delta2
    fred2_fit.SetParLimits(3, 0.5, time_span / 3)                                # tau1
    fred2_fit.SetParLimits(4, 0.1, 5)                                             # xi1
    fred2_fit.SetParLimits(5, 0, (flux_max - flux_min) * 1.0)                    # A2
    fred2_fit.SetParLimits(6, Delta1_init + min_separation, times[-1])          # Delta2 必须晚于Delta1
    fred2_fit.SetParLimits(7, 0.5, time_span / 3)                                # tau2
    fred2_fit.SetParLimits(8, 0.1, 5)                                             # xi2
    
    # 固定baseline，减少自由度
    fred2_fit.FixParameter(0, baseline_init)
    
    gr.Fit(fred2_fit, "RQ+")
    
    # 绘制图例
    leg = TLegend(0.65, 0.7, 0.88, 0.88)
    leg.SetTextSize(0.03)
    leg.AddEntry(gr, "Data", "lp")
    leg.AddEntry(fred1_fit, "FRED1 (Single)", "l")
    leg.AddEntry(fred2_fit, "FRED2 (Double)", "l")
    leg.Draw()
    
    c1.Update()
    c1.SaveAs(os.path.join(OUT_DIR, f"flux_time_{rig.replace('.', 'p')}_double.png"))
    print(f"\n[OK] 保存 {rig} 的 flux~time 拟合图（单/双FRED）")
    
    # 输出拟合参数
    chi2_1 = fred1_fit.GetChisquare()
    ndf_1 = fred1_fit.GetNDF()
    chi2_2 = fred2_fit.GetChisquare()
    ndf_2 = fred2_fit.GetNDF()
    
    print(f"  FRED1: Chi2/NDF = {chi2_1:.2f}/{ndf_1} = {chi2_1/ndf_1 if ndf_1>0 else 0:.3f}")
    print(f"    Delta={fred1_fit.GetParameter(2):.2f}h, A={fred1_fit.GetParameter(1):.3e}")
    print(f"  FRED2: Chi2/NDF = {chi2_2:.2f}/{ndf_2} = {chi2_2/ndf_2 if ndf_2>0 else 0:.3f}")
    print(f"    Delta1={fred2_fit.GetParameter(2):.2f}h, A1={fred2_fit.GetParameter(1):.3e}")
    print(f"    Delta2={fred2_fit.GetParameter(6):.2f}h, A2={fred2_fit.GetParameter(5):.3e}")


# ====================== 2. 能谱拟合：直接拟合 flux~R，画图时乘以R^2.7 ======================

# 标准SBPL函数: flux(R) = C(R/45)^(gamma1) × [1 + (R/R0)^(gamma2/s)]^s
def sbpl_func(x, par):
    """
    标准SBPL能谱函数（直接拟合flux）
    flux(R) = C(R/45)^(gamma1) × [1 + (R/R0)^(gamma2/s)]^s
    
    par[0] = C (归一化常数)
    par[1] = gamma1 (低能谱指数)
    par[2] = R0 (断点刚度)
    par[3] = gamma2 (谱指数变化)
    par[4] = s (平滑参数)
    """
    R = x[0]
    C = par[0]
    gamma1 = par[1]
    R0 = par[2]
    gamma2 = par[3]
    s = par[4]
    
    if R <= 0:
        return 0
    
    R0 = max(R0, MIN_FLOAT)
    s = max(s, MIN_FLOAT)
    
    # flux(R) = C * (R/45)^gamma1 * [1 + (R/R0)^(gamma2/s)]^s
    power_term = ROOT.TMath.Power(R / 45.0, gamma1)
    bracket_term = ROOT.TMath.Power(1.0 + ROOT.TMath.Power(R / R0, gamma2 / s), -s)
    
    return C * power_term * bracket_term


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

print(f"\n[INFO] 刚度范围: {rigidity_centers.min():.2f} - {rigidity_centers.max():.2f} GV")
print(f"[INFO] 共有 {len(rigidity_centers)} 个刚度区间")

# 选择几个代表性时刻绘制能谱
n_times = len(ams)
time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]

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
    
    # ===== 第一步：用原始flux数据拟合 =====
    gr_fit = TGraphErrors(len(rig_valid))
    for i in range(len(rig_valid)):
        gr_fit.SetPoint(i, rig_valid[i], flux_valid[i])
        gr_fit.SetPointError(i, rig_err_valid[i], flux_err_valid[i])
    
    # SBPL拟合（拟合原始flux）
    fit_min = rig_valid.min()
    fit_max = rig_valid.max()
    sbpl_fit = TF1(f"sbpl_{idx}", sbpl_func, fit_min, fit_max)
    sbpl_fit.SetParNames("C", "gamma1", "R0", "gamma2", "s")
    
    # 初值估计
    R_mid = float(np.sqrt(fit_min * fit_max))
    flux_mid = float(flux_valid[np.argmin(np.abs(rig_valid - R_mid))])
    
    gamma1_init = -2.7
    s_init = 0.5
    # 在断点处估计C
    C_init = flux_mid / (np.power(R_mid / 45.0, gamma1_init) * np.power(2.0, s_init))
    R0_init = R_mid
    gamma2_init = 0.3
    
    sbpl_fit.SetParameters(C_init, gamma1_init, R0_init, gamma2_init, s_init)
    sbpl_fit.SetParLimits(0, C_init * 1e-3, C_init * 1e3)     # C > 0
    sbpl_fit.SetParLimits(1, -4.0, -1.0)                       # gamma1 < 0
    sbpl_fit.SetParLimits(2, fit_min * 0.5, fit_max * 2.0)   # R0
    sbpl_fit.SetParLimits(3, -2.0, 2.0)                        # gamma2
    sbpl_fit.SetParLimits(4, 0.1, 3.0)                         # s > 0
    
    sbpl_fit.SetLineColor(ROOT.kRed)
    sbpl_fit.SetLineWidth(2)
    
    # 执行拟合（不显示）
    fit_result = gr_fit.Fit(sbpl_fit, "RSNQ")  # S: 保存结果, N: 不画, Q: 安静
    
    chi2 = sbpl_fit.GetChisquare()
    ndf = sbpl_fit.GetNDF()
    print(f"       SBPL拟合: C={sbpl_fit.GetParameter(0):.3e}, "
          f"gamma1={sbpl_fit.GetParameter(1):.3f}, "
          f"R0={sbpl_fit.GetParameter(2):.3f} GV, "
          f"gamma2={sbpl_fit.GetParameter(3):.3f}, "
          f"s={sbpl_fit.GetParameter(4):.3f}")
    print(f"       Chi2/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.3f}")
    
    # ===== 第二步：画图时把所有东西都乘以R^2.7 =====
    flux_display = flux_valid * np.power(rig_valid, 2.7)
    flux_err_display = flux_err_valid * np.power(rig_valid, 2.7)
    
    gr_display = TGraphErrors(len(rig_valid))
    for i in range(len(rig_valid)):
        gr_display.SetPoint(i, rig_valid[i], flux_display[i])
        gr_display.SetPointError(i, rig_err_valid[i], flux_err_display[i])
    
    gr_display.SetMarkerStyle(20)
    gr_display.SetMarkerSize(0.8)
    gr_display.SetMarkerColor(ROOT.kBlack)
    gr_display.SetLineColor(ROOT.kBlack)
    
    # 创建画布
    c_spec = TCanvas(f"c_spectrum_{idx}", f"Flux Spectrum {timestamp.strftime('%m-%d %H:%M')}", 1000, 700)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    
    gr_display.SetTitle(f"Proton Flux Spectrum {timestamp.strftime('%m-%d %H:%M')};Rigidity [GV];Flux #times R^{{2.7}}")
    gr_display.Draw("AP")
    
    # 创建拟合曲线的显示版本（拟合函数值乘以R^2.7）
    fit_display = TF1(f"fit_display_{idx}", 
                      f"[0]*TMath::Power(x/45.0, [1])*TMath::Power(1+TMath::Power(x/[2], [3]/[4]), [4]) * TMath::Power(x, 2.7)",
                      fit_min, fit_max)
    # 复制拟合参数
    for i in range(5):
        fit_display.SetParameter(i, sbpl_fit.GetParameter(i))
    fit_display.SetLineColor(ROOT.kRed)
    fit_display.SetLineWidth(2)
    fit_display.Draw("SAME")
    
    # 设置图例
    ROOT.gPad.Update()
    gr_display.GetXaxis().SetLimits(rig_valid.min() * 0.8, rig_valid.max() * 1.2)
    leg = TLegend(0.15, 0.7, 0.5, 0.85)
    leg.SetTextSize(0.03)
    leg.AddEntry(gr_display, "Data (Flux #times R^{2.7})", "lp")
    leg.AddEntry(fit_display, "SBPL fit", "l")
    leg.Draw()
    
    # 标注拟合函数
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.035)
    latex.SetTextColor(ROOT.kBlue)
    latex.DrawLatex(0.12, 0.15, "Flux(R) = C ( #frac{R}{45} )^{#gamma_{1}} [ 1 + ( #frac{R}{R_{0}} )^{ #frac{#gamma_{2}}{s} } ]^{s}")
    
    c_spec.Update()
    out_name = os.path.join(OUT_DIR, f"flux_spectrum_modified_{timestamp.strftime('%m%d_%H%M')}.png")
    c_spec.SaveAs(out_name)
    print(f"[OK] 保存能谱图: {out_name}")

print(f"\n✅ 全部ROOT绘图和拟合完成，结果保存在 {OUT_DIR}/")