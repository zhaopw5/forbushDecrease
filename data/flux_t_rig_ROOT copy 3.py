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

# ====================== 2. flux~rigidity 图 + SBPL拟合 ======================

# SBPL 函数字符串（直接用刚度 R 作为自变量）
sbpl_func = (
    "[0]*TMath::Power(x/[1], -[2]) * "
    "TMath::Power(1 + TMath::Power(x/[1], ([3]-[2])/[4]), -[4])"
)

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

# 创建画布 - SBPL 拟合
c2 = TCanvas("c_spectrum", "Flux Spectra (Smooth Broken Power-Law Fit)", 1000, 700)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetGrid()

# 每个时刻单独画布绘制；colors 保留复用
colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]

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
    gr_spec.SetMarkerColor(ROOT.kBlack)
    gr_spec.SetLineColor(ROOT.kBlack)
    
    # 为该时刻创建独立画布并绘制
    c_spec = TCanvas(f"c_spectrum_{idx}", f"Flux Spectrum (SBPL) {timestamp.strftime('%m-%d %H:%M')}", 1000, 700)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    # ROOT.gPad.SetGrid()
    gr_spec.SetTitle(f"Proton Flux Spectrum {timestamp.strftime('%m-%d %H:%M')};Rigidity [GV];Flux")
    gr_spec.Draw("AP")
    
    # SBPL 拟合: N(R) = A * (R/E_break)^(-Gamma1) * [1 + (R/E_break)^((Gamma2-Gamma1)/s)]^(-s)
    fit_min = rig_valid.min()
    fit_max = rig_valid.max()
    sbpl_fit = TF1(f"sbpl_{idx}", sbpl_func, fit_min, fit_max)
    sbpl_fit.SetParNames("A", "E_break", "Gamma1", "Gamma2", "s")

    # 极简初值与宽松范围
    R_min = rig_valid.min()
    R_max = rig_valid.max()
    # 取几何中值作为断点
    R_break_init = float(np.sqrt(max(R_min, MIN_FLOAT) * max(R_max, MIN_FLOAT)))
    s_init = 1.0
    # 用最接近断点的点估计 A（使 N(E_break) ≈ F_near，A=F_near*2^s）
    i_near = int(np.argmin(np.abs(rig_valid - R_break_init)))
    A_init = float(max(flux_valid[i_near], MIN_FLOAT) * (2.0 ** s_init))
    Gamma1_init = 2.0
    Gamma2_init = 3.0

    sbpl_fit.SetParameters(A_init, R_break_init, Gamma1_init, Gamma2_init, s_init)
    sbpl_fit.SetParLimits(0, max(A_init * 1e-3, MIN_FLOAT), A_init * 1e3)  # A > 0
    sbpl_fit.SetParLimits(1, max(0.5 * R_min, MIN_FLOAT), 2.0 * R_max)     # E_break > 0
    sbpl_fit.SetParLimits(2, 0.0, 10.0)                                     # Gamma1 ≥ 0
    sbpl_fit.SetParLimits(3, 0.0, 10.0)                                     # Gamma2 ≥ 0
    sbpl_fit.SetParLimits(4, 0.1, 5.0)                                      # s > 0

    sbpl_fit.SetLineColor(ROOT.kRed)
    sbpl_fit.SetLineWidth(2)
    sbpl_fit.SetLineStyle(1)
    fit_result = gr_spec.Fit(sbpl_fit, "RSMQ")
    
    chi2 = sbpl_fit.GetChisquare()
    ndf = sbpl_fit.GetNDF()
    print(f"       SBPL拟合: A={sbpl_fit.GetParameter(0):.3e}, "
          f"E_break={sbpl_fit.GetParameter(1):.3f} GV, "
          f"Gamma1={sbpl_fit.GetParameter(2):.3f}, "
          f"Gamma2={sbpl_fit.GetParameter(3):.3f}, "
          f"s={sbpl_fit.GetParameter(4):.3f}")
    print(f"       Chi2/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")
    
    # 设置坐标范围与图例并保存
    ROOT.gPad.Update()
    gr_spec.GetXaxis().SetLimits(rig_valid.min() * 0.8, rig_valid.max() * 1.2)
    leg = TLegend(0.15, 0.3, 0.45, 0.4)
    leg.SetTextSize(0.03)
    leg.AddEntry(gr_spec, "Data", "lp")
    leg.AddEntry(sbpl_fit, "SBPL fit", "l")
    leg.Draw()
    # 在图上标注拟合函数表达式（左上角，避免与右上角拟合信息框重叠）
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.045)
    latex.SetTextColor(ROOT.kBlue)
    latex.DrawLatex(0.12, 0.15, "Flux(R) = A ( #frac{R}{E_{break}} )^{-#Gamma_{1}} [ 1 + ( #frac{R}{E_{break}} )^{ #frac{#Gamma_{2}-#Gamma_{1}}{s} } ]^{-s}")
    c_spec.Update()
    out_name = os.path.join(OUT_DIR, f"flux_spectrum_sbpl_{timestamp.strftime('%m%d_%H%M')}.png")
    c_spec.SaveAs(out_name)
    print(f"[OK] 保存能谱图: {out_name}")

print(f"\n✅ 全部ROOT绘图和拟合完成，结果保存在 {OUT_DIR}/")