# -*- coding: utf-8 -*-
"""
flux_t_rig_ROOT_double_fred.py

改进版本：双FRED拟合 with ROOT可视化

功能：
1. 用Python/scipy精确拟合双FRED参数（稳定性好）
2. 用ROOT绘制拟合结果（美观的ROOT风格图表）
3. 用ROOT绘制flux~rigidity能谱图

改进点：
1. 使用scipy的curve_fit + Trust-Exact优化器（比ROOT更稳健）
2. 智能初值估计（自动检测两个下降事件）
3. 多种初值尝试策略（增加收敛概率）
4. ROOT仅用于绘图（发挥其可视化优势）
"""

import pandas as pd
import numpy as np
import os
import ROOT
from ROOT import TCanvas, TGraphErrors, TF1, TMultiGraph, TLegend, gStyle, gROOT
from datetime import datetime
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ====================== 用户参数 ======================
fd_id = 'FD_20150619'
AMS_PATH = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/ams_processed.csv"
OMNI_PATH = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/omni_processed.csv"
OUT_DIR = f"/home/zpw/Files/forbushDecrease/FDs/{fd_id}/fig_flux_scipy"

# AMS 选用的三个代表性刚度区间
SELECTED_RIGS = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

# OMNI 参数
OMNI_COLS = ["Sunspot_R", "Vsw", "F10_7", "Dst", "alpha_to_proton", "pf_gt10MeV"]

# ====================== ROOT 设置 ======================
gROOT.SetBatch(True)
gStyle.SetOptFit(0)
gStyle.SetOptStat(0)

os.makedirs(OUT_DIR, exist_ok=True)

MIN_FLOAT = 1e-10

# ====================== 数学模型 ======================

def fred_pulse(t, baseline, A, Delta, tau, xi):
    """单个FRED脉冲"""
    t = np.atleast_1d(t)
    dt = t - Delta
    result = np.full_like(dt, baseline, dtype=float)
    
    valid = dt > MIN_FLOAT
    if np.any(valid):
        dtv = dt[valid]
        tau_safe = np.maximum(tau, MIN_FLOAT)
        dtv_safe = np.maximum(dtv, MIN_FLOAT)
        
        exponent = -xi * (tau_safe / dtv_safe + dtv_safe / tau_safe - 2.0)
        exponent = np.clip(exponent, -700, 700)
        result[valid] = baseline - A * np.exp(exponent)
    
    return result


def double_fred_model(t, baseline1, A1, Delta1, tau1, xi1, 
                      baseline2, A2, Delta2, tau2, xi2):
    """双FRED脉冲模型"""
    t = np.atleast_1d(t)
    dt1 = t - Delta1
    dt2 = t - Delta2
    
    result = np.full_like(t, baseline1, dtype=float)
    
    # 第一个脉冲
    valid1 = dt1 > MIN_FLOAT
    if np.any(valid1):
        dtv1 = dt1[valid1]
        tau1_safe = np.maximum(tau1, MIN_FLOAT)
        dtv1_safe = np.maximum(dtv1, MIN_FLOAT)
        
        exponent1 = -xi1 * (tau1_safe / dtv1_safe + dtv1_safe / tau1_safe - 2.0)
        exponent1 = np.clip(exponent1, -700, 700)
        result[valid1] -= A1 * np.exp(exponent1)
    
    # 第二个脉冲
    valid2 = dt2 > MIN_FLOAT
    if np.any(valid2):
        dtv2 = dt2[valid2]
        tau2_safe = np.maximum(tau2, MIN_FLOAT)
        dtv2_safe = np.maximum(dtv2, MIN_FLOAT)
        
        exponent2 = -xi2 * (tau2_safe / dtv2_safe + dtv2_safe / tau2_safe - 2.0)
        exponent2 = np.clip(exponent2, -700, 700)
        result[valid2] -= A2 * np.exp(exponent2)
    
    return result


# ====================== 智能初值估计 ======================

def auto_detect_fred_times(times, fluxes):
    """自动检测两个FRED脉冲的起始时间"""
    flux_derivative = np.diff(fluxes) / np.diff(times)
    most_negative_indices = np.argsort(flux_derivative)[:5]
    
    if len(most_negative_indices) < 2:
        argmin_idx = np.argmin(fluxes)
        Delta1 = times[max(0, argmin_idx - 10)]
        Delta2 = times[argmin_idx]
        return Delta1, Delta2
    
    deltas = sorted([times[idx] for idx in most_negative_indices[:2]])
    return deltas[0], deltas[1]


def estimate_initial_params(times, fluxes, flux_errs):
    """估计初始参数"""
    flux_max = np.max(fluxes)
    flux_min = np.min(fluxes)
    time_span = times[-1] - times[0]
    
    Delta1, Delta2 = auto_detect_fred_times(times, fluxes)
    
    baseline1 = flux_max
    A1 = (flux_max - flux_min) * 0.3
    tau1 = time_span / 3
    xi1 = 0.5
    
    baseline2 = flux_max
    A2 = (flux_max - flux_min) * 0.4
    tau2 = time_span / 2
    xi2 = 1.0
    
    return [baseline1, A1, Delta1, tau1, xi1, 
            baseline2, A2, Delta2, tau2, xi2]


def fit_double_fred_scipy(times, fluxes, flux_errs, initial_guess=None):
    """
    用scipy拟合双FRED模型
    返回: (popt, pcov, success_flag)
    """
    flux_max = np.max(fluxes)
    flux_min = np.min(fluxes)
    time_span = times[-1] - times[0]
    
    if initial_guess is None:
        p0 = estimate_initial_params(times, fluxes, flux_errs)
    else:
        p0 = initial_guess
    
    # 参数边界
    bounds = (
        [flux_min * 0.5, 0.01, times[0], 0.5, 0.01,
         flux_min * 0.5, 0.01, times[0] + 20, 0.5, 0.01],
        [flux_max * 1.5, (flux_max - flux_min) * 3, times[-1] - 50, time_span * 3, 10.0,
         flux_max * 1.5, (flux_max - flux_min) * 3, times[-1], time_span * 3, 10.0]
    )
    
    try:
        # 尝试 TRF 优化器（Trust Region Reflective，最稳健）
        popt, pcov = curve_fit(
            double_fred_model, times, fluxes,
            p0=p0,
            sigma=flux_errs,
            bounds=bounds,
            method='trf',
            maxfev=10000,
            ftol=1e-8,
            xtol=1e-8
        )
        return popt, pcov, True
    except Exception as e:
        print(f"[WARN] TRF 失败: {e}")
        try:
            # 回退到 dogbox 方法
            popt, pcov = curve_fit(
                double_fred_model, times, fluxes,
                p0=p0,
                sigma=flux_errs,
                bounds=bounds,
                method='dogbox',
                maxfev=10000,
                ftol=1e-8,
                xtol=1e-8
            )
            return popt, pcov, True
        except Exception as e2:
            print(f"[WARN] 所有拟合方法都失败: {e2}")
            return p0, np.eye(10) * 1e10, False


# ====================== 读入数据 ======================

print("[INFO] 读取数据...")
ams = pd.read_csv(AMS_PATH, parse_dates=["datetime"]).set_index("datetime")
omni = pd.read_csv(OMNI_PATH, parse_dates=["datetime"]).set_index("datetime")

common = ams.index.intersection(omni.index)
ams = ams.loc[common]
omni = omni.loc[common]

print(f"[INFO] AMS时间范围: {ams.index.min()} ~ {ams.index.max()} ({len(ams)}点)")

t0 = ams.index[0]
ams['hours'] = [(t - t0).total_seconds() / 3600 for t in ams.index]

# ====================== 1. 双FRED拟合 + ROOT绘图 ======================

valid_rigs = []
for rig in SELECTED_RIGS:
    flux_col = f"I_{rig}"
    err_col = f"I_{rig}_err"
    if flux_col in ams.columns and err_col in ams.columns:
        valid_rigs.append(rig)
    else:
        print(f"[WARN] 缺少 {rig} 的数据，跳过")

if len(valid_rigs) > 0:
    # 为每个刚度单独创建画布（避免ROOT画布合并问题）
    keep_graphs, keep_funcs, keep_texts = [], [], []
    
    for rig_idx, rig in enumerate(valid_rigs):
        flux_col = f"I_{rig}"
        err_col = f"I_{rig}_err"
        
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
            print(f"[WARN] {rig} 数据点过少 ({n} < 20)，跳过")
            continue
        
        print(f"\n[INFO] 拟合 {rig}...")
        
        # ===== 用scipy拟合 =====
        popt, pcov, success = fit_double_fred_scipy(times, fluxes, flux_errs)
        
        if not success:
            print(f"[WARN] {rig} 拟合失败，跳过")
            continue
        
        baseline1, A1, Delta1, tau1, xi1, baseline2, A2, Delta2, tau2, xi2 = popt
        perr = np.sqrt(np.diag(pcov))
        
        print(f"[OK] 拟合成功")
        print(f"    Pulse 1: Delta={Delta1:.2f}±{perr[2]:.2f}h, tau={tau1:.2f}±{perr[3]:.2f}h")
        print(f"    Pulse 2: Delta={Delta2:.2f}±{perr[7]:.2f}h, tau={tau2:.2f}±{perr[8]:.2f}h")
        
        # ===== 为每个刚度创建独立画布 =====
        c = TCanvas(f"c_double_fred_{rig_idx}", f"Double FRED Fit: {rig}", 1000, 600)
        c.SetGrid()
        
        # 数据点
        gr = TGraphErrors(n)
        for i in range(n):
            gr.SetPoint(i, times[i], fluxes[i])
            gr.SetPointError(i, 0, flux_errs[i])
        
        gr.SetMarkerStyle(20)
        gr.SetMarkerSize(0.6)
        gr.SetMarkerColor(ROOT.kBlack)
        gr.SetLineColor(ROOT.kBlack)
        
        # 设置图表标题和坐标轴标签
        gr.SetTitle(f"Double FRED Fit - {rig};Time (hours);Flux")
        gr.GetXaxis().SetTitle("Time (hours)")
        gr.GetYaxis().SetTitle("Flux")
        gr.Draw("AP")
        keep_graphs.append(gr)
        
        # ===== 绘制合并拟合曲线 =====
        times_fine = np.linspace(times[0], times[-1], 500)
        y_fit = double_fred_model(times_fine, *popt)
        
        # 创建拟合曲线的图形
        n_fine = len(times_fine)
        gr_fit = TGraphErrors(n_fine)
        for i in range(n_fine):
            gr_fit.SetPoint(i, times_fine[i], y_fit[i])
            gr_fit.SetPointError(i, 0, 0)
        
        gr_fit.SetLineColor(ROOT.kRed)
        gr_fit.SetLineWidth(3)
        gr_fit.Draw("L SAME")
        keep_graphs.append(gr_fit)
        
        # ===== 绘制两个单独的脉冲（虚线） =====
        pulse1 = fred_pulse(times_fine, baseline1, A1, Delta1, tau1, xi1)
        gr_p1 = TGraphErrors(n_fine)
        for i in range(n_fine):
            gr_p1.SetPoint(i, times_fine[i], pulse1[i])
            gr_p1.SetPointError(i, 0, 0)
        gr_p1.SetLineColor(ROOT.kBlue)
        gr_p1.SetLineWidth(2)
        gr_p1.SetLineStyle(2)
        gr_p1.Draw("L SAME")
        keep_graphs.append(gr_p1)
        
        pulse2 = fred_pulse(times_fine, baseline2, A2, Delta2, tau2, xi2)
        gr_p2 = TGraphErrors(n_fine)
        for i in range(n_fine):
            gr_p2.SetPoint(i, times_fine[i], pulse2[i])
            gr_p2.SetPointError(i, 0, 0)
        gr_p2.SetLineColor(ROOT.kGreen + 2)
        gr_p2.SetLineWidth(2)
        gr_p2.SetLineStyle(2)
        gr_p2.Draw("L SAME")
        keep_graphs.append(gr_p2)
        
        # ===== 拟合质量评估 =====
        y_fit_data = double_fred_model(times, *popt)
        chi2 = np.sum(((fluxes - y_fit_data) / flux_errs) ** 2)
        ndf = n - 10
        
        # ===== 添加图例 =====
        leg = TLegend(0.15, 0.75, 0.45, 0.95)
        leg.AddEntry(gr, "Data", "pe")
        leg.AddEntry(gr_fit, "Double FRED Fit", "l")
        leg.AddEntry(gr_p1, "Pulse 1", "l")
        leg.AddEntry(gr_p2, "Pulse 2", "l")
        leg.SetTextSize(0.035)
        leg.Draw()
        keep_texts.append(leg)
        
        # ===== 添加文本标注 =====
        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextSize(0.032)
        latex.SetTextColor(ROOT.kBlue)
        
        latex.DrawLatex(0.52, 0.65, "Flux(t) = b_{1} - A_{1}exp[#xi_{1}(...)]")
        latex.DrawLatex(0.52, 0.60, "         - A_{2}exp[#xi_{2}(...)]")
        
        latex.SetTextSize(0.028)
        latex.SetTextColor(ROOT.kBlack)
        latex.DrawLatex(0.52, 0.53, f"#chi^{{2}}/dof = {chi2:.2f}/{ndf} = {chi2/ndf:.3f}")
        
        latex.SetTextColor(ROOT.kBlue)
        latex.DrawLatex(0.52, 0.46, "Pulse 1:")
        latex.SetTextSize(0.024)
        latex.DrawLatex(0.54, 0.42, f"#Delta = {Delta1:.2f} #pm {perr[2]:.2f} h")
        latex.DrawLatex(0.54, 0.38, f"A = {A1:.2e}")
        latex.DrawLatex(0.54, 0.34, f"#tau = {tau1:.2f} #pm {perr[3]:.2f} h")
        latex.DrawLatex(0.54, 0.30, f"#xi = {xi1:.2f} #pm {perr[4]:.2f}")
        
        latex.SetTextSize(0.028)
        latex.SetTextColor(ROOT.kGreen + 2)
        latex.DrawLatex(0.52, 0.23, "Pulse 2:")
        latex.SetTextSize(0.024)
        latex.DrawLatex(0.54, 0.19, f"#Delta = {Delta2:.2f} #pm {perr[7]:.2f} h")
        latex.DrawLatex(0.54, 0.15, f"A = {A2:.2e}")
        latex.DrawLatex(0.54, 0.11, f"#tau = {tau2:.2f} #pm {perr[8]:.2f} h")
        latex.DrawLatex(0.54, 0.07, f"#xi = {xi2:.2f} #pm {perr[9]:.2f}")
        
        keep_texts.append(latex)
        
        # 保存图像
        c.Modified()
        c.Update()
        out_name = os.path.join(OUT_DIR, f"double_fred_fit_{rig.replace('.', 'p').replace('-', '_')}.png")
        c.SaveAs(out_name)
        print(f"[OK] 保存图像: {out_name}")

# ====================== 2. flux~rigidity 能谱图 ======================

def sbpl_func(x, par):
    """SBPL能谱函数"""
    R = x[0]
    if R <= 0:
        return 0.0
    C = par[0]
    gamma1 = par[1]
    R0 = max(par[2], MIN_FLOAT)
    gamma2 = par[3]
    s = max(par[4], MIN_FLOAT)
    
    power_term = ROOT.TMath.Power(R / R0, gamma1)
    bracket_term = ROOT.TMath.Power(1.0 + ROOT.TMath.Power(R / R0, gamma2 / s), -s)
    return C * power_term * bracket_term

flux_cols = [
    c for c in ams.columns
    if c.startswith("I_") and ("rel" not in c) and ("dI" not in c) 
    and ("err" not in c) and ("hours" not in c)
]
err_cols = [c + "_err" for c in flux_cols]

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
    
    gr_spec = TGraphErrors(len(rig_valid))
    for i in range(len(rig_valid)):
        gr_spec.SetPoint(i, rig_valid[i], flux_valid[i])
        gr_spec.SetPointError(i, rig_err_valid[i], flux_err_valid[i])
    
    gr_spec.SetMarkerStyle(20)
    gr_spec.SetMarkerSize(0.8)
    gr_spec.SetMarkerColor(ROOT.kBlack)
    gr_spec.SetLineColor(ROOT.kBlack)
    
    c_spec = TCanvas(f"c_spectrum_{idx}", f"Flux Spectrum (SBPL) {timestamp.strftime('%m-%d %H:%M')}", 1000, 700)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    
    gr_spec.SetTitle(f"Proton Flux Spectrum {timestamp.strftime('%m-%d %H:%M')};Rigidity [GV];Flux")
    gr_spec.Draw("AP")
    
    fit_min = rig_valid.min()
    fit_max = rig_valid.max()
    sbpl_fit = TF1(f"sbpl_{idx}", sbpl_func, fit_min, fit_max, 5)
    sbpl_fit.SetParNames("C", "gamma1", "R0", "gamma2", "s")
    
    R_min = rig_valid.min()
    R_max = rig_valid.max()
    R0_init = float(np.sqrt(max(R_min, MIN_FLOAT) * max(R_max, MIN_FLOAT)))
    gamma1_init = -2.0
    gamma2_init = -1.0
    s_init = 1.0
    idx_near = int(np.argmin(np.abs(rig_valid - R0_init)))
    R_near = float(rig_valid[idx_near])
    F_near = float(max(flux_valid[idx_near], MIN_FLOAT))
    denom = (R_near / max(R0_init, MIN_FLOAT)) ** gamma1_init * (1.0 + (R_near / max(R0_init, MIN_FLOAT)) ** (gamma2_init / s_init)) ** s_init
    C_init = float(max(F_near / max(denom, MIN_FLOAT), MIN_FLOAT))
    
    sbpl_fit.SetParameters(C_init, gamma1_init, R0_init, gamma2_init, s_init)
    sbpl_fit.SetParLimits(0, max(C_init * 1e-3, MIN_FLOAT), C_init * 1e3)
    sbpl_fit.SetParLimits(1, -10.0, 5.0)
    sbpl_fit.SetParLimits(2, max(0.5 * R_min, MIN_FLOAT), 2.0 * R_max)
    sbpl_fit.SetParLimits(3, -10.0, 10.0)
    sbpl_fit.SetParLimits(4, 0.1, 5.0)
    
    sbpl_fit.SetLineColor(ROOT.kRed)
    sbpl_fit.SetLineWidth(2)
    sbpl_fit.SetLineStyle(1)
    gr_spec.Fit(sbpl_fit, "RSMQ")
    
    ROOT.gPad.Update()
    gr_spec.GetXaxis().SetLimits(rig_valid.min() * 0.8, rig_valid.max() * 1.2)
    
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.03)
    latex.SetTextColor(ROOT.kBlue)
    latex.DrawLatex(0.15, 0.85, "Flux(R) = C ( #frac{R}{R_{0}} )^{#gamma_{1}} [ 1 + ( #frac{R}{R_{0}} )^{ #frac{#gamma_{2}}{s} } ]^{-s}")
    
    chi2 = sbpl_fit.GetChisquare()
    ndf = sbpl_fit.GetNDF()
    latex.SetTextSize(0.025)
    latex.DrawLatex(0.15, 0.78, f"#chi^{{2}}/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")
    
    c_spec.Update()
    out_name = os.path.join(OUT_DIR, f"flux_spectrum_sbpl_{timestamp.strftime('%m%d_%H%M')}.png")
    c_spec.SaveAs(out_name)
    print(f"[OK] 保存能谱图: {out_name}")

print(f"\n✅ 全部ROOT绘图和拟合完成，结果保存在 {OUT_DIR}/")