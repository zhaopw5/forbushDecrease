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
AMS_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20211101/ams_processed.csv"
OMNI_PATH = "/home/zpw/Files/forbushDecrease/data/FD_20211101/omni_processed.csv"
OUT_DIR = "fig_flux_ROOT"

# AMS 选用的三个代表性刚度区间
SELECTED_RIGS = ["1.00-1.16GV", "5.37-5.90GV", "22.80-33.50GV"]

# OMNI 参数
OMNI_COLS = ["Sunspot_R", "Vsw", "F10_7", "Dst", "alpha_to_proton", "pf_gt10MeV"]

# ====================== ROOT 设置 ======================
gROOT.SetBatch(True)  # 批处理模式
gStyle.SetOptFit(0)    # 不显示拟合参数框
gStyle.SetOptStat(0)   # 不显示统计框

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

# ====================== 辅助函数：hours 转日期格式标签 ======================
def hours_to_datestr(hours, t0_ref):
    """将相对小时数转换为日期字符串，用于 X 轴标签"""
    from datetime import timedelta
    dt = t0_ref + timedelta(hours=hours)
    return dt.strftime("%m-%d\n%H:%M")

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

# 为每个刚度创建 flux~time 图并拟合（合并到一张 3 行 1 列画布）
# 先筛选真实可用的刚度列表
valid_rigs = []
for rig in SELECTED_RIGS:
    flux_col = f"I_{rig}"
    err_col = f"I_{rig}_err"
    if flux_col in ams.columns and err_col in ams.columns:
        valid_rigs.append(rig)
    else:
        print(f"[WARN] 缺少 {rig} 的数据，跳过")

if len(valid_rigs) == 0:
    print("[WARN] 所选刚度均无可用数据，跳过 flux~time 绘图")
else:
    # 创建合并画布：N 行 1 列
    c_time_all = TCanvas("c_time_all", "Flux vs Time (Selected Rigidities)", 800, 300 * len(valid_rigs))
    c_time_all.Divide(1, len(valid_rigs))

    # 保持对象引用，避免被 PyROOT 回收
    keep_graphs, keep_funcs, keep_texts = [], [], []

    for pad_idx, rig in enumerate(valid_rigs):
        flux_col = f"I_{rig}"
        err_col = f"I_{rig}_err"

        # 切换到对应子画布
        pad = c_time_all.cd(pad_idx + 1)
        pad.SetGrid()

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
            print(f"[WARN] {rig} 无有效点，跳过该子图")
            continue

        # 创建TGraphErrors
        gr = TGraphErrors(n)
        for i in range(n):
            gr.SetPoint(i, times[i], fluxes[i])
            gr.SetPointError(i, 0, flux_errs[i])

        gr.SetMarkerStyle(20)
        gr.SetMarkerSize(0.5)
        gr.SetMarkerColor(ROOT.kBlack)
        gr.SetLineColor(ROOT.kBlack)
        gr.SetTitle(f"{rig};;Flux")
        gr.Draw("AP")
        # 持有引用
        keep_graphs.append(gr)

        # 拟合代码（保持不变）
        flux_max = np.max(fluxes)
        flux_min = np.min(fluxes)
        baseline_init = flux_max
        A_init = flux_max - flux_min
        Delta_init = times[np.argmin(fluxes)] - (times[-1] - times[0]) / 10
        tau_init = (times[-1] - times[0]) / 10
        xi_init = 1.0

        fred_fit = TF1(f"fred_{rig}", negative_fred_func, times[0], times[-1], 5)
        fred_fit.SetParameters(baseline_init, A_init, Delta_init, tau_init, xi_init)
        fred_fit.SetParNames("baseline", "A", "Delta", "tau", "xi")
        fred_fit.SetLineColor(ROOT.kRed)
        fred_fit.SetLineWidth(2)

        fred_fit.SetParLimits(0, flux_min * 0.8, flux_max * 1.2)
        fred_fit.SetParLimits(1, 0, (flux_max - flux_min) * 2)
        fred_fit.SetParLimits(2, times[0], times[-1])
        fred_fit.SetParLimits(3, 0.1, (times[-1] - times[0]))
        fred_fit.SetParLimits(4, 0.01, 10)

        gr.Fit(fred_fit, "RQ")
        # 持有引用
        keep_funcs.append(fred_fit)

        # ===== 自定义 X 轴标签（日期格式） =====
        xaxis = gr.GetXaxis()
        # xaxis.SetTitle("Date & Time")

        t0_ref = ams.index[0]
        midnight_indices = []
        midnight_strs = []

        for i, data_idx in enumerate(range(len(times))):
            current_time = t0_ref + pd.Timedelta(hours=float(times[data_idx]))
            if current_time.hour == 0 and current_time.minute == 0:
                midnight_indices.append(int(data_idx))
                date_label = current_time.strftime("%B %d")
                year_label = current_time.strftime("%Y")
                midnight_strs.append(f"#splitline{{{date_label}}}{{{year_label}}}")

        for data_idx, date_str in zip(midnight_indices, midnight_strs):
            xaxis.SetBinLabel(data_idx + 1, date_str)
        xaxis.LabelsOption("h")

        # 在图上添加拟合参数
        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextSize(0.035)
        latex.SetTextColor(ROOT.kBlue)
        latex.DrawLatex(0.5, 0.35, "Flux(t) = baseline - A#times exp[-#xi #times (#frac{#tau}{t-#Delta} + #frac{t-#Delta}{#tau} - 2)]")

        latex.SetTextSize(0.03)
        baseline = fred_fit.GetParameter(0)
        A = fred_fit.GetParameter(1)
        Delta = fred_fit.GetParameter(2)
        tau = fred_fit.GetParameter(3)
        xi = fred_fit.GetParameter(4)
        chi2 = fred_fit.GetChisquare()
        ndf = fred_fit.GetNDF()

        latex.DrawLatex(0.5, 0.28, f"baseline = {baseline:.3e} #pm {fred_fit.GetParError(0):.2e}")
        latex.DrawLatex(0.5, 0.24, f"#Delta = {Delta:.2f} #pm {fred_fit.GetParError(2):.2f} h")
        latex.DrawLatex(0.5, 0.20, f"#tau = {tau:.2f} #pm {fred_fit.GetParError(3):.2f} h")
        latex.DrawLatex(0.5, 0.16, f"#xi = {xi:.2f} #pm {fred_fit.GetParError(4):.2f}")
        latex.DrawLatex(0.5, 0.12, f"#chi^{{2}}/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")
        # 持有引用
        keep_texts.append(latex)

        print(f"[OK] 子图完成: {rig} 的 flux~time 拟合")
        pad.Modified()
        pad.Update()

    c_time_all.Modified()
    c_time_all.Update()
    c_time_all.SaveAs(os.path.join(OUT_DIR, "flux_time_selected_rigs.png"))
    print(f"[OK] 保存合并图: {os.path.join(OUT_DIR, 'flux_time_selected_rigs.png')}")

# ====================== 2. flux~rigidity 图 + SBPL拟合 ======================

# 标准SBPL函数: flux(R) = C(R/R0)^(gamma1) × [1 + (R/R0)^(gamma2/s)]^s
def sbpl_func(x, par):
    """
    标准SBPL能谱函数（直接拟合flux）
    flux(R) = C(R/R0)^(gamma1) × [1 + (R/R0)^(gamma2/s)]^s
    
    par[0] = C (归一化常数)
    par[1] = gamma1 (低能谱指数)
    par[2] = R0 (断点刚度)
    par[3] = gamma2 (谱指数变化)
    par[4] = s (平滑参数)
    """
    R = x[0]
    if R <= 0:
        return 0.0
    C = par[0]
    gamma1 = par[1]
    R0 = max(par[2], MIN_FLOAT)
    gamma2 = par[3]
    s = max(par[4], MIN_FLOAT)
    # flux(R) = C * (R/R0)^gamma1 * [1 + (R/R0)^(gamma2/s)]^s
    power_term = ROOT.TMath.Power(R / R0, gamma1)
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
    
    # SBPL 拟合（标准形式）
    fit_min = rig_valid.min()
    fit_max = rig_valid.max()
    sbpl_fit = TF1(f"sbpl_{idx}", sbpl_func, fit_min, fit_max, 5)
    sbpl_fit.SetParNames("C", "gamma1", "R0", "gamma2", "s")

    # 极简初值与宽松范围
    R_min = rig_valid.min()
    R_max = rig_valid.max()
    R0_init = float(np.sqrt(max(R_min, MIN_FLOAT) * max(R_max, MIN_FLOAT)))
    gamma1_init = -2.0
    gamma2_init = -1.0
    s_init = 1.0
    # 选取最接近 R0_init 的点估计 C
    idx_near = int(np.argmin(np.abs(rig_valid - R0_init)))
    R_near = float(rig_valid[idx_near])
    F_near = float(max(flux_valid[idx_near], MIN_FLOAT))
    denom = (R_near / max(R0_init, MIN_FLOAT)) ** gamma1_init * (1.0 + (R_near / max(R0_init, MIN_FLOAT)) ** (gamma2_init / s_init)) ** s_init
    C_init = float(max(F_near / max(denom, MIN_FLOAT), MIN_FLOAT))
    sbpl_fit.SetParameters(C_init, gamma1_init, R0_init, gamma2_init, s_init)
    sbpl_fit.SetParLimits(0, max(C_init * 1e-3, MIN_FLOAT), C_init * 1e3)   # C > 0
    sbpl_fit.SetParLimits(1, -10.0, 5.0)                                    # gamma1
    sbpl_fit.SetParLimits(2, max(0.5 * R_min, MIN_FLOAT), 2.0 * R_max)      # R0 > 0
    sbpl_fit.SetParLimits(3, -10.0, 10.0)                                   # gamma2
    sbpl_fit.SetParLimits(4, 0.1, 5.0)                                      # s > 0

    # 线型颜色样式保持
    sbpl_fit.SetLineColor(ROOT.kRed)
    sbpl_fit.SetLineWidth(2)
    sbpl_fit.SetLineStyle(1)
    fit_result = gr_spec.Fit(sbpl_fit, "RSMQ")
    
    chi2 = sbpl_fit.GetChisquare()
    ndf = sbpl_fit.GetNDF()
    print(f"       标准SBPL拟合: C={sbpl_fit.GetParameter(0):.3e}, "
          f"gamma1={sbpl_fit.GetParameter(1):.3f}, "
          f"R0={sbpl_fit.GetParameter(2):.3f} GV, "
          f"gamma2={sbpl_fit.GetParameter(3):.3f}, "
          f"s={sbpl_fit.GetParameter(4):.3f}")
    print(f"       Chi2/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")
    
    # 设置坐标范围与图例
    ROOT.gPad.Update()
    gr_spec.GetXaxis().SetLimits(rig_valid.min() * 0.8, rig_valid.max() * 1.2)
    # leg = TLegend(0.15, 0.3, 0.45, 0.4)
    # leg.SetTextSize(0.03)
    # leg.AddEntry(gr_spec, "Data", "lp")
    # leg.AddEntry(sbpl_fit, "SBPL fit", "l")
    # leg.Draw()
    
    # 在图上标注拟合函数表达式（分式形式，使用 R/R0）
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.035)
    latex.SetTextColor(ROOT.kBlue)
    latex.DrawLatex(0.15, 0.85-0.45, "Flux(R) = C ( #frac{R}{R_{0}} )^{#gamma_{1}} [ 1 + ( #frac{R}{R_{0}} )^{ #frac{#gamma_{2}}{s} } ]^{-s}")
    
    # 添加拟合参数值
    latex.SetTextSize(0.03)
    latex.SetTextColor(ROOT.kBlue)
    C = sbpl_fit.GetParameter(0)
    gamma1 = sbpl_fit.GetParameter(1)
    R0 = sbpl_fit.GetParameter(2)
    gamma2 = sbpl_fit.GetParameter(3)
    s = sbpl_fit.GetParameter(4)
    
    latex.DrawLatex(0.15, 0.78-0.45, f"C = {C:.3e} #pm {sbpl_fit.GetParError(0):.2e}")
    latex.DrawLatex(0.15, 0.74-0.45, f"#gamma_{{1}} = {gamma1:.3f} #pm {sbpl_fit.GetParError(1):.3f}")
    latex.DrawLatex(0.15, 0.70-0.45, f"R_{{0}} = {R0:.3f} #pm {sbpl_fit.GetParError(2):.3f} GV")
    latex.DrawLatex(0.15, 0.66-0.45, f"#gamma_{{2}} = {gamma2:.3f} #pm {sbpl_fit.GetParError(3):.3f}")
    latex.DrawLatex(0.15, 0.62-0.45, f"s = {s:.3f} #pm {sbpl_fit.GetParError(4):.3f}")
    latex.DrawLatex(0.15, 0.58-0.45, f"#chi^{{2}}/NDF = {chi2:.2f}/{ndf} = {chi2/ndf if ndf > 0 else 0:.2f}")

    c_spec.Update()
    out_name = os.path.join(OUT_DIR, f"flux_spectrum_sbpl_{timestamp.strftime('%m%d_%H%M')}.png")
    c_spec.SaveAs(out_name)
    print(f"[OK] 保存能谱图: {out_name}")

print(f"\n✅ 全部ROOT绘图和拟合完成，结果保存在 {OUT_DIR}/")