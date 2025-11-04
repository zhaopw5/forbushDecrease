# xwt_correlation.py
# Purpose: Cross-Wavelet (XWT) and optional Wavelet Coherence (WTC) between an OMNI parameter and AMS dI at a target rigidity bin.
# Data layout follows your 'correlation.py': EVENT_DIR contains 'ams_processed.csv' and 'omni_processed.csv'.
# Dependencies: numpy, pandas, matplotlib, waveletFunctions.py (from your repo).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm  # 修复：使用正确的 LogNorm 导入

# === use your wavelet core ===
import waveletFunctions as wf  # must be in the same folder or PYTHONPATH

# ---------------- User Config ----------------
EVENT_DIR = "FD_20150619"                  # same as correlation.py
AMS_FILE  = f"{EVENT_DIR}/ams_processed.csv"
OMNI_FILE = f"{EVENT_DIR}/omni_processed.csv"

TARGET_RIGIDITY = "5.37-5.90GV"            # choose one bin to start; you can loop later
OMNI_PARAM      = "B_vec_mag"               # e.g. Vsw, Bz_gse, Ey_mVpm, etc.

# Sampling interval in HOURS (your FD files are hourly)
DT_H = 1.0

# Wavelet settings
MOTHER  = 'MORLET'
K0      = 6.0         # Morlet wavenumber
PAD     = 1           # zero padding for speed
DJ      = 0.25        # scale spacing (powers of two)
S0      = -1          # default 2*dt inside wf.wavelet
J1      = -1          # default inside wf.wavelet

# Coherence smoothing kernels (simple, robust defaults)
TIME_SMOOTH_STEPS  = 6     # moving average half-window in time (± steps)
SCALE_SMOOTH_STEPS = 1     # moving average half-window in scale (± scales)

# ------------------------------------------------

def zscore(x):
    """Standardize a series robustly (ignore NaNs)."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return x - m
    return (x - m) / s

def moving_average_2d(A, t_win=6, s_win=1):
    """
    Simple separable smoothing on (scale x time) array.
    t_win, s_win are half-window sizes (# of samples).
    """
    if A.ndim != 2:
        raise ValueError("A must be 2D (scale x time)")
    S, T = A.shape
    out = np.copy(A)

    # smooth along time
    if t_win > 0:
        k_t = np.ones(2*t_win+1) / (2*t_win+1)
        out = np.apply_along_axis(lambda v: np.convolve(v, k_t, mode='same'), axis=1, arr=out)

    # smooth along scale
    if s_win > 0:
        k_s = np.ones(2*s_win+1) / (2*s_win+1)
        out = np.apply_along_axis(lambda v: np.convolve(v, k_s, mode='same'), axis=0, arr=out)

    return out

def cross_wavelet(x, y, dt, pad=1, dj=0.25, s0=-1, j1=-1, mother='MORLET', param=6.0):
    """
    Compute Cross-Wavelet Transform (XWT) between x and y.
    Returns:
      time_index, period, Wxy (complex), Xpower, Ypower, XWT_power, phase, coi
    Shapes:
      wave arrays: (n_scales, n_times)
    """
    # NaN handling: common mask
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # CWT of both
    Wx, period, scale, coi_x = wf.wavelet(x, dt, pad=pad, dj=dj, s0=s0, J1=j1, mother=mother, param=param)
    Wy, period, scale, coi_y = wf.wavelet(y, dt, pad=pad, dj=dj, s0=s0, J1=j1, mother=mother, param=param)

    # truncate to original length (wavelet() already trims padding)
    # Shapes: (n_scales, n_times)
    # Cross wavelet
    Wxy = Wx * np.conjugate(Wy)

    # Power & phase
    Xpower = (np.abs(Wx))**2
    Ypower = (np.abs(Wy))**2
    XWT_power = np.abs(Wxy)
    phase = np.angle(Wxy)   # radians; positive: x leads y (x -> y)

    # Cone of Influence: smallest (conservative)
    coi = np.minimum(coi_x, coi_y)

    # time index relative to mask (uniform grid)
    t = np.arange(x.shape[0]) * dt
    return t, period, Wxy, Xpower, Ypower, XWT_power, phase, coi

def wavelet_coherence(Wx, Wy, time_smooth=6, scale_smooth=1):
    """
    Compute Wavelet Coherence R^2 following Torrence & Webster (1999) idea
    with simple separable boxcar smoothing (robust approximation).
      R2 = | S(Wx * Wy*) |^2 / ( S(|Wx|^2) * S(|Wy|^2) )
    Inputs are the complex wavelet fields of x and y (scale x time).
    Returns R2 (scale x time), with values in [0, 1].
    """
    Wxy = Wx * np.conjugate(Wy)
    S_Wxy = moving_average_2d(Wxy, t_win=time_smooth, s_win=scale_smooth)
    S_X = moving_average_2d(np.abs(Wx)**2, t_win=time_smooth, s_win=scale_smooth)
    S_Y = moving_average_2d(np.abs(Wy)**2, t_win=time_smooth, s_win=scale_smooth)

    num = np.abs(S_Wxy)**2
    den = S_X * S_Y
    with np.errstate(divide='ignore', invalid='ignore'):
        R2 = np.where(den > 0, num / den, np.nan)
    R2 = np.clip(R2, 0.0, 1.0)
    return R2

def plot_xwt(time_num, time_dt_index, period, XWT_power, phase, coi, title, out_png, vmin=None, vmax=None):
    """
    Plot Cross-Wavelet Power with phase arrows and COI.
    time_num: numeric time in hours relative (length N)
    time_dt_index: pandas.DatetimeIndex aligned to time_num for labels
    period: array of periods (len S)
    """
    T = time_num
    P = period

    # power scaling（健壮：仅基于有限且>0的数据，提供兜底）
    power = np.asarray(XWT_power, dtype=float).copy()
    finite_pos = np.isfinite(power) & (power > 0)

    if vmin is None or not np.isfinite(vmin):
        vmin = (np.nanpercentile(power[finite_pos], 5) if np.any(finite_pos) else 1e-6)
    if vmax is None or not np.isfinite(vmax):
        vmax = (np.nanpercentile(power[finite_pos], 95) if np.any(finite_pos) else 1e-5)

    # 兜底与修正
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin <= 0 or vmax <= vmin:
        vmin, vmax = 1e-6, 1e-5

    # 非正值对 LogNorm 不合法，置为 NaN
    power[~finite_pos] = np.nan

    fig, ax = plt.subplots(figsize=(12, 6))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 40)
    cs = ax.contourf(T, P, power, levels=levels, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='jet')

    # COI
    ax.plot(T, coi, 'k', alpha=0.7)
    ax.fill_between(T, coi, P.max(), color='k', alpha=0.15, hatch='//', lw=0)

    # Phase arrows (downsample for clarity)
    skip_t = max(1, len(T)//60)         # ~60 arrows over time
    skip_s = max(1, len(P)//30)         # ~30 over scales
    TT, PP = np.meshgrid(T, P)
    ang = phase
    U = np.cos(ang)
    V = np.sin(ang)
    ax.quiver(TT[::skip_s, ::skip_t], PP[::skip_s, ::skip_t],
              U[::skip_s, ::skip_t], V[::skip_s, ::skip_t],
              pivot='mid', headwidth=2, headlength=3, headaxislength=3, alpha=0.6, color='k', scale=20)

    ax.set_yscale('log', base=2)
    ax.set_ylabel('Period (hours)')
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.set_ylim(2, 32)  # 设置 y 轴范围为 2~32 小时

    # custom x ticks with datetime labels
    xticks = np.linspace(T.min(), T.max(), 6)
    ax.set_xticks(xticks)
    t0 = time_dt_index[0]
    labels = []
    for x in xticks:
        idx = int(round(x/DT_H))
        idx = np.clip(idx, 0, len(time_dt_index)-1)
        labels.append(time_dt_index[idx].strftime('%Y-%m-%d\n%H:%M'))
    ax.set_xticklabels(labels)

    cbar = plt.colorbar(cs, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Cross-Wavelet Power |Wxy|')

    # nice minor ticks
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=2, subs=(1.5,)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_coherence(time_num, time_dt_index, period, R2, coi, title, out_png):
    fig, ax = plt.subplots(figsize=(12, 5))
    cs = ax.contourf(time_num, period, R2, levels=np.linspace(0, 1, 21), cmap='viridis')
    ax.plot(time_num, coi, 'k', alpha=0.7)
    ax.fill_between(time_num, coi, period.max(), color='k', alpha=0.15, hatch='//', lw=0)

    ax.set_yscale('log', base=2)
    ax.set_ylabel('Period (hours)')
    ax.set_xlabel('Time')
    ax.set_title(title + ' (Wavelet Coherence $R^2$)')
    ax.set_ylim(2, 32)  # 设置 y 轴范围为 2~32 小时

    xticks = np.linspace(time_num.min(), time_num.max(), 6)
    ax.set_xticks(xticks)
    labels = []
    for x in xticks:
        idx = int(round(x/DT_H))
        idx = np.clip(idx, 0, len(time_dt_index)-1)
        labels.append(time_dt_index[idx].strftime('%Y-%m-%d\n%H:%M'))
    ax.set_xticklabels(labels)

    cbar = plt.colorbar(cs, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Coherence $R^2$')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_global_spectra(period, XWT_power, R2, title, out_png):
    """
    Global cross-wavelet power (averaged over time) and coherence (median over time).
    """
    # 按尺度检测是否存在有限值，避免全 NaN 触发 RuntimeWarning
    valid_power = np.any(np.isfinite(XWT_power), axis=1)
    power_tavg = np.full(period.shape, np.nan)
    power_tavg[valid_power] = np.nanmean(XWT_power[valid_power, :], axis=1)

    R2_tmed = None
    if R2 is not None:
        valid_R2 = np.any(np.isfinite(R2), axis=1)
        R2_tmed = np.full(period.shape, np.nan)
        R2_tmed[valid_R2] = np.nanmedian(R2[valid_R2, :], axis=1)

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(power_tavg, period, lw=2)

    # 仅当存在正值时使用对数 x 轴，否则退回线性，避免报错
    pos = np.isfinite(power_tavg) & (power_tavg > 0)
    if np.any(pos):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')

    ax1.set_yscale('log', base=2)
    ax1.set_xlabel('Global Cross-Power (time-avg)')
    ax1.set_ylabel('Period (hours)')
    ax1.set_title(title)
    ax1.set_ylim(2, 32)  # 设置 y 轴范围为 2~32 小时

    if R2_tmed is not None:
        ax2 = ax1.twiny()
        ax2.plot(R2_tmed, period, lw=1.8, color='tab:green')
        ax2.set_xlim(0,1)
        ax2.set_xlabel('Median Coherence $R^2$')

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    os.makedirs(EVENT_DIR, exist_ok=True)

    # ---- load and merge ----
    ams = pd.read_csv(AMS_FILE, parse_dates=['datetime'])
    omni = pd.read_csv(OMNI_FILE, parse_dates=['datetime'])
    df = pd.merge(ams, omni, on='datetime', how='inner').sort_values('datetime')

    # ---- pick columns ----
    ams_col = f"dI_{TARGET_RIGIDITY}"   # follows your correlation.py naming
    if ams_col not in df.columns:
        raise RuntimeError(f"AMS column not found: {ams_col}")
    if OMNI_PARAM not in df.columns:
        raise RuntimeError(f"OMNI param not found: {OMNI_PARAM}")

    # ---- 时间插值去 NaN 后再标准化 ----
    t_index = df['datetime'].reset_index(drop=True)
    y_raw = pd.Series(df[ams_col].astype(float).values, index=t_index)
    x_raw = pd.Series(df[OMNI_PARAM].astype(float).values, index=t_index)
    y = zscore(y_raw.interpolate(method='time').bfill().ffill().values)
    x = zscore(x_raw.interpolate(method='time').bfill().ffill().values)

    # ---- compute individual wavelets (also needed for coherence) ----
    Wx, period, scale, coi_x = wf.wavelet(x, DT_H, pad=PAD, dj=DJ, s0=S0, J1=J1, mother=MOTHER, param=K0)
    Wy, period, scale, coi_y = wf.wavelet(y, DT_H, pad=PAD, dj=DJ, s0=S0, J1=J1, mother=MOTHER, param=K0)

    # ---- cross wavelet from the two fields ----
    tnum = np.arange(Wx.shape[1]) * DT_H
    Wxy = Wx * np.conjugate(Wy)
    XWT_power = np.abs(Wxy)
    phase = np.angle(Wxy)
    coi = np.minimum(coi_x, coi_y)

    # ---- coherence (optional but useful) ----
    R2 = wavelet_coherence(Wx, Wy, time_smooth=TIME_SMOOTH_STEPS, scale_smooth=SCALE_SMOOTH_STEPS)

    # ---- plots ----
    # t_index 已在上方定义
    # 1) Cross-Wavelet Power with phase
    ttl = f"XWT: {OMNI_PARAM} vs AMS dI({TARGET_RIGIDITY})"
    out = os.path.join(EVENT_DIR, f"xwt_{OMNI_PARAM}_vs_{TARGET_RIGIDITY.replace(' ','')}.png")
    plot_xwt(tnum, t_index, period, XWT_power, phase, coi, ttl, out)

    # 2) Wavelet Coherence
    ttl2 = f"WTC: {OMNI_PARAM} vs AMS dI({TARGET_RIGIDITY})"
    out2 = os.path.join(EVENT_DIR, f"wtc_{OMNI_PARAM}_vs_{TARGET_RIGIDITY.replace(' ','')}.png")
    plot_coherence(tnum, t_index, period, R2, coi, ttl2, out2)

    # 3) Global spectra
    out3 = os.path.join(EVENT_DIR, f"global_{OMNI_PARAM}_vs_{TARGET_RIGIDITY.replace(' ','')}.png")
    plot_global_spectra(period, XWT_power, R2, f"Global spectra\n{OMNI_PARAM} vs {TARGET_RIGIDITY}", out3)

    # 4) Save CSV of global curves
    gcsv = os.path.join(EVENT_DIR, f"global_{OMNI_PARAM}_vs_{TARGET_RIGIDITY.replace(' ','')}.csv")
    pd.DataFrame({
        'period_hours': period,
        'global_cross_power': np.nanmean(XWT_power, axis=1),
        'median_R2': np.nanmedian(R2, axis=1)
    }).to_csv(gcsv, index=False)

    print(f"[OK] Saved:\n  {out}\n  {out2}\n  {out3}\n  {gcsv}")

if __name__ == "__main__":
    main()
