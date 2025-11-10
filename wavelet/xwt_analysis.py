#!/usr/bin/env python3
"""
Cross Wavelet Transform (XWT) Analysis Program
==============================================
Compares hourly AMS flux data with OMNI solar wind parameters using Cross Wavelet Transform.

This program:
1. Reads ams_processed.csv and omni_processed.csv files
2. Extracts rigidity channels from AMS data (I_X-YGV columns)
3. Performs Cross Wavelet Transform between each rigidity channel and each OMNI parameter
4. Generates XWT plots showing:
   - XWT power spectrum with contour levels
   - 95% significance level contours
   - Cone of Influence (COI) shading
   - Phase arrows indicating lead/lag relationships
   - Colorbar with logarithmic scale
5. Saves results as PDF and PNG in organized directory structure
6. Includes time-averaged global XWT spectrum plots
7. Outputs XWT power statistics to CSV files

Usage:
    python xwt_analysis.py --event-dir FD_20150619
    python xwt_analysis.py --event-dir FD_20170301 --rigidities "5.37-5.90GV,9.26-10.10GV"

Author: Generated for forbushDecrease project
Date: 2025-11-10
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import FuncFormatter

# Import wavelet functions from the repository
import waveletFunctions as wf


# ============== Configuration ==============
# Default OMNI parameters to analyze
DEFAULT_OMNI_PARAMS = ['B_avg_abs', 'Bz_gse', 'Np', 'Vsw']

# Default rigidity bins (same as wavelet_v251110.py)
RIG_BINS = [1.0, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.4, 2.67, 2.97,
            3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 6.47, 7.09, 7.76,
            8.48, 9.26, 10.1, 11.0, 13.0, 16.6, 22.8, 33.5, 48.5, 69.7, 100.0]

# Wavelet parameters
DT_H = 1.0              # Sampling interval in hours
MOTHER = 'MORLET'       # Wavelet type
K0 = 6.0                # Morlet wavenumber
PAD = 1                 # Zero padding for speed
DJ = 0.25               # Scale spacing
S0 = -1                 # Smallest scale (default: 2*dt)
J1 = -1                 # Number of scales (default: auto)

# Plotting parameters
VMIN = 1e-2             # Minimum for colorbar
VMAX = 1e2              # Maximum for colorbar
N_CONTOURS = 40         # Number of contour levels

# Configure matplotlib style (matching wavelet_v251110.py)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# ============== Setup Logging ==============
def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamp and level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============== Helper Functions ==============
def scientific_notation_formatter(value, _):
    """Format colorbar ticks in scientific notation."""
    if value == 1:
        return "1"
    exponent = int(np.log10(value))
    return r"$10^{{{}}}$".format(exponent)


def zscore(x):
    """Standardize a series robustly (ignore NaNs)."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return x - m
    return (x - m) / s


def detrend_normalize(data):
    """Remove mean and normalize by standard deviation."""
    data = np.asarray(data, dtype=float)
    # Remove NaNs for processing
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 2:
        return data
    
    # Remove mean and normalize
    data_clean = data[valid_mask]
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=1)
    
    if std_val > 0:
        normalized = (data - mean_val) / std_val
    else:
        normalized = data - mean_val
    
    return normalized


# ============== XWT Core Functions ==============
def cross_wavelet_transform(x, y, dt, pad=1, dj=0.25, s0=-1, j1=-1, 
                           mother='MORLET', param=6.0):
    """
    Compute Cross-Wavelet Transform (XWT) between x and y.
    
    Parameters
    ----------
    x, y : array_like
        Input time series
    dt : float
        Sampling interval
    pad : int
        Pad the time series with zeros (1=yes, 0=no)
    dj : float
        Spacing between discrete scales
    s0 : float
        Smallest scale (default: 2*dt)
    j1 : int
        Number of scales minus one (default: auto)
    mother : str
        Wavelet type ('MORLET', 'PAUL', 'DOG')
    param : float
        Mother wavelet parameter
    
    Returns
    -------
    t : array
        Time index
    period : array
        Period array
    Wxy : array (complex)
        Cross wavelet transform
    Xpower : array
        Wavelet power of x
    Ypower : array
        Wavelet power of y
    XWT_power : array
        Cross wavelet power |Wxy|
    phase : array
        Phase angle (radians)
    coi : array
        Cone of influence
    """
    # Handle NaNs: common mask
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        raise ValueError("Too few valid data points after removing NaNs")
    
    # Compute CWT of both series
    logger.debug(f"Computing wavelet transform for series with {len(x_clean)} points")
    Wx, period, scale, coi_x = wf.wavelet(x_clean, dt, pad=pad, dj=dj, 
                                          s0=s0, J1=j1, mother=mother, param=param)
    Wy, period, scale, coi_y = wf.wavelet(y_clean, dt, pad=pad, dj=dj, 
                                          s0=s0, J1=j1, mother=mother, param=param)
    
    # Cross wavelet transform
    Wxy = Wx * np.conjugate(Wy)
    
    # Power and phase
    Xpower = np.abs(Wx) ** 2
    Ypower = np.abs(Wy) ** 2
    XWT_power = np.abs(Wxy)
    phase = np.angle(Wxy)  # radians; positive: x leads y
    
    # Cone of Influence: most conservative (minimum of both)
    coi = np.minimum(coi_x, coi_y)
    
    # Time index
    t = np.arange(len(x_clean)) * dt
    
    return t, period, Wxy, Xpower, Ypower, XWT_power, phase, coi


def compute_significance(data1, data2, period, scale, dt, mother='MORLET', param=6.0):
    """
    Compute 95% significance level for cross wavelet transform.
    
    Parameters
    ----------
    data1, data2 : array_like
        Input time series
    period : array
        Period array from wavelet transform
    scale : array
        Scale array from wavelet transform
    dt : float
        Sampling interval
    mother : str
        Wavelet type
    param : float
        Mother wavelet parameter
    
    Returns
    -------
    sig95 : array
        95% significance level
    """
    # Calculate lag-1 autocorrelation for both series
    def calc_lag1(data):
        data = np.asarray(data, dtype=float)
        valid = ~np.isnan(data)
        d = data[valid]
        if len(d) < 2:
            return 0.0
        mean_d = np.mean(d)
        numerator = np.sum((d[:-1] - mean_d) * (d[1:] - mean_d))
        denominator = np.sum((d - mean_d) ** 2)
        if denominator > 0:
            return numerator / denominator
        return 0.0
    
    lag1_1 = calc_lag1(data1)
    lag1_2 = calc_lag1(data2)
    # Use geometric mean for cross spectrum
    lag1 = np.sqrt(np.abs(lag1_1 * lag1_2))
    
    # Variance
    var1 = np.nanvar(data1, ddof=1)
    var2 = np.nanvar(data2, ddof=1)
    variance = np.sqrt(var1 * var2)
    
    # Compute significance
    signif = wf.wave_signif(variance, dt=dt, scale=scale, sigtest=0,
                            lag1=lag1, mother=mother, param=param)
    
    return signif


# ============== Plotting Functions ==============
def plot_xwt(time_num, time_index, period, XWT_power, phase, coi, sig95,
             title, output_path, vmin=None, vmax=None, format='both'):
    """
    Plot Cross-Wavelet Power spectrum with phase arrows and COI.
    
    Parameters
    ----------
    time_num : array
        Numeric time in hours
    time_index : DatetimeIndex
        Datetime index for x-axis labels
    period : array
        Period array
    XWT_power : array
        Cross wavelet power
    phase : array
        Phase angle
    coi : array
        Cone of influence
    sig95 : array
        95% significance level (2D array, scale x time)
    title : str
        Plot title
    output_path : str
        Output file path (without extension)
    vmin, vmax : float
        Colorbar limits
    format : str
        Output format ('pdf', 'png', or 'both')
    """
    # Handle power scaling
    power = np.asarray(XWT_power, dtype=float).copy()
    finite_pos = np.isfinite(power) & (power > 0)
    
    if vmin is None or not np.isfinite(vmin):
        vmin = np.nanpercentile(power[finite_pos], 5) if np.any(finite_pos) else VMIN
    if vmax is None or not np.isfinite(vmax):
        vmax = np.nanpercentile(power[finite_pos], 95) if np.any(finite_pos) else VMAX
    
    # Ensure valid range
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin <= 0 or vmax <= vmin:
        vmin, vmax = VMIN, VMAX
    
    # Set non-positive values to NaN for LogNorm
    power[~finite_pos] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Contour levels
    levels = np.logspace(np.log10(vmin), np.log10(vmax), N_CONTOURS)
    
    # Plot XWT power
    cs = ax.contourf(time_num, period, power, levels=levels,
                     norm=LogNorm(vmin=vmin, vmax=vmax), cmap='jet', extend='both')
    
    # Add 95% significance contour
    if sig95 is not None:
        ax.contour(time_num, period, sig95, levels=[-99, 1], colors='k', 
                  linewidths=2, linestyles='solid')
    
    # Add COI
    ax.plot(time_num, coi, 'k', linewidth=2, alpha=0.7)
    ax.fill_between(time_num, coi, period.max(), facecolor='none',
                    edgecolor='#00000040', hatch='x', alpha=0.5)
    
    # Add phase arrows (downsample for clarity)
    skip_t = max(1, len(time_num) // 60)  # ~60 arrows over time
    skip_s = max(1, len(period) // 30)     # ~30 over scales
    TT, PP = np.meshgrid(time_num, period)
    U = np.cos(phase)
    V = np.sin(phase)
    ax.quiver(TT[::skip_s, ::skip_t], PP[::skip_s, ::skip_t],
              U[::skip_s, ::skip_t], V[::skip_s, ::skip_t],
              pivot='mid', headwidth=2, headlength=3, headaxislength=3,
              alpha=0.6, color='white', scale=25, width=0.003)
    
    # Format axes
    ax.set_yscale('log', base=2)
    ax.set_ylabel('Period (hours)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set y-axis limits and format
    y_min = max(2, period.min())
    y_max = min(256, period.max())
    ax.set_ylim([y_max, y_min])  # Inverted
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Format x-axis with dates
    if len(time_index) > 0:
        # Create evenly spaced ticks
        n_ticks = 6
        tick_indices = np.linspace(0, len(time_num) - 1, n_ticks, dtype=int)
        ax.set_xticks(time_num[tick_indices])
        
        # Map numeric time back to datetime
        tick_labels = []
        for idx in tick_indices:
            if idx < len(time_index):
                tick_labels.append(time_index[idx].strftime('%Y-%m-%d\n%H:%M'))
            else:
                tick_labels.append('')
        ax.set_xticklabels(tick_labels)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Cross-Wavelet Power |Wxy|', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Format colorbar ticks
    cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=8)
    cbar.formatter = FuncFormatter(scientific_notation_formatter)
    cbar.update_ticks()
    
    # Set tick parameters
    ax.tick_params(axis='both', direction='in', which='major', width=1.5, length=5,
                  top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='minor', width=1, length=3,
                  top=True, bottom=True, left=True, right=True)
    
    # Make spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save figure
    if format in ['pdf', 'both']:
        fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}.pdf")
    if format in ['png', 'both']:
        fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}.png")
    
    plt.close(fig)


def plot_global_spectrum(period, XWT_power, title, output_path, format='both'):
    """
    Plot time-averaged global XWT spectrum.
    
    Parameters
    ----------
    period : array
        Period array
    XWT_power : array
        Cross wavelet power (scale x time)
    title : str
        Plot title
    output_path : str
        Output file path (without extension)
    format : str
        Output format ('pdf', 'png', or 'both')
    """
    # Compute time average
    valid_power = np.isfinite(XWT_power)
    global_power = np.full(period.shape, np.nan)
    
    for i in range(len(period)):
        valid_at_scale = valid_power[i, :]
        if np.any(valid_at_scale):
            global_power[i] = np.nanmean(XWT_power[i, valid_at_scale])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot global spectrum
    valid_mask = np.isfinite(global_power) & (global_power > 0)
    ax.plot(global_power[valid_mask], period[valid_mask], 'b-', linewidth=2)
    ax.fill_betweenx(period[valid_mask], 0, global_power[valid_mask], 
                     alpha=0.3, color='blue')
    
    # Format axes
    if np.any(valid_mask):
        ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Time-Averaged Cross-Wavelet Power', fontsize=14)
    ax.set_ylabel('Period (hours)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set y-axis limits
    y_min = max(2, period.min())
    y_max = min(256, period.max())
    ax.set_ylim([y_min, y_max])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set tick parameters
    ax.tick_params(axis='both', direction='in', which='major', width=1.5, length=5)
    ax.tick_params(axis='both', direction='in', which='minor', width=1, length=3)
    
    # Make spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save figure
    if format in ['pdf', 'both']:
        fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}.pdf")
    if format in ['png', 'both']:
        fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}.png")
    
    plt.close(fig)


# ============== Data Loading ==============
def load_and_prepare_data(event_dir):
    """
    Load and prepare AMS and OMNI data from CSV files.
    
    Parameters
    ----------
    event_dir : str
        Path to event directory containing ams_processed.csv and omni_processed.csv
    
    Returns
    -------
    df : DataFrame
        Merged dataframe with common time indices
    rig_columns : list
        List of rigidity column names (I_X-YGV format)
    omni_params : list
        List of available OMNI parameter names
    """
    ams_file = os.path.join(event_dir, 'ams_processed.csv')
    omni_file = os.path.join(event_dir, 'omni_processed.csv')
    
    # Check files exist
    if not os.path.exists(ams_file):
        raise FileNotFoundError(f"AMS file not found: {ams_file}")
    if not os.path.exists(omni_file):
        raise FileNotFoundError(f"OMNI file not found: {omni_file}")
    
    logger.info(f"Loading AMS data from: {ams_file}")
    ams_df = pd.read_csv(ams_file, parse_dates=['datetime'])
    
    logger.info(f"Loading OMNI data from: {omni_file}")
    omni_df = pd.read_csv(omni_file, parse_dates=['datetime'])
    
    # Merge on datetime
    logger.info("Merging AMS and OMNI data on common time indices")
    df = pd.merge(ams_df, omni_df, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"Merged data contains {len(df)} time points")
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Find rigidity columns (format: I_X-YGV or dI_X-YGV)
    rig_columns = [col for col in df.columns if col.startswith('I_') or col.startswith('dI_')]
    logger.info(f"Found {len(rig_columns)} rigidity channels: {rig_columns[:5]}...")
    
    # Find OMNI parameter columns
    omni_params = [col for col in df.columns if col in DEFAULT_OMNI_PARAMS or 
                  col.endswith('_abs') or col.endswith('_gse') or 
                  col in ['Vsw', 'Np', 'Pdyn', 'Bz', 'By', 'Bx']]
    logger.info(f"Found {len(omni_params)} OMNI parameters: {omni_params}")
    
    return df, rig_columns, omni_params


# ============== Main XWT Analysis ==============
def perform_xwt_analysis(event_dir, rigidities=None, omni_params=None, 
                        output_format='both'):
    """
    Perform Cross Wavelet Transform analysis between AMS and OMNI data.
    
    Parameters
    ----------
    event_dir : str
        Path to event directory
    rigidities : list, optional
        List of specific rigidity channels to analyze (None = all)
    omni_params : list, optional
        List of OMNI parameters to analyze (None = defaults)
    output_format : str
        Output format: 'pdf', 'png', or 'both'
    """
    logger.info("="*70)
    logger.info("Starting Cross Wavelet Transform (XWT) Analysis")
    logger.info("="*70)
    logger.info(f"Event directory: {event_dir}")
    
    # Load data
    df, rig_columns, available_omni = load_and_prepare_data(event_dir)
    
    # Select rigidities to analyze
    if rigidities is not None:
        rig_columns = [col for col in rig_columns if any(rig in col for rig in rigidities)]
        logger.info(f"Analyzing {len(rig_columns)} selected rigidity channels")
    
    # Select OMNI parameters to analyze
    if omni_params is None:
        omni_params = [p for p in DEFAULT_OMNI_PARAMS if p in available_omni]
    else:
        omni_params = [p for p in omni_params if p in available_omni]
    
    if not omni_params:
        logger.error("No valid OMNI parameters found!")
        return
    
    logger.info(f"Analyzing {len(omni_params)} OMNI parameters: {omni_params}")
    
    # Create output directories
    plots_dir = os.path.join(event_dir, 'plots_xwt')
    data_dir = os.path.join(event_dir, 'data_xwt')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Output directories: {plots_dir}, {data_dir}")
    
    # Perform XWT for each combination
    total_analyses = len(rig_columns) * len(omni_params)
    current = 0
    
    for rig_col in rig_columns:
        for omni_param in omni_params:
            current += 1
            logger.info(f"\n[{current}/{total_analyses}] Processing: {rig_col} vs {omni_param}")
            
            try:
                # Extract and prepare data
                ams_data = df[rig_col].values
                omni_data = df[omni_param].values
                time_index = df['datetime']
                
                # Check for sufficient data
                valid_mask = ~(np.isnan(ams_data) | np.isnan(omni_data))
                n_valid = np.sum(valid_mask)
                
                if n_valid < 20:
                    logger.warning(f"Insufficient valid data points ({n_valid}), skipping")
                    continue
                
                # Preprocess: detrend and normalize
                ams_processed = detrend_normalize(ams_data)
                omni_processed = detrend_normalize(omni_data)
                
                # Perform Cross Wavelet Transform
                logger.debug("Computing Cross Wavelet Transform...")
                t, period, Wxy, Xpower, Ypower, XWT_power, phase, coi = \
                    cross_wavelet_transform(ams_processed, omni_processed, DT_H,
                                          pad=PAD, dj=DJ, s0=S0, j1=J1,
                                          mother=MOTHER, param=K0)
                
                # Compute significance levels
                logger.debug("Computing significance levels...")
                # Need to get scale array for significance calculation
                _, _, scale, _ = wf.wavelet(ams_processed[valid_mask], DT_H, 
                                           pad=PAD, dj=DJ, s0=S0, J1=J1,
                                           mother=MOTHER, param=K0)
                
                signif = compute_significance(ams_data, omni_data, period, scale,
                                             DT_H, mother=MOTHER, param=K0)
                
                # Expand significance to 2D array
                sig95 = signif[:, np.newaxis] @ np.ones((1, XWT_power.shape[1]))
                sig95 = XWT_power / sig95  # Ratio > 1 means significant
                
                # Create time index for plotting (aligned with valid data)
                time_idx_valid = time_index[valid_mask].reset_index(drop=True)
                
                # Generate plots
                rig_clean = rig_col.replace('_', '').replace('dI', '').replace('I', '')
                
                # Main XWT plot
                title = f"XWT: {omni_param} vs AMS {rig_clean}"
                output_base = os.path.join(plots_dir, 
                                          f"xwt_{omni_param}_vs_{rig_clean}")
                plot_xwt(t, time_idx_valid, period, XWT_power, phase, coi, sig95,
                        title, output_base, format=output_format)
                
                # Global spectrum plot
                title_global = f"Global Spectrum: {omni_param} vs AMS {rig_clean}"
                output_global = os.path.join(plots_dir,
                                            f"global_xwt_{omni_param}_vs_{rig_clean}")
                plot_global_spectrum(period, XWT_power, title_global, output_global,
                                   format=output_format)
                
                # Save XWT power statistics to CSV
                global_power = np.nanmean(XWT_power, axis=1)
                stats_df = pd.DataFrame({
                    'period_hours': period,
                    'global_xwt_power': global_power,
                    'min_xwt_power': np.nanmin(XWT_power, axis=1),
                    'max_xwt_power': np.nanmax(XWT_power, axis=1),
                    'median_xwt_power': np.nanmedian(XWT_power, axis=1)
                })
                
                csv_file = os.path.join(data_dir, 
                                       f"xwt_stats_{omni_param}_vs_{rig_clean}.csv")
                stats_df.to_csv(csv_file, index=False)
                logger.info(f"Saved statistics: {csv_file}")
                
                # Save full XWT power spectrum as CSV
                time_labels = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in time_idx_valid]
                xwt_full_df = pd.DataFrame(XWT_power, 
                                          index=period,
                                          columns=time_labels)
                csv_full = os.path.join(data_dir,
                                       f"xwt_power_{omni_param}_vs_{rig_clean}.csv")
                xwt_full_df.to_csv(csv_full)
                logger.info(f"Saved full XWT power: {csv_full}")
                
            except Exception as e:
                logger.error(f"Error processing {rig_col} vs {omni_param}: {str(e)}", 
                           exc_info=True)
                continue
    
    logger.info("\n" + "="*70)
    logger.info("XWT Analysis Complete!")
    logger.info("="*70)
    logger.info(f"Results saved to:")
    logger.info(f"  Plots: {plots_dir}")
    logger.info(f"  Data:  {data_dir}")


# ============== Command Line Interface ==============
def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description='Cross Wavelet Transform Analysis for AMS and OMNI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --event-dir FD_20150619
  %(prog)s --event-dir FD_20170301 --rigidities "5.37-5.90GV,9.26-10.10GV"
  %(prog)s --event-dir FD_20150619 --omni-params "B_avg_abs,Vsw" --format pdf
        """
    )
    
    parser.add_argument('--event-dir', type=str, required=True,
                       help='Path to event directory containing processed CSV files')
    parser.add_argument('--rigidities', type=str, default=None,
                       help='Comma-separated list of rigidity channels to analyze (default: all)')
    parser.add_argument('--omni-params', type=str, default=None,
                       help='Comma-separated list of OMNI parameters (default: B_avg_abs,Bz_gse,Np,Vsw)')
    parser.add_argument('--format', type=str, default='both', choices=['pdf', 'png', 'both'],
                       help='Output format for plots (default: both)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    global logger
    logger = setup_logging(getattr(logging, args.log_level))
    
    # Parse rigidities and OMNI parameters
    rigidities = None if args.rigidities is None else args.rigidities.split(',')
    omni_params = None if args.omni_params is None else args.omni_params.split(',')
    
    # Run analysis
    try:
        perform_xwt_analysis(args.event_dir, rigidities, omni_params, args.format)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
