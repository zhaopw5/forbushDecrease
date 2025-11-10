# -*- coding: utf-8 -*-
"""
AMS Data Extraction and Wavelet Analysis Script
================================================
This script automatically extracts AMS data for a specified time range and performs wavelet analysis.

Features:
1. Read all AMS data from /home/zpw/Files/forbushDecrease/raw_data/
2. Extract data for user-specified time range (e.g., 2017-3-1 to 2017-12-1)
3. Create FD_{YYYYMMDD} folder in /home/zpw/Files/forbushDecrease/FDs/
4. Save extracted data as CSV file
5. Perform wavelet analysis and generate visualization plots
6. Support multiple rigidity ranges
7. Save analysis results (power values, periods, etc.)

Usage:
    python extract_and_analyze_wavelet.py --start 2017-03-01 --end 2017-12-01
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

# Add wavelet directory to path for importing waveletFunctions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAVELET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'wavelet')
sys.path.insert(0, WAVELET_DIR)

import waveletFunctions as wf


# ==================== Configuration ====================
AMS_RAW_DATA_DIR = "/home/zpw/Files/forbushDecrease/raw_data/"
OUTPUT_BASE_DIR = "/home/zpw/Files/forbushDecrease/FDs/"

# Rigidity bins (GV) - standard AMS bins
RIG_BINS = [1.00, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.40, 2.67, 2.97,
            3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.90, 6.47, 7.09, 7.76,
            8.48, 9.26, 10.1, 11.0, 13.0, 16.6, 22.8, 33.5, 48.5, 69.7, 100.0]

# Selected rigidity bins for analysis (can be modified)
SELECTED_RIGS = [1.00, 2.15, 5.37, 9.26, 16.6]

# Wavelet parameters (dt is handled separately in the analysis)
WAVELET_PARAMS = {
    'pad': 1,           # Zero padding
    'dj': 0.25,         # Scale spacing
    's0': -1,           # Smallest scale (default: 2*dt)
    'J1': -1,           # Number of scales (default: automatic)
    'mother': 'MORLET', # Wavelet type
    'param': 6.0        # Morlet wavenumber k0
}

# Default sampling interval in days
DEFAULT_DT = 1.0


# ==================== Helper Functions ====================

def scientific_notation_formatter(value, _):
    """Format colorbar labels in scientific notation."""
    if value == 1:
        return "1"
    return r"$10^{{{}}}$".format(int(np.log10(value)))


def load_ams_data(data_dir):
    """
    Load AMS data from the raw data directory.
    
    Args:
        data_dir: Path to raw data directory
        
    Returns:
        DataFrame with AMS flux data
    """
    # Look for AMS data file
    ams_file = os.path.join(data_dir, "ams", "flux_long.csv")
    
    if not os.path.exists(ams_file):
        raise FileNotFoundError(f"AMS data file not found: {ams_file}")
    
    print(f"Loading AMS data from: {ams_file}")
    df = pd.read_csv(ams_file, parse_dates=["date"])
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def extract_time_range(df, start_date, end_date):
    """
    Extract data for specified time range.
    
    Args:
        df: DataFrame with AMS data
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)
        
    Returns:
        Filtered DataFrame
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_filtered = df[mask].copy()
    
    print(f"\nExtracted data for {start_date.date()} to {end_date.date()}")
    print(f"Data points: {len(df_filtered)}")
    
    return df_filtered


def prepare_output_directory(start_date, base_dir=OUTPUT_BASE_DIR):
    """
    Create output directory with FD_{YYYYMMDD} format.
    
    Args:
        start_date: Start date for naming
        base_dir: Base directory for output
        
    Returns:
        Path to created directory
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    dir_name = f"FD_{start_date.strftime('%Y%m%d')}"
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    return output_dir


def save_extracted_data(df, output_dir):
    """
    Save extracted data to CSV file.
    
    Args:
        df: DataFrame with extracted data
        output_dir: Output directory path
        
    Returns:
        Path to saved CSV file
    """
    csv_path = os.path.join(output_dir, "ams_extracted.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved extracted data to: {csv_path}")
    return csv_path


def perform_wavelet_analysis(flux_series, dt=1.0, **wavelet_params):
    """
    Perform wavelet analysis on a flux time series.
    
    Args:
        flux_series: 1D array of flux values
        dt: Sampling interval
        **wavelet_params: Additional wavelet parameters
        
    Returns:
        Dictionary with wavelet analysis results
    """
    # Remove NaN values and normalize
    flux = np.array(flux_series, dtype=float)
    valid_mask = ~np.isnan(flux)
    flux_clean = flux[valid_mask]
    
    if len(flux_clean) == 0:
        return None
    
    # Normalize by removing mean
    flux_normalized = flux_clean - np.mean(flux_clean)
    
    # Perform wavelet transform
    wave, period, scale, coi = wf.wavelet(
        flux_normalized,
        dt,
        pad=wavelet_params.get('pad', 1),
        dj=wavelet_params.get('dj', 0.25),
        s0=wavelet_params.get('s0', -1),
        J1=wavelet_params.get('J1', -1),
        mother=wavelet_params.get('mother', 'MORLET'),
        param=wavelet_params.get('param', 6.0)
    )
    
    # Calculate power
    power = np.abs(wave) ** 2
    
    # Calculate global wavelet spectrum (time-averaged power)
    global_ws = np.mean(power, axis=1)
    
    # Calculate significance (optional - using simple red noise)
    variance = np.var(flux_normalized)
    
    try:
        signif = wf.wave_signif(
            variance,
            dt,
            scale,
            sigtest=0,
            lag1=0.0,
            siglvl=0.95,
            mother=wavelet_params.get('mother', 'MORLET'),
            param=wavelet_params.get('param', 6.0)
        )
        
        # Ensure signif has the same length as period/scale
        if len(signif) != len(period):
            # Resize signif to match
            signif = np.interp(
                np.arange(len(period)),
                np.linspace(0, len(period)-1, len(signif)),
                signif
            )
        
        # Expand significance to 2D array
        sig95 = signif[:, np.newaxis] * np.ones((1, power.shape[1]))
        sig95 = power / sig95
    except Exception as e:
        # If significance calculation fails, use dummy values
        print(f"  Warning: Significance calculation failed: {e}")
        sig95 = np.ones_like(power)
    
    # Global significance
    dof = len(flux_clean)
    try:
        global_signif = wf.wave_signif(
            variance,
            dt,
            scale,
            sigtest=1,
            lag1=0.0,
            siglvl=0.95,
            dof=dof,
            mother=wavelet_params.get('mother', 'MORLET'),
            param=wavelet_params.get('param', 6.0)
        )
        
        # Ensure global_signif has the same length as period/scale
        if len(global_signif) != len(period):
            global_signif = np.interp(
                np.arange(len(period)),
                np.linspace(0, len(period)-1, len(global_signif)),
                global_signif
            )
    except Exception as e:
        # If global significance calculation fails, use dummy values
        print(f"  Warning: Global significance calculation failed: {e}")
        global_signif = np.ones(len(period))
    
    return {
        'wave': wave,
        'period': period,
        'scale': scale,
        'coi': coi,
        'power': power,
        'global_ws': global_ws,
        'sig95': sig95,
        'global_signif': global_signif,
        'valid_mask': valid_mask
    }


def plot_wavelet_results(time_array, flux_series, wavelet_results, 
                         rigidity_label, output_dir):
    """
    Create visualization plots for wavelet analysis results.
    
    Args:
        time_array: Array of datetime values
        flux_series: Original flux time series
        wavelet_results: Dictionary from perform_wavelet_analysis
        rigidity_label: Label for rigidity bin (e.g., "1.00-1.16GV")
        output_dir: Directory to save plots
    """
    if wavelet_results is None:
        print(f"Skipping plot for {rigidity_label} (no valid data)")
        return
    
    # Extract results
    period = wavelet_results['period']
    power = wavelet_results['power']
    coi = wavelet_results['coi']
    global_ws = wavelet_results['global_ws']
    global_signif = wavelet_results['global_signif']
    sig95 = wavelet_results['sig95']
    valid_mask = wavelet_results['valid_mask']
    
    # Filter time array to valid data
    time_valid = time_array[valid_mask]
    flux_valid = flux_series[valid_mask]
    
    # Create figure with two panels
    fig = plt.figure(figsize=(16, 10))
    plt.rcParams['font.size'] = 12
    
    # Panel 1: Time series
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_valid, flux_valid, 'k-', linewidth=1.5)
    ax1.set_ylabel('Flux [m$^{-2}$s$^{-1}$sr$^{-1}$GV$^{-1}$]')
    ax1.set_title(f'AMS Proton Flux and Wavelet Power Spectrum - {rigidity_label}')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Panel 2: Wavelet power spectrum
    ax2 = plt.subplot(2, 1, 2)
    
    # Color scale for power
    vmin, vmax = 1e-1, 1e1
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
    norm = BoundaryNorm(
        boundaries=np.logspace(np.log10(vmin), np.log10(vmax), 21),
        ncolors=256,
        extend='both'
    )
    
    # Clip power values
    power_clipped = np.clip(power, vmin, vmax)
    
    # Plot power spectrum
    CS = ax2.contourf(
        time_valid, period, power_clipped,
        levels=levels, cmap='jet', norm=norm
    )
    
    # Add significance contour
    ax2.contour(time_valid, period, sig95, [-99, 1], colors='k', linewidths=1.5)
    
    # Add cone of influence
    ax2.fill_between(
        time_valid,
        coi * 0 + period[-1],
        coi,
        facecolor="none",
        edgecolor="#00000040",
        hatch='x',
        alpha=0.5
    )
    ax2.plot(time_valid, coi, 'k', alpha=0.5)
    
    ax2.set_ylabel('Period (days)')
    ax2.set_xlabel('Date')
    ax2.set_yscale('log', base=2)
    ax2.set_ylim([period.min(), period.max()])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(CS, ax=ax2, aspect=10, fraction=0.05, pad=0.02)
    cbar.set_label('Normalized Power')
    cbar.locator = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
    cbar.formatter = FuncFormatter(scientific_notation_formatter)
    cbar.update_ticks()
    
    plt.tight_layout()
    
    # Save figure
    plot_filename = f"wavelet_{rigidity_label.replace(' ', '_')}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved wavelet plot: {plot_path}")


def save_wavelet_results(time_array, wavelet_results, rigidity_label, output_dir):
    """
    Save wavelet analysis results to CSV files.
    
    Args:
        time_array: Array of datetime values
        wavelet_results: Dictionary from perform_wavelet_analysis
        rigidity_label: Label for rigidity bin
        output_dir: Directory to save results
    """
    if wavelet_results is None:
        print(f"Skipping CSV save for {rigidity_label} (no valid data)")
        return
    
    period = wavelet_results['period']
    power = wavelet_results['power']
    global_ws = wavelet_results['global_ws']
    global_signif = wavelet_results['global_signif']
    valid_mask = wavelet_results['valid_mask']
    time_valid = time_array[valid_mask]
    
    # Save power spectrum as CSV
    power_df = pd.DataFrame(
        power,
        index=period,
        columns=time_valid
    )
    power_filename = f"power_{rigidity_label.replace(' ', '_')}.csv"
    power_path = os.path.join(output_dir, power_filename)
    power_df.to_csv(power_path)
    print(f"Saved power spectrum: {power_path}")
    
    # Save global wavelet spectrum
    global_df = pd.DataFrame({
        'period_days': period,
        'global_power': global_ws,
        'significance_95': global_signif
    })
    global_filename = f"global_spectrum_{rigidity_label.replace(' ', '_')}.csv"
    global_path = os.path.join(output_dir, global_filename)
    global_df.to_csv(global_path, index=False)
    print(f"Saved global spectrum: {global_path}")


def analyze_multiple_rigidities(df, output_dir, selected_rigs=SELECTED_RIGS, rig_bins=RIG_BINS):
    """
    Perform wavelet analysis for multiple rigidity ranges.
    
    Args:
        df: DataFrame with AMS data
        output_dir: Output directory for results
        selected_rigs: List of selected rigidity bin lower bounds
        rig_bins: Full list of rigidity bins
    """
    print("\n" + "="*60)
    print("WAVELET ANALYSIS FOR MULTIPLE RIGIDITY RANGES")
    print("="*60)
    
    for rig in selected_rigs:
        if rig not in rig_bins:
            print(f"Warning: Rigidity {rig} not in bins, skipping")
            continue
        
        rig_index = rig_bins.index(rig)
        rig_min = rig_bins[rig_index]
        rig_max = rig_bins[rig_index + 1]
        rigidity_label = f"{rig_min:.2f}-{rig_max:.2f}GV"
        
        print(f"\nAnalyzing rigidity range: {rigidity_label}")
        
        # Filter data for this rigidity range
        mask = (df['rigidity_min'] == rig_min) & (df['rigidity_max'] == rig_max)
        df_rig = df[mask].copy().sort_values('date')
        
        if len(df_rig) == 0:
            print(f"  No data found for {rigidity_label}")
            continue
        
        print(f"  Data points: {len(df_rig)}")
        
        # Extract time and flux
        time_array = df_rig['date'].values
        flux_series = df_rig['flux'].values
        
        # Perform wavelet analysis
        print(f"  Performing wavelet analysis...")
        wavelet_results = perform_wavelet_analysis(
            flux_series,
            dt=DEFAULT_DT,
            **WAVELET_PARAMS
        )
        
        if wavelet_results is not None:
            # Create plots
            print(f"  Creating visualization...")
            plot_wavelet_results(
                time_array, flux_series, wavelet_results,
                rigidity_label, output_dir
            )
            
            # Save results
            print(f"  Saving results...")
            save_wavelet_results(
                time_array, wavelet_results,
                rigidity_label, output_dir
            )
        
        print(f"  ✓ Completed analysis for {rigidity_label}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract AMS data and perform wavelet analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_and_analyze_wavelet.py --start 2017-03-01 --end 2017-12-01
  python extract_and_analyze_wavelet.py --start 2015-06-19 --end 2015-07-07
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=AMS_RAW_DATA_DIR,
        help=f'Raw data directory (default: {AMS_RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f'Output base directory (default: {OUTPUT_BASE_DIR})'
    )
    
    parser.add_argument(
        '--rigidities',
        type=float,
        nargs='+',
        default=SELECTED_RIGS,
        help=f'Rigidity bins to analyze (default: {SELECTED_RIGS})'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("AMS DATA EXTRACTION AND WAVELET ANALYSIS")
    print("="*60)
    print(f"Start date: {args.start}")
    print(f"End date: {args.end}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output base directory: {args.output_dir}")
    print(f"Selected rigidities: {args.rigidities}")
    
    try:
        # Step 1: Load AMS data
        print("\n" + "="*60)
        print("STEP 1: LOADING AMS DATA")
        print("="*60)
        ams_data = load_ams_data(args.data_dir)
        
        # Step 2: Extract time range
        print("\n" + "="*60)
        print("STEP 2: EXTRACTING TIME RANGE")
        print("="*60)
        extracted_data = extract_time_range(ams_data, args.start, args.end)
        
        # Step 3: Prepare output directory
        print("\n" + "="*60)
        print("STEP 3: PREPARING OUTPUT DIRECTORY")
        print("="*60)
        output_dir = prepare_output_directory(args.start, args.output_dir)
        
        # Step 4: Save extracted data
        print("\n" + "="*60)
        print("STEP 4: SAVING EXTRACTED DATA")
        print("="*60)
        save_extracted_data(extracted_data, output_dir)
        
        # Step 5: Perform wavelet analysis
        print("\n" + "="*60)
        print("STEP 5: WAVELET ANALYSIS")
        print("="*60)
        analyze_multiple_rigidities(
            extracted_data,
            output_dir,
            selected_rigs=args.rigidities,
            rig_bins=RIG_BINS
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
