#!/usr/bin/env python3
"""
Generate synthetic test data for XWT analysis testing.
Creates ams_processed.csv and omni_processed.csv with realistic structure.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_test_data(output_dir, n_hours=720, n_rigidities=5):
    """
    Generate synthetic test data for XWT analysis.
    
    Parameters
    ----------
    output_dir : str
        Directory to save test data files
    n_hours : int
        Number of hourly data points (default: 720 = 30 days)
    n_rigidities : int
        Number of rigidity channels to generate (default: 5)
    """
    print(f"Generating test data with {n_hours} hours and {n_rigidities} rigidity channels")
    
    # Create time series
    start_date = datetime(2015, 6, 19, 0, 0, 0)
    datetimes = [start_date + timedelta(hours=i) for i in range(n_hours)]
    
    # Define rigidity bins (GV)
    rig_bins = [1.0, 2.15, 5.9, 9.26, 16.6, 33.5][:n_rigidities+1]
    
    # Generate AMS data
    print("Generating AMS flux data...")
    ams_data = {'datetime': datetimes}
    
    # Create synthetic Forbush decrease
    t = np.arange(n_hours)
    # Forbush decrease starting at hour 100
    t0 = 100
    t1 = 200  # Recovery starts
    
    for i in range(n_rigidities):
        rig_min = rig_bins[i]
        rig_max = rig_bins[i+1]
        col_name = f"I_{rig_min:.2f}-{rig_max:.2f}GV"
        col_name_dI = f"dI_{rig_min:.2f}-{rig_max:.2f}GV"
        
        # Baseline flux (relative to 1.0)
        baseline = 1.0
        
        # Add periodic variations (27-day solar rotation)
        periodic = 0.02 * np.sin(2 * np.pi * t / (27 * 24))
        
        # Add Forbush decrease
        # Amplitude depends on rigidity (larger decrease at lower rigidity)
        amplitude = 0.15 * (1.0 - i * 0.15)
        tau_decrease = 24  # hours
        tau_recovery = 120  # hours
        
        fd = np.zeros(n_hours)
        mask_decrease = (t >= t0) & (t < t1)
        mask_recovery = t >= t1
        
        fd[mask_decrease] = -amplitude * (1 - np.exp(-(t[mask_decrease] - t0) / tau_decrease))
        fd[mask_recovery] = -amplitude * np.exp(-(t[mask_recovery] - t1) / tau_recovery)
        
        # Add noise
        noise = 0.01 * np.random.randn(n_hours)
        
        # Combine
        flux = baseline + periodic + fd + noise
        flux_relative = flux / baseline  # I = Phi/Phi_baseline
        
        ams_data[col_name] = flux_relative
        ams_data[col_name_dI] = flux_relative - 1.0  # dI = I - 1
    
    ams_df = pd.DataFrame(ams_data)
    
    # Generate OMNI data
    print("Generating OMNI solar wind data...")
    omni_data = {'datetime': datetimes}
    
    # B field (nT) - increases during shock and sheath
    B_baseline = 5.0
    B_shock = 20.0 * np.exp(-(t - t0)**2 / (48**2))  # Gaussian shock
    B_sheath = 15.0 * np.exp(-(t - t0) / 72)  # Exponential decay
    B_variation = 2.0 * np.sin(2 * np.pi * t / (27 * 24))
    B_noise = 0.5 * np.random.randn(n_hours)
    omni_data['B_avg_abs'] = np.maximum(0.5, B_baseline + B_shock + B_sheath + B_variation + B_noise)
    
    # Bz GSE (nT) - southward during main phase
    Bz_baseline = 0.0
    Bz_south = -10.0 * np.exp(-(t - (t0+12))**2 / (36**2))  # Southward excursion
    Bz_noise = 2.0 * np.random.randn(n_hours)
    omni_data['Bz_gse'] = Bz_baseline + Bz_south + Bz_noise
    
    # Proton density (n/cm^3)
    Np_baseline = 5.0
    Np_shock = 30.0 * np.exp(-(t - t0)**2 / (24**2))
    Np_noise = 1.0 * np.random.randn(n_hours)
    omni_data['Np'] = np.maximum(0.5, Np_baseline + Np_shock + Np_noise)
    
    # Solar wind speed (km/s)
    Vsw_baseline = 400.0
    Vsw_shock = 300.0 * np.exp(-(t - t0)**2 / (48**2))
    Vsw_noise = 10.0 * np.random.randn(n_hours)
    omni_data['Vsw'] = np.maximum(250.0, Vsw_baseline + Vsw_shock + Vsw_noise)
    
    omni_df = pd.DataFrame(omni_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    ams_file = os.path.join(output_dir, 'ams_processed.csv')
    omni_file = os.path.join(output_dir, 'omni_processed.csv')
    
    ams_df.to_csv(ams_file, index=False)
    print(f"Saved AMS data: {ams_file}")
    print(f"  Columns: {list(ams_df.columns)}")
    print(f"  Shape: {ams_df.shape}")
    
    omni_df.to_csv(omni_file, index=False)
    print(f"Saved OMNI data: {omni_file}")
    print(f"  Columns: {list(omni_df.columns)}")
    print(f"  Shape: {omni_df.shape}")
    
    print("\nTest data generation complete!")
    return ams_file, omni_file


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = '/tmp/test_xwt'
    
    generate_test_data(output_dir)
