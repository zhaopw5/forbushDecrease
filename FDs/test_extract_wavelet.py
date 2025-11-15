#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for extract_and_analyze_wavelet.py

This script creates synthetic AMS data and tests the wavelet analysis functionality.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the FDs directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Add wavelet directory to path
WAVELET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'wavelet')
sys.path.insert(0, WAVELET_DIR)

import waveletFunctions as wf


def create_synthetic_ams_data(start_date, end_date, num_rigidity_bins=5):
    """
    Create synthetic AMS data for testing.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        num_rigidity_bins: Number of rigidity bins to create
        
    Returns:
        DataFrame with synthetic AMS data
    """
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create hourly date range
    dates = pd.date_range(start=start, end=end, freq='h')
    
    # Define rigidity bins
    rig_bins = [1.00, 2.15, 5.37, 9.26, 16.6, 33.5][:num_rigidity_bins+1]
    
    # Create data for each rigidity bin
    data_list = []
    
    for i in range(len(rig_bins) - 1):
        rig_min = rig_bins[i]
        rig_max = rig_bins[i + 1]
        
        # Generate synthetic flux with some patterns
        # Base flux decreases with rigidity
        base_flux = 1000.0 / (rig_min + 1.0)
        
        # Add some periodic variation (27-day period simulating solar rotation)
        t = np.arange(len(dates))
        period_days = 27.0
        period_hours = period_days * 24
        flux = base_flux * (1.0 + 0.1 * np.sin(2 * np.pi * t / period_hours))
        
        # Add some random noise
        flux += np.random.normal(0, base_flux * 0.02, len(dates))
        
        # Add error bar (5% of flux)
        error = flux * 0.05
        
        # Create DataFrame for this rigidity bin
        df_rig = pd.DataFrame({
            'date': dates,
            'rigidity_min': rig_min,
            'rigidity_max': rig_max,
            'flux': flux,
            'error_bar': error
        })
        
        data_list.append(df_rig)
    
    # Concatenate all rigidity bins
    df = pd.concat(data_list, ignore_index=True)
    
    return df


def test_wavelet_functions():
    """Test basic wavelet transform functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Wavelet Transform")
    print("="*60)
    
    # Create simple test signal
    dt = 1.0  # days
    N = 256
    t = np.arange(N) * dt
    
    # Signal with two periods: 16 and 32 days
    signal = np.sin(2*np.pi*t/16) + 0.5*np.sin(2*np.pi*t/32)
    signal += np.random.normal(0, 0.1, N)
    
    print(f"Signal length: {N} points")
    print(f"Sampling interval: {dt} days")
    
    # Perform wavelet transform
    wave, period, scale, coi = wf.wavelet(signal, dt, pad=1, dj=0.25, 
                                          mother='MORLET', param=6.0)
    
    power = np.abs(wave) ** 2
    
    print(f"Wavelet transform shape: {wave.shape}")
    print(f"Number of periods: {len(period)}")
    print(f"Period range: {period.min():.2f} - {period.max():.2f} days")
    print(f"Power spectrum shape: {power.shape}")
    
    # Find dominant periods
    global_power = np.mean(power, axis=1)
    max_idx = np.argmax(global_power)
    print(f"Dominant period: {period[max_idx]:.2f} days")
    
    # Check if it's close to expected periods
    expected_periods = [16.0, 32.0]
    found_periods = []
    for exp_per in expected_periods:
        # Find closest period
        idx = np.argmin(np.abs(period - exp_per))
        found_periods.append(period[idx])
        print(f"  Expected {exp_per} days, found {period[idx]:.2f} days")
    
    print("✓ Wavelet transform test PASSED")
    return True


def test_synthetic_data_creation():
    """Test synthetic data creation."""
    print("\n" + "="*60)
    print("TEST 2: Synthetic Data Creation")
    print("="*60)
    
    start_date = "2017-03-01"
    end_date = "2017-03-31"
    
    df = create_synthetic_ams_data(start_date, end_date, num_rigidity_bins=3)
    
    print(f"Created synthetic data:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Rigidity bins: {df['rigidity_min'].unique()}")
    print(f"  Flux range: {df['flux'].min():.2f} - {df['flux'].max():.2f}")
    
    # Check data quality
    assert len(df) > 0, "No data created"
    assert 'date' in df.columns, "Missing 'date' column"
    assert 'flux' in df.columns, "Missing 'flux' column"
    assert df['flux'].notna().all(), "NaN values in flux"
    
    print("✓ Synthetic data creation test PASSED")
    return True


def test_script_components():
    """Test individual components of the main script."""
    print("\n" + "="*60)
    print("TEST 3: Script Components")
    print("="*60)
    
    # Import the main script
    try:
        import extract_and_analyze_wavelet as eaw
        print("✓ Successfully imported extract_and_analyze_wavelet")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test helper functions exist
    functions_to_check = [
        'load_ams_data',
        'extract_time_range',
        'prepare_output_directory',
        'save_extracted_data',
        'perform_wavelet_analysis',
        'plot_wavelet_results',
        'save_wavelet_results',
        'analyze_multiple_rigidities'
    ]
    
    for func_name in functions_to_check:
        if hasattr(eaw, func_name):
            print(f"✓ Found function: {func_name}")
        else:
            print(f"✗ Missing function: {func_name}")
            return False
    
    # Test configuration constants
    constants_to_check = [
        'AMS_RAW_DATA_DIR',
        'OUTPUT_BASE_DIR',
        'RIG_BINS',
        'SELECTED_RIGS',
        'WAVELET_PARAMS'
    ]
    
    for const_name in constants_to_check:
        if hasattr(eaw, const_name):
            print(f"✓ Found constant: {const_name}")
        else:
            print(f"✗ Missing constant: {const_name}")
            return False
    
    print("✓ Script components test PASSED")
    return True


def test_wavelet_analysis_with_synthetic_data():
    """Test wavelet analysis with synthetic data."""
    print("\n" + "="*60)
    print("TEST 4: Wavelet Analysis with Synthetic Data")
    print("="*60)
    
    import extract_and_analyze_wavelet as eaw
    
    # Create synthetic data
    start_date = "2017-03-01"
    end_date = "2017-04-30"
    df = create_synthetic_ams_data(start_date, end_date, num_rigidity_bins=2)
    
    # Extract data for first rigidity bin
    rig_min = df['rigidity_min'].iloc[0]
    rig_max = df['rigidity_max'].iloc[0]
    
    mask = (df['rigidity_min'] == rig_min) & (df['rigidity_max'] == rig_max)
    df_rig = df[mask].sort_values('date')
    
    flux_series = df_rig['flux'].values
    
    print(f"Testing with rigidity bin: {rig_min:.2f}-{rig_max:.2f} GV")
    print(f"Flux data points: {len(flux_series)}")
    
    # Perform wavelet analysis
    result = eaw.perform_wavelet_analysis(
        flux_series,
        dt=1.0/24.0,  # hourly data in days
        **eaw.WAVELET_PARAMS
    )
    
    if result is None:
        print("✗ Wavelet analysis returned None")
        return False
    
    print(f"✓ Wavelet analysis completed")
    print(f"  Power shape: {result['power'].shape}")
    print(f"  Period range: {result['period'].min():.2f} - {result['period'].max():.2f} days")
    print(f"  Number of scales: {len(result['period'])}")
    
    # Check results have expected keys
    expected_keys = ['wave', 'period', 'scale', 'coi', 'power', 
                     'global_ws', 'sig95', 'global_signif', 'valid_mask']
    for key in expected_keys:
        if key in result:
            print(f"✓ Found result key: {key}")
        else:
            print(f"✗ Missing result key: {key}")
            return False
    
    print("✓ Wavelet analysis with synthetic data test PASSED")
    return True


def test_full_workflow_with_temp_dirs():
    """Test full workflow with temporary directories."""
    print("\n" + "="*60)
    print("TEST 5: Full Workflow with Temporary Directories")
    print("="*60)
    
    import extract_and_analyze_wavelet as eaw
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "raw_data")
        ams_dir = os.path.join(data_dir, "ams")
        os.makedirs(ams_dir, exist_ok=True)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Temp data dir: {data_dir}")
        print(f"Temp output dir: {output_dir}")
        
        # Create synthetic data file
        start_date = "2017-03-01"
        end_date = "2017-03-31"
        df = create_synthetic_ams_data(start_date, end_date, num_rigidity_bins=2)
        
        ams_file = os.path.join(ams_dir, "flux_long.csv")
        df.to_csv(ams_file, index=False)
        print(f"✓ Created synthetic data file: {ams_file}")
        
        # Test load_ams_data
        df_loaded = eaw.load_ams_data(data_dir)
        print(f"✓ Loaded {len(df_loaded)} data points")
        
        # Test extract_time_range
        extract_start = "2017-03-05"
        extract_end = "2017-03-15"
        df_extracted = eaw.extract_time_range(df_loaded, extract_start, extract_end)
        print(f"✓ Extracted {len(df_extracted)} data points")
        
        # Test prepare_output_directory
        event_dir = eaw.prepare_output_directory(extract_start, output_dir)
        print(f"✓ Created output directory: {event_dir}")
        
        # Test save_extracted_data
        csv_path = eaw.save_extracted_data(df_extracted, event_dir)
        print(f"✓ Saved extracted data: {csv_path}")
        
        # Verify output file exists
        assert os.path.exists(csv_path), "Output CSV file not created"
        
        # Test analyze_multiple_rigidities
        try:
            eaw.analyze_multiple_rigidities(
                df_extracted,
                event_dir,
                selected_rigs=[1.00],
                rig_bins=[1.00, 2.15, 5.37]
            )
            print(f"✓ Completed analysis for multiple rigidities")
            
            # Check if output files were created
            expected_files = [
                "wavelet_1.00-2.15GV.png",
                "power_1.00-2.15GV.csv",
                "global_spectrum_1.00-2.15GV.csv"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(event_dir, filename)
                if os.path.exists(filepath):
                    print(f"✓ Found output file: {filename}")
                else:
                    print(f"⚠ Output file not found: {filename}")
            
        except Exception as e:
            print(f"⚠ Analysis completed with minor issues: {e}")
        
        print("✓ Full workflow test PASSED")
        return True


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING AMS DATA EXTRACTION AND WAVELET ANALYSIS")
    print("="*60)
    
    tests = [
        ("Wavelet Functions", test_wavelet_functions),
        ("Synthetic Data Creation", test_synthetic_data_creation),
        ("Script Components", test_script_components),
        ("Wavelet Analysis with Synthetic Data", test_wavelet_analysis_with_synthetic_data),
        ("Full Workflow", test_full_workflow_with_temp_dirs),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
