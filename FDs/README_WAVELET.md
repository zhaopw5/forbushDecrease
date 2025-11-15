# AMS Data Extraction and Wavelet Analysis

## Overview

The `extract_and_analyze_wavelet.py` script provides automated extraction and wavelet analysis of AMS (Alpha Magnetic Spectrometer) proton flux data for Forbush Decrease (FD) events.

## Features

1. **Data Extraction**: Automatically reads AMS data from raw data directory
2. **Time Range Selection**: Extracts data for user-specified time ranges
3. **Organized Output**: Creates standardized FD_{YYYYMMDD} folders
4. **CSV Export**: Saves extracted data in CSV format
5. **Wavelet Analysis**: Performs continuous wavelet transform using Morlet wavelet
6. **Visualization**: Generates wavelet power spectrum plots
7. **Multi-Rigidity Support**: Analyzes multiple rigidity ranges simultaneously
8. **Result Archiving**: Saves power values, periods, and global spectra

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```bash
pip install numpy pandas matplotlib scipy
```

## Usage

### Basic Usage

```bash
python extract_and_analyze_wavelet.py --start YYYY-MM-DD --end YYYY-MM-DD
```

### Examples

1. Analyze data from March to December 2017:
```bash
python extract_and_analyze_wavelet.py --start 2017-03-01 --end 2017-12-01
```

2. Analyze the 2015 FD event:
```bash
python extract_and_analyze_wavelet.py --start 2015-06-19 --end 2015-07-07
```

3. Custom data and output directories:
```bash
python extract_and_analyze_wavelet.py \
    --start 2017-03-01 \
    --end 2017-12-01 \
    --data-dir /path/to/data \
    --output-dir /path/to/output
```

4. Analyze specific rigidity bins:
```bash
python extract_and_analyze_wavelet.py \
    --start 2017-03-01 \
    --end 2017-12-01 \
    --rigidities 1.0 2.15 5.37 9.26 16.6
```

### Command-Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--start` | Yes | - | Start date in YYYY-MM-DD format |
| `--end` | Yes | - | End date in YYYY-MM-DD format |
| `--data-dir` | No | `/home/zpw/Files/forbushDecrease/raw_data/` | Raw data directory path |
| `--output-dir` | No | `/home/zpw/Files/forbushDecrease/FDs/` | Output base directory |
| `--rigidities` | No | `1.0 2.15 5.37 9.26 16.6` | Rigidity bins (GV) to analyze |

## Output Structure

For a start date of 2017-03-01, the script creates:

```
/home/zpw/Files/forbushDecrease/FDs/FD_20170301/
├── ams_extracted.csv                          # Extracted AMS data
├── wavelet_1.00-1.16GV.png                   # Wavelet plot
├── power_1.00-1.16GV.csv                     # Power spectrum
├── global_spectrum_1.00-1.16GV.csv           # Global wavelet spectrum
├── wavelet_2.15-2.40GV.png
├── power_2.15-2.40GV.csv
├── global_spectrum_2.15-2.40GV.csv
└── ... (similar files for other rigidity ranges)
```

## Output Files

### 1. ams_extracted.csv
Raw AMS data extracted for the specified time range with columns:
- `date`: Datetime stamp
- `rigidity_min`: Lower bound of rigidity bin (GV)
- `rigidity_max`: Upper bound of rigidity bin (GV)
- `flux`: Proton flux (m⁻²s⁻¹sr⁻¹GV⁻¹)
- Additional metadata columns

### 2. wavelet_{rigidity}.png
Visualization plot containing:
- **Top panel**: Time series of proton flux
- **Bottom panel**: Wavelet power spectrum with:
  - Color-coded power levels (logarithmic scale)
  - Black contours: 95% significance level
  - Hatched region: Cone of influence (edge effects)

### 3. power_{rigidity}.csv
2D power spectrum matrix:
- Rows: Period values (days)
- Columns: Time values (datetime)
- Values: Normalized wavelet power

### 4. global_spectrum_{rigidity}.csv
Time-averaged wavelet spectrum:
- `period_days`: Period values
- `global_power`: Time-averaged power
- `significance_95`: 95% confidence level

## Wavelet Analysis Details

### Method
- **Wavelet Type**: Morlet wavelet with k₀ = 6
- **Normalization**: Time series are mean-subtracted
- **Scales**: Logarithmically spaced (dj = 0.25)
- **Padding**: Zero-padding applied for FFT efficiency

### Parameters
The default wavelet parameters can be modified in the script:

```python
WAVELET_PARAMS = {
    'dt': 1.0,          # Sampling interval in days
    'pad': 1,           # Zero padding
    'dj': 0.25,         # Scale spacing
    's0': -1,           # Smallest scale (auto: 2*dt)
    'J1': -1,           # Number of scales (auto)
    'mother': 'MORLET', # Wavelet type
    'param': 6.0        # Morlet wavenumber k₀
}
```

### Significance Testing
- **Method**: Chi-square test for red noise background
- **Level**: 95% confidence
- **Lag-1 autocorrelation**: 0.0 (white noise assumption)

## Rigidity Bins

The script uses standard AMS rigidity bins (in GV):

```
1.00, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.40, 2.67, 2.97,
3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.90, 6.47, 7.09, 7.76,
8.48, 9.26, 10.1, 11.0, 13.0, 16.6, 22.8, 33.5, 48.5, 69.7, 100.0
```

Default analysis rigidities: **1.00, 2.15, 5.37, 9.26, 16.6 GV**

## Data Requirements

### Input Data Format
The script expects AMS data in CSV format at:
```
{data_dir}/ams/flux_long.csv
```

Required columns:
- `date`: Timestamp (parseable by pandas)
- `rigidity_min`: Lower rigidity bound (GV)
- `rigidity_max`: Upper rigidity bound (GV)
- `flux`: Proton flux values

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install required dependencies
   ```bash
   pip install numpy pandas matplotlib scipy
   ```

2. **FileNotFoundError**: Check data directory path
   - Verify `--data-dir` points to correct location
   - Ensure `ams/flux_long.csv` exists in data directory

3. **No data found**: Verify date range
   - Check that AMS data exists for specified dates
   - Ensure date format is YYYY-MM-DD

4. **Empty plots**: Check rigidity bins
   - Verify rigidity values exist in data
   - Try different rigidity bins with `--rigidities`

## Integration with Existing Scripts

This script is designed to work alongside existing FD analysis tools:

- **preprocess.py**: Pre-processes AMS and OMNI data
- **waveletFunctions.py**: Core wavelet transform functions
- **xwt_correlation.py**: Cross-wavelet analysis

## References

### Wavelet Analysis
- Torrence, C. and G. P. Compo (1998): "A Practical Guide to Wavelet Analysis", 
  Bull. Amer. Meteor. Soc., 79, 61-78.
- Wavelet software: http://paos.colorado.edu/research/wavelets/

### AMS Data
- AMS Collaboration: Alpha Magnetic Spectrometer data
- Cosmic ray proton flux measurements

## Author & License

This script is part of the forbushDecrease analysis pipeline.

For questions or issues, please contact the repository maintainer.
