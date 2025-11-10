# Cross Wavelet Transform (XWT) Analysis

This directory contains tools for performing Cross Wavelet Transform (XWT) analysis to compare AMS hourly flux data with OMNI solar wind parameters.

## Overview

The XWT analysis program (`xwt_analysis.py`) performs cross-wavelet transforms between cosmic ray flux measurements from AMS and solar wind parameters from OMNI data. This reveals time-frequency correlations and phase relationships between cosmic ray variations and solar wind conditions.

## Files

- **xwt_analysis.py** - Main XWT analysis program
- **generate_test_data.py** - Generate synthetic test data for validation
- **waveletFunctions.py** - Core wavelet transform functions (Torrence & Compo, 1998)
- **ams_wavelet.py** - AMS-specific wavelet analysis utilities
- **wavelet_v251110.py** - Template wavelet analysis script
- **xwt_correlation.py** - Alternative XWT implementation

## Requirements

```bash
pip install numpy pandas matplotlib scipy
```

## Input Data Format

The program expects two CSV files in the event directory:

### ams_processed.csv
Must contain:
- `datetime` - Timestamp column (hourly)
- `I_X.XX-Y.YYGV` - Relative intensity for rigidity bins (e.g., `I_1.00-2.15GV`)
- `dI_X.XX-Y.YYGV` - Relative intensity deviation (e.g., `dI_1.00-2.15GV`)

### omni_processed.csv
Must contain:
- `datetime` - Timestamp column (hourly)
- OMNI parameters: `B_avg_abs`, `Bz_gse`, `Np`, `Vsw`, etc.

## Usage

### Basic Usage

Analyze all rigidity channels with default OMNI parameters:

```bash
python xwt_analysis.py --event-dir FD_20150619
```

### Select Specific Rigidities

```bash
python xwt_analysis.py --event-dir FD_20150619 \
    --rigidities "5.37-5.90GV,9.26-10.10GV"
```

### Select Specific OMNI Parameters

```bash
python xwt_analysis.py --event-dir FD_20150619 \
    --omni-params "B_avg_abs,Vsw"
```

### Control Output Format

```bash
# PDF only
python xwt_analysis.py --event-dir FD_20150619 --format pdf

# PNG only
python xwt_analysis.py --event-dir FD_20150619 --format png

# Both formats (default)
python xwt_analysis.py --event-dir FD_20150619 --format both
```

### Adjust Logging Level

```bash
python xwt_analysis.py --event-dir FD_20150619 --log-level DEBUG
```

## Output Structure

The program creates two directories in the event folder:

```
FD_20150619/
├── ams_processed.csv
├── omni_processed.csv
├── plots_xwt/                          # XWT visualization plots
│   ├── xwt_B_avg_abs_vs_1.00-2.15GV.png
│   ├── xwt_B_avg_abs_vs_1.00-2.15GV.pdf
│   ├── global_xwt_B_avg_abs_vs_1.00-2.15GV.png
│   └── ...
└── data_xwt/                           # XWT data output
    ├── xwt_stats_B_avg_abs_vs_1.00-2.15GV.csv
    ├── xwt_power_B_avg_abs_vs_1.00-2.15GV.csv
    └── ...
```

## Output Files Description

### XWT Plots (`plots_xwt/`)

1. **XWT Power Spectrum** (`xwt_*.png/pdf`)
   - Cross-wavelet power as function of time and period
   - Contour levels showing power magnitude
   - 95% significance contours (black lines)
   - Cone of Influence (COI) shading
   - Phase arrows indicating lead/lag relationships
   - Logarithmic colorbar

2. **Global Spectrum** (`global_xwt_*.png/pdf`)
   - Time-averaged cross-wavelet power vs period
   - Shows dominant periodicities in the correlation

### Data Files (`data_xwt/`)

1. **Statistics CSV** (`xwt_stats_*.csv`)
   - Columns: `period_hours`, `global_xwt_power`, `min_xwt_power`, `max_xwt_power`, `median_xwt_power`
   - Summary statistics for each period scale

2. **Full XWT Power** (`xwt_power_*.csv`)
   - Complete time-frequency power spectrum
   - Rows: Period scales
   - Columns: Timestamps
   - Values: Cross-wavelet power

## Interpretation

### XWT Power
- High power (red) indicates strong correlation at that time-frequency
- Significance contours show where correlations exceed 95% confidence

### Phase Arrows
- → Right: Variables in phase (0°)
- ↑ Up: First variable leads by 90°
- ← Left: Variables anti-phase (180°)
- ↓ Down: Second variable leads by 90°

### Cone of Influence (COI)
- Hatched region where edge effects become important
- Results in this region should be interpreted with caution

## Testing

Generate synthetic test data and run analysis:

```bash
# Generate test data
python generate_test_data.py /tmp/test_xwt

# Run XWT analysis on test data
python xwt_analysis.py --event-dir /tmp/test_xwt \
    --rigidities "1.00-2.15GV" \
    --omni-params "B_avg_abs,Vsw" \
    --format png
```

## Configuration

Key parameters can be modified in `xwt_analysis.py`:

```python
# Wavelet parameters
DT_H = 1.0              # Sampling interval (hours)
MOTHER = 'MORLET'       # Wavelet type
K0 = 6.0                # Morlet wavenumber
DJ = 0.25               # Scale spacing

# Plotting parameters
VMIN = 1e-2             # Colorbar minimum
VMAX = 1e2              # Colorbar maximum
N_CONTOURS = 40         # Number of contour levels
```

## Methodology

The analysis performs the following steps:

1. **Data Loading**: Merge AMS and OMNI data on common timestamps
2. **Preprocessing**: 
   - Remove mean
   - Normalize by standard deviation
   - Handle missing data
3. **Wavelet Transform**: 
   - Compute Continuous Wavelet Transform (CWT) for both series
   - Calculate cross-wavelet transform: Wxy = Wx × conj(Wy)
4. **Significance Testing**:
   - Compute 95% confidence level based on red noise background
   - Account for lag-1 autocorrelation
5. **Visualization**:
   - Generate XWT power spectrum with significance contours
   - Add COI and phase arrows
   - Create global spectrum (time-averaged)
6. **Output**: Save plots and statistical summaries

## References

- Torrence, C., and G. P. Compo (1998), A Practical Guide to Wavelet Analysis, 
  Bull. Amer. Meteor. Soc., 79, 61–78.
- Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross 
  wavelet transform and wavelet coherence to geophysical time series. 
  Nonlinear processes in geophysics, 11(5/6), 561-566.

## Support

For questions or issues, please refer to the main project documentation or 
create an issue in the repository.

## License

This code is provided for research purposes. The wavelet functions are based on 
the work of Torrence & Compo and should be acknowledged in publications.
