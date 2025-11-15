#!/bin/bash
#
# Example Usage of extract_and_analyze_wavelet.py
# ================================================
#
# This script demonstrates how to use the AMS data extraction and wavelet analysis tool.

echo "============================================================"
echo "AMS DATA EXTRACTION AND WAVELET ANALYSIS - EXAMPLES"
echo "============================================================"
echo ""

# Example 1: Basic usage with date range
echo "Example 1: Analyze FD event from June-July 2015"
echo "------------------------------------------------"
echo "Command:"
echo "  python3 extract_and_analyze_wavelet.py --start 2015-06-19 --end 2015-07-07"
echo ""
echo "This will:"
echo "  - Extract AMS data from 2015-06-19 to 2015-07-07"
echo "  - Create folder: /home/zpw/Files/forbushDecrease/FDs/FD_20150619/"
echo "  - Analyze default rigidity bins: 1.0, 2.15, 5.37, 9.26, 16.6 GV"
echo "  - Generate plots and CSV files for each rigidity"
echo ""

# Example 2: Custom rigidity bins
echo "Example 2: Analyze specific rigidity bins"
echo "------------------------------------------"
echo "Command:"
echo "  python3 extract_and_analyze_wavelet.py \\"
echo "      --start 2017-03-01 \\"
echo "      --end 2017-12-01 \\"
echo "      --rigidities 1.0 5.37 16.6"
echo ""
echo "This will analyze only three rigidity bins: 1.0, 5.37, and 16.6 GV"
echo ""

# Example 3: Custom data and output directories
echo "Example 3: Use custom directories"
echo "----------------------------------"
echo "Command:"
echo "  python3 extract_and_analyze_wavelet.py \\"
echo "      --start 2017-03-01 \\"
echo "      --end 2017-12-01 \\"
echo "      --data-dir /custom/path/to/data \\"
echo "      --output-dir /custom/path/to/output"
echo ""
echo "This allows using data from a different location"
echo ""

# Example 4: Analyze full year
echo "Example 4: Analyze full year of data"
echo "-------------------------------------"
echo "Command:"
echo "  python3 extract_and_analyze_wavelet.py \\"
echo "      --start 2017-01-01 \\"
echo "      --end 2017-12-31"
echo ""
echo "This extracts and analyzes an entire year of AMS data"
echo ""

# Example 5: Batch processing multiple events
echo "Example 5: Batch process multiple FD events"
echo "--------------------------------------------"
echo "Script content (save as batch_analyze.sh):"
echo ""
cat << 'EOF'
#!/bin/bash

# Define FD events to analyze
events=(
    "2015-06-19 2015-07-07"
    "2017-03-01 2017-04-01"
    "2017-09-01 2017-10-01"
)

# Loop through events
for event in "${events[@]}"; do
    read -r start end <<< "$event"
    echo "Processing: $start to $end"
    python3 extract_and_analyze_wavelet.py --start $start --end $end
    echo "Completed: $start to $end"
    echo "----------------------------------------"
done

echo "All events processed!"
EOF
echo ""

# Example 6: Check output
echo "Example 6: Verify output files"
echo "-------------------------------"
echo "After running analysis, check the output directory:"
echo ""
echo "  ls -lh /home/zpw/Files/forbushDecrease/FDs/FD_20150619/"
echo ""
echo "You should see files like:"
echo "  - ams_extracted.csv                    (extracted raw data)"
echo "  - wavelet_1.00-1.16GV.png             (visualization plot)"
echo "  - power_1.00-1.16GV.csv               (power spectrum matrix)"
echo "  - global_spectrum_1.00-1.16GV.csv     (time-averaged spectrum)"
echo ""

# Example 7: View help
echo "Example 7: Get help"
echo "-------------------"
echo "Command:"
echo "  python3 extract_and_analyze_wavelet.py --help"
echo ""

echo "============================================================"
echo "For more details, see README_WAVELET.md"
echo "============================================================"
