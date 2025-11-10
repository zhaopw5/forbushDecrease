#!/bin/bash
# Example usage script for XWT analysis
# This script demonstrates various ways to run the XWT analysis program

# Set the base directory
WAVELET_DIR="/home/runner/work/forbushDecrease/forbushDecrease/wavelet"
cd "$WAVELET_DIR" || exit 1

echo "============================================"
echo "XWT Analysis - Example Usage"
echo "============================================"

# Example 1: Generate test data
echo -e "\n[Example 1] Generating test data..."
python3 generate_test_data.py /tmp/test_xwt_demo

# Example 2: Basic analysis with all defaults
echo -e "\n[Example 2] Running basic XWT analysis..."
python3 xwt_analysis.py --event-dir /tmp/test_xwt_demo --format png

# Example 3: Analyze specific rigidity channels only
echo -e "\n[Example 3] Analyzing specific rigidity channels..."
python3 xwt_analysis.py \
    --event-dir /tmp/test_xwt_demo \
    --rigidities "1.00-2.15GV,5.90-9.26GV" \
    --format png

# Example 4: Analyze specific OMNI parameters
echo -e "\n[Example 4] Analyzing specific OMNI parameters..."
python3 xwt_analysis.py \
    --event-dir /tmp/test_xwt_demo \
    --omni-params "B_avg_abs,Vsw" \
    --format both

# Example 5: With debug logging
echo -e "\n[Example 5] Running with debug logging..."
python3 xwt_analysis.py \
    --event-dir /tmp/test_xwt_demo \
    --rigidities "1.00-2.15GV" \
    --omni-params "B_avg_abs" \
    --format png \
    --log-level DEBUG

# Display results
echo -e "\n============================================"
echo "Analysis Complete!"
echo "============================================"
echo "Results saved to:"
echo "  Plots: /tmp/test_xwt_demo/plots_xwt/"
echo "  Data:  /tmp/test_xwt_demo/data_xwt/"
echo ""
echo "Total files generated:"
echo "  Plots: $(ls /tmp/test_xwt_demo/plots_xwt/ 2>/dev/null | wc -l)"
echo "  Data:  $(ls /tmp/test_xwt_demo/data_xwt/ 2>/dev/null | wc -l)"
