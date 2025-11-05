#!/bin/bash
# Generate predictions for all ADMET endpoints on test set

# Set the correct Python executable from conda environment
PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin/python"

# Verify Python version
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

echo ""
echo "============================================================"
echo "Generating predictions for all ADMET endpoints"
echo "============================================================"

# Run the prediction script
$PYTHON_EXEC generate_all_predictions.py

# Check if the script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Predictions generated successfully!"
    echo ""
    echo "Output files:"
    echo "  - predictions/all_endpoints_predictions_latest.csv"
    echo "  - predictions/all_endpoints_predictions_*.csv (timestamped)"
else
    echo ""
    echo "❌ Error generating predictions. Check the error messages above."
    exit 1
fi