#!/bin/bash
# Run MPPB training with LogD as additional feature

# Set the correct Python executable from conda environment
PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin/python"

# Verify Python version
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

echo ""
echo "============================================================"
echo "MPPB Sequential Prediction Training (with LogD feature)"
echo "============================================================"

# Check command line argument
if [ "$1" == "test" ]; then
    echo "Running TEST configuration (reduced settings for quick testing)"
    echo ""

    $PYTHON_EXEC train_mppb_with_logd.py \
        --data_path ../Data/endpoints/train_big_LogD_MPPB.csv \
        --test_path ../Data/paired_endpoints/train_LogD_MPPB.csv \
        --depth 3 \
        --hidden_size 300 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 300 \
        --dropout 0.1 \
        --batch_size 32 \
        --epochs 5 \
        --patience 3 \
        --save_dir mppb_logd_test_results

elif [ "$1" == "full" ]; then
    echo "Running FULL configuration (complete training)"
    echo ""

    $PYTHON_EXEC train_mppb_with_logd.py \
        --data_path ../Data/endpoints/train_big_LogD_MPPB.csv \
        --test_path ../Data/paired_endpoints/train_LogD_MPPB.csv \
        --depth 4 \
        --hidden_size 400 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 400 \
        --dropout 0.1 \
        --batch_size 64 \
        --epochs 100 \
        --patience 15 \
        --save_dir mppb_logd_full_results

else
    echo "Usage: ./run_mppb_logd.sh [test|full]"
    echo ""
    echo "Configurations:"
    echo "  test - Quick test run (5 epochs) ~5 minutes"
    echo "  full - Complete training (100 epochs) ~30-60 minutes"
    echo ""
    echo "Example:"
    echo "  ./run_mppb_logd.sh test    # Quick test"
    echo "  ./run_mppb_logd.sh full    # Full training"
    exit 1
fi

# Check if the script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
else
    echo ""
    echo "❌ Training encountered an error. Check the logs for details."
    exit 1
fi