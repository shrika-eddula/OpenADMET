#!/bin/bash
# Run advanced MPPB training with LogD feature and ensemble methods

# Set the correct Python executable from conda environment
PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin/python"

# Verify Python version
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

echo ""
echo "============================================================"
echo "Advanced MPPB Training Pipeline (Sequential: LogD → MPPB)"
echo "============================================================"

# Check command line argument
if [ "$1" == "test" ]; then
    echo "Running TEST configuration (quick validation)"
    echo "Configuration: 2 folds × 2 seeds = 4 models"
    echo ""

    $PYTHON_EXEC train_mppb_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD_MPPB.csv \
        --test_path ../Data/paired_endpoints/train_LogD_MPPB.csv \
        --depth 3 \
        --hidden_size 300 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 300 \
        --dropout 0.1 \
        --batch_size 32 \
        --epochs 10 \
        --patience 5 \
        --num_folds 2 \
        --ensemble_size 2 \
        --save_dir mppb_advanced_test

elif [ "$1" == "medium" ]; then
    echo "Running MEDIUM configuration (balanced performance)"
    echo "Configuration: 3 folds × 3 seeds = 9 models"
    echo "Expected time: 1-2 hours"
    echo ""

    $PYTHON_EXEC train_mppb_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD_MPPB.csv \
        --test_path ../Data/paired_endpoints/train_LogD_MPPB.csv \
        --depth 4 \
        --hidden_size 400 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 400 \
        --dropout 0.1 \
        --batch_size 64 \
        --epochs 50 \
        --patience 15 \
        --num_folds 3 \
        --ensemble_size 3 \
        --use_swa \
        --swa_start_epoch 40 \
        --save_dir mppb_advanced_medium

elif [ "$1" == "full" ]; then
    echo "Running FULL configuration (maximum performance)"
    echo "Configuration: 5 folds × 5 seeds = 25 models"
    echo "WARNING: This will take 4-8 hours!"
    echo ""

    # Create log directory
    mkdir -p mppb_advanced_full/logs

    # Run with nohup for long training
    nohup $PYTHON_EXEC train_mppb_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD_MPPB.csv \
        --test_path ../Data/paired_endpoints/train_LogD_MPPB.csv \
        --depth 5 \
        --hidden_size 500 \
        --ffn_num_layers 3 \
        --ffn_hidden_size 500 \
        --dropout 0.10 \
        --batch_size 64 \
        --epochs 100 \
        --patience 20 \
        --num_folds 5 \
        --ensemble_size 5 \
        --use_swa \
        --swa_start_epoch 80 \
        --save_dir mppb_advanced_full \
        > mppb_advanced_full/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Training started in background with PID: $!"
    echo "Monitor progress with: tail -f mppb_advanced_full/logs/training_*.log"

else
    echo "Usage: ./run_mppb_advanced.sh [test|medium|full]"
    echo ""
    echo "Configurations:"
    echo "  test   - Quick test (10 epochs, 2×2 models) ~15-30 minutes"
    echo "  medium - Balanced (50 epochs, 3×3 models) ~1-2 hours"
    echo "  full   - Maximum (100 epochs, 5×5 models) ~4-8 hours"
    echo ""
    echo "Features:"
    echo "  • Sequential prediction using LogD as molecular feature"
    echo "  • K-fold cross-validation"
    echo "  • Multi-seed ensemble"
    echo "  • Uncertainty quantification"
    echo "  • Stochastic Weight Averaging (medium/full)"
    echo ""
    echo "Example:"
    echo "  ./run_mppb_advanced.sh test    # Quick validation"
    echo "  ./run_mppb_advanced.sh medium  # Good performance"
    echo "  ./run_mppb_advanced.sh full    # Best performance"
    exit 1
fi

# Check if the script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training pipeline started successfully!"
else
    echo ""
    echo "❌ Failed to start training. Check the logs for details."
    exit 1
fi