#!/bin/bash
# Run advanced LogD training with all improvements

# Set the correct Python executable from conda environment
PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin/python"

# Verify Python version
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

echo ""
echo "============================================================"
echo "Advanced LogD Training Pipeline"
echo "============================================================"

# Check command line argument
if [ "$1" == "test" ]; then
    echo "Running TEST configuration (reduced settings for quick testing)"
    echo ""

    $PYTHON_EXEC train_logd_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD.csv \
        --test_path ../Data/test.csv \
        --depth 3 \
        --hidden_size 300 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 300 \
        --dropout 0.1 \
        --batch_size 32 \
        --epochs 5 \
        --num_folds 2 \
        --ensemble_size 2 \
        --save_dir logd_test_results

elif [ "$1" == "medium" ]; then
    echo "Running MEDIUM configuration (balanced settings)"
    echo ""

    $PYTHON_EXEC train_logd_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD.csv \
        --test_path ../Data/test.csv \
        --features_generator rdkit_2d_normalized \
        --depth 4 \
        --hidden_size 400 \
        --ffn_num_layers 2 \
        --ffn_hidden_size 400 \
        --dropout 0.1 \
        --batch_size 64 \
        --epochs 30 \
        --patience 10 \
        --num_folds 3 \
        --ensemble_size 3 \
        --use_swa \
        --swa_start_epoch 20 \
        --save_dir logd_medium_results

elif [ "$1" == "full" ]; then
    echo "Running FULL configuration (all features, maximum performance)"
    echo "WARNING: This will take 24-48 hours on GPU!"
    echo ""

    # Create log directory
    mkdir -p logd_advanced_results/logs

    # Run with nohup for long training
    nohup $PYTHON_EXEC train_logd_advanced.py \
        --data_path ../Data/endpoints/train_big_LogD.csv \
        --test_path ../Data/test.csv \
        --features_generator rdkit_2d_normalized,morgan_count \
        --depth 6 \
        --hidden_size 600 \
        --ffn_num_layers 3 \
        --ffn_hidden_size 600 \
        --dropout 0.10 \
        --batch_size 64 \
        --epochs 120 \
        --warmup_epochs 5 \
        --patience 20 \
        --num_folds 5 \
        --ensemble_size 5 \
        --use_swa \
        --swa_start_epoch 90 \
        --swa_lr 1e-4 \
        --winsorize \
        --save_dir logd_advanced_results \
        > logd_advanced_results/logs/training_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Training started in background with PID: $!"
    echo "Monitor progress with: tail -f logd_advanced_results/logs/training_full_*.log"

else
    echo "Usage: ./run_advanced_logd.sh [test|medium|full]"
    echo ""
    echo "Configurations:"
    echo "  test   - Quick test run (5 epochs, 2 folds, 2 seeds) ~30 minutes"
    echo "  medium - Balanced training (30 epochs, 3 folds, 3 seeds) ~4-6 hours"
    echo "  full   - Maximum performance (120 epochs, 5 folds, 5 seeds) ~24-48 hours"
    echo ""
    echo "Example:"
    echo "  ./run_advanced_logd.sh test    # Quick test to verify everything works"
    echo "  ./run_advanced_logd.sh medium  # Good balance of time and performance"
    echo "  ./run_advanced_logd.sh full    # Best performance (runs in background)"
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