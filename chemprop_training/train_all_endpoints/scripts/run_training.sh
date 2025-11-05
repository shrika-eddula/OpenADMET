#!/bin/bash
# Direct run script using the correct Python from conda environment

# Set the correct Python executable
PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin/python"

# Verify Python version
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

# Check if this is a dry run or full run
if [ "$1" == "dry" ]; then
    echo "Running dry run..."
    $PYTHON_EXEC train_all_endpoints.py --dry_run
elif [ "$1" == "full" ]; then
    echo "Running full training..."
    LOG_FILE="outputs/training_all_endpoints_$(date +%Y%m%d_%H%M%S).log"
    echo "Log file: $LOG_FILE"
    nohup $PYTHON_EXEC train_all_endpoints.py > "$LOG_FILE" 2>&1 &
    echo "Training started in background. PID: $!"
    echo "Monitor with: tail -f $LOG_FILE"
elif [ "$1" == "test" ]; then
    echo "Running test with 1 epoch..."
    $PYTHON_EXEC train_all_endpoints.py --max_epochs 1
else
    echo "Usage: ./run_training.sh [dry|test|full]"
    echo "  dry  - Run in dry-run mode (no actual training)"
    echo "  test - Run with max_epochs=1 for testing"
    echo "  full - Run full training in background"
fi