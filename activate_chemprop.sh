#!/bin/bash
# Activation script for chemprop conda environment with proper PATH configuration

echo "Activating chemprop environment..."

# Source conda
eval "$(conda shell.bash hook)"

# Activate the chemprop environment
conda activate chemprop

# Fix PATH to prioritize conda Python over pyenv
export PATH="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin:$PATH"

# Remove pyenv from PATH temporarily for this session
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/.pyenv/" | tr '\n' ':' | sed 's/:$//')

# Add conda bin back at the beginning to ensure priority
export PATH="/opt/homebrew/Caskroom/miniforge/base/envs/chemprop/bin:$PATH"

# Verify the correct Python is being used
echo "Python location: $(which python)"
echo "Python version: $(python -V)"

# Verify chemprop can be imported
python -c "import chemprop; print('✓ Chemprop imported successfully')" 2>/dev/null || echo "⚠ Warning: Could not import chemprop"

echo "✓ Environment ready!"
echo ""
echo "To run training for all endpoints:"
echo "  Dry run: python chemprop_training/train_all_endpoints.py --dry_run"
echo "  Test (1 epoch): python chemprop_training/train_all_endpoints.py --max_epochs 1"
echo "  Full run: nohup python chemprop_training/train_all_endpoints.py > chemprop_training/outputs/training_all_endpoints_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"