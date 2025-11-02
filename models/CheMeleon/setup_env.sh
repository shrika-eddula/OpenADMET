#!/bin/bash

# Setup script for CheMeleon environment
# This script creates a Python environment using UV and installs all required dependencies

# Exit on error
set -e

echo "Setting up CheMeleon environment..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
fi

# Create environment
echo "Creating Python 3.11 environment with UV..."
uv venv --python=3.11 chemprop_env

# Activate environment
echo "Activating environment..."
source chemprop_env/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run: source chemprop_env/bin/activate"
echo ""
echo "IMPORTANT: Before running training or inference, you need to log in to HuggingFace:"
echo "huggingface-cli login"
echo ""
echo "Then you can run:"
echo "python training.py /path/to/output/directory"
echo "python inference.py /path/to/output/directory"
