#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=inference
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-%x.out

# Load your shell environment to activate your uv environment
# source chemprop_env/bin/activate
conda activate OpenADMET

# Run command or script
python models/CheMeleon/inference.py models/CheMeleon/out/2025-11-03_11-47-05


