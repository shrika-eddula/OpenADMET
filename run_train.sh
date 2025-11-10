#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=chemeleon
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j-%x.out

# Load your shell environment to activate your uv environment
# source chemprop_env/bin/activate
conda activate OpenADMET

# Run command or script
python models/CheMeleon/training.py models/CheMeleon/out
