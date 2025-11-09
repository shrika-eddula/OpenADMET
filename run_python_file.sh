#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=embed
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-%x.out

# Load your shell environment to activate your Conda environment
source /home/jeanshe/.bashrc
source chemprop_env/bin/activate

# Run command or script
# python representations/chemeleon_fingerprint.py
python representations/generate_fingerprints.py
echo "Command completed."