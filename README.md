# OpenADMET

OpenADMET ExpansionRx Challenge - Machine Learning for ADMET Property Prediction

## Overview

This project implements machine learning models for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of molecules. The project includes both traditional machine learning (XGBoost) and deep learning (Chemprop MPNN) approaches for various endpoints including LogD, kinetic solubility (KSOL), human/mouse liver microsome clearance (HLM/MLM CLint), and other pharmacological properties.

### Key Features
- **Dual Approach**: Both XGBoost and Chemprop MPNN models for comprehensive predictions
- **9 ADMET Endpoints**: Complete coverage of key pharmacological properties
- **Automated Training Pipeline**: Scripts for training all endpoints with a single command
- **HuggingFace Integration**: Direct data loading from the OpenADMET repository
- **Scaffold Splitting**: Ensures robust model evaluation and generalization

## Project Structure

```
OpenADMET/
├── Data/                       # Data directory
│   ├── load_data.py           # Script to download data from HuggingFace
│   ├── save_data.py           # Script to save data in various formats
│   ├── endpoints/             # Separated datasets by endpoint (generated)
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset (blinded labels)
│   └── teaser.csv             # Teaser dataset for exploration
├── chemprop_training/         # Chemprop training scripts
│   ├── train_all_endpoints.py # Train models for all ADMET endpoints
│   ├── config.json            # Training configuration
│   └── outputs/               # Training logs and outputs
├── checkpoints/               # Saved model checkpoints
├── predictions/               # Model predictions
├── Notebooks/                 # Jupyter notebooks for exploration
│   ├── TeaserDataExploration.ipynb
│   └── chemprop_LogD_training.ipynb
├── activate_chemprop.sh      # Chemprop environment activation script
├── run_training.sh            # Training execution script
├── config.yaml                # Configuration file with data URLs
├── requirements.txt           # Python package dependencies (XGBoost)
├── separate_endpoints.py      # Script to separate data by endpoints
└── openADMETvenv/            # Virtual environment (created during setup)
```

## Setup Instructions

This project uses two different environments:
1. **XGBoost environment** - For traditional ML models
2. **Chemprop environment** - For deep learning MPNN models

### Part A: XGBoost Setup

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd OpenADMET
```

#### 2. Create Virtual Environment for XGBoost

```bash
# Create virtual environment
python3 -m venv openADMETvenv

# Activate virtual environment
source openADMETvenv/bin/activate  # On macOS/Linux
# or
openADMETvenv\Scripts\activate     # On Windows
```

#### 3. Install XGBoost Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

The `requirements.txt` includes:
- **Core ML**: xgboost, scikit-learn, pandas, numpy
- **Cheminformatics**: rdkit
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, huggingface-hub, datasets

### Part B: Chemprop Setup (for MPNN models)

Chemprop requires a separate conda environment due to specific PyTorch and Lightning dependencies.

#### 1. Install Conda (if not already installed)

Download and install Miniforge or Anaconda from:
- Miniforge (recommended): https://github.com/conda-forge/miniforge
- Anaconda: https://www.anaconda.com/products/individual

#### 2. Create Chemprop Conda Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n chemprop python=3.12 -y

# Activate the environment
conda activate chemprop
```

#### 3. Install Chemprop

```bash
# Install Chemprop v2.0+ with conda
conda install -c conda-forge chemprop -y

# Or install with pip (alternative)
pip install chemprop

# Install additional dependencies
pip install lightning tensorboard jupyter ipykernel
```

#### 4. Verify Chemprop Installation

```bash
python -c "import chemprop; print(f'Chemprop version: {chemprop.__version__}')"
```

#### 5. Use the Activation Scripts

For convenience, we provide scripts to handle environment activation and PATH issues:

```bash
# Activate Chemprop environment with proper PATH setup
source ./activate_chemprop.sh

# Or use the run script directly (bypasses PATH issues)
./run_training.sh dry   # Dry run
./run_training.sh test  # Test with 1 epoch
./run_training.sh full  # Full training
```

**Note**: On macOS with both pyenv and conda installed, the activation script ensures the conda Python takes precedence.

### Part C: Data Preparation

#### 1. Download Data from HuggingFace

The project includes scripts to automatically download data from the OpenADMET HuggingFace repository.

##### Option A: Download and save all datasets at once

```bash
cd Data
python save_data.py
cd ..
```

This will:
- Download train, test, and teaser datasets from HuggingFace
- Save them as CSV files in the Data directory
- Display dataset shapes and basic information

##### Option B: Load datasets programmatically

```python
from Data.load_data import load_train, load_test, load_teaser

# Load datasets directly from HuggingFace
df_train = load_train()
df_test = load_test()
df_teaser = load_teaser()
```

#### 2. Separate Training Data by Endpoints

After downloading the data, separate the training set into individual endpoint-specific datasets:

```bash
python separate_endpoints.py
```

This script will:
- Read the training dataset (5,326 molecules)
- Create 9 separate CSV files, one for each endpoint
- Generate files in `Data/endpoints/` directory
- Create a summary report with statistics

#### Generated Endpoint Files:

| Endpoint | File | Molecules | Description |
|----------|------|-----------|-------------|
| LogD | train_LogD.csv | 5,039 | Lipophilicity at pH 7.4 |
| KSOL | train_KSOL.csv | 5,128 | Kinetic solubility |
| HLM CLint | train_HLM_CLint.csv | 3,759 | Human liver microsome clearance |
| MLM CLint | train_MLM_CLint.csv | 4,522 | Mouse liver microsome clearance |
| Caco-2 Papp A>B | train_Caco_2_Permeability_Papp_AtoB.csv | 2,157 | Intestinal permeability |
| Caco-2 Efflux | train_Caco_2_Permeability_Efflux.csv | 2,161 | Efflux ratio |
| MPPB | train_MPPB.csv | 1,302 | Mouse plasma protein binding |
| MBPB | train_MBPB.csv | 975 | Mouse brain protein binding |
| MGMB | train_MGMB.csv | 222 | Mouse gut microsome binding |

Additional files generated:
- `Data/endpoints/summary.txt` - Detailed statistics for each endpoint
- `Data/endpoints/data_distribution.csv` - Data distribution across train/test/teaser sets

## Running Models

### Option 1: XGBoost Models

Activate the XGBoost environment:
```bash
source openADMETvenv/bin/activate
```

Train individual endpoint models:
```bash
# Train LogD model
python logd_xgb.py

# Train KSOL model
python ksol_xgb.py

# Train HLM CLint model
python hlm_xgb.py

# Compare HLM models with/without LogD features
python hlm_xgb_compare.py
```

Generate parity plots:
```bash
# Generate LogD parity plot
python plot_parity_logd.py

# Generate HLM parity plot
python plot_parity_hlm.py

# Generate KSOL parity plot
python plot_ksol.py
```

### Option 2: Chemprop MPNN Models

Chemprop provides state-of-the-art Message Passing Neural Networks for molecular property prediction.

#### Quick Start with Scripts

```bash
# Use the convenient run script (handles environment automatically)
./run_training.sh dry   # Dry run to check setup
./run_training.sh test  # Test with 1 epoch
./run_training.sh full  # Full training for all endpoints
```

#### Manual Training

```bash
# Activate Chemprop environment
conda activate chemprop

# Or use the activation script for proper PATH setup
source ./activate_chemprop.sh

# Train all endpoints (except LogD if already done)
python chemprop_training/train_all_endpoints.py

# Train specific endpoints only
python chemprop_training/train_all_endpoints.py --endpoints KSOL HLM_CLint MLM_CLint

# Train with custom settings
python chemprop_training/train_all_endpoints.py --max_epochs 50 --batch_size 64
```

#### Monitor Training Progress

```bash
# View real-time training logs
tail -f chemprop_training/outputs/training_all_endpoints_*.log

# Check the latest log file
tail -f chemprop_training/outputs/$(ls -t chemprop_training/outputs/*.log | head -1)
```

#### Training Output Files

- **Checkpoints**: `checkpoints/{endpoint}/` - Saved model weights
- **Predictions**: `predictions/` - Test set predictions
- **Logs**: `chemprop_training/outputs/` - Training logs
- **Summary**: `chemprop_training/training_summary.json` - Performance metrics

## Data Exploration

### Using Jupyter Notebooks

```bash
jupyter notebook Notebooks/TeaserDataExploration.ipynb
```

### Using Python Scripts

```bash
python dataExploration.py
```

## Endpoints Description

The dataset contains 9 ADMET endpoints:

1. **LogD**: Distribution coefficient at pH 7.4 (lipophilicity)
2. **KSOL**: Kinetic solubility in μg/mL
3. **HLM CLint**: Human liver microsome intrinsic clearance (μL/min/mg)
4. **MLM CLint**: Mouse liver microsome intrinsic clearance (μL/min/mg)
5. **Caco-2 Permeability Papp A>B**: Apical to basolateral permeability (10^-6 cm/s)
6. **Caco-2 Permeability Efflux**: Efflux ratio (BA/AB)
7. **MPPB**: Mouse plasma protein binding (% bound)
8. **MBPB**: Mouse brain protein binding (% bound)
9. **MGMB**: Mouse gut microsome binding (% bound)

## Model Features

### XGBoost Models
- **Morgan Fingerprints**: 2048-bit, radius=2, with chirality
- **Molecular Descriptors**: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, etc.
- **Scaffold-based splitting**: For better generalization
- **Hyperparameter optimization**: Using RandomizedSearchCV

### Chemprop MPNN Models
- **Message Passing Neural Networks**: Graph-based deep learning
- **Automatic feature learning**: Learns molecular representations end-to-end
- **Architecture**:
  - Hidden size: 300
  - Depth: 3 message passing steps
  - FFN layers: 2
  - Batch normalization and dropout for regularization
- **Training**:
  - Max epochs: 100 with early stopping (patience=20)
  - Batch size: 32
  - Learning rate: 0.001 with gradient clipping
  - Scaffold-balanced splitting

## Model Performance

Example performance on teaser set:

| Model | Endpoint | MAE | RMSE | R² |
|-------|----------|-----|------|-----|
| Chemprop | LogD | 0.152 | 0.206 | 0.969 |
| XGBoost | LogD | 0.18 | 0.24 | 0.96 |
| Chemprop | KSOL | TBD | TBD | TBD |
| XGBoost | KSOL | 28.5 | 41.2 | 0.89 |

*Note: Full results will be available after complete training*

## Troubleshooting

### SyntaxWarnings in chemprop
The chemprop library SyntaxWarnings have been fixed for Python 3.13 compatibility. No action required.

### Missing dependencies
If you encounter import errors, ensure you've activated the virtual environment:
```bash
source openADMETvenv/bin/activate
```

### Data download issues
If automatic download fails, you can manually download from:
- Base URL: `hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/`

## Contributing

Please submit issues and pull requests through GitHub.

## License

[Add license information here]

## Citation

If you use this code or data, please cite:
[Add citation information here]