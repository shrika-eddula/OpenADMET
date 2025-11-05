# Chemprop Training Directory

This directory contains organized training pipelines for ADMET property prediction using Message Passing Neural Networks (MPNN). The structure is organized by training approach, with each subdirectory containing all relevant scripts, configurations, and results.

## Directory Structure

```
chemprop_training/
├── train_all_endpoints/              # Standard training for all 9 ADMET endpoints
│   ├── scripts/                      # Training and prediction scripts
│   ├── configs/                      # Model configurations
│   ├── results/                      # Training results
│   ├── logs/                         # Training logs
│   └── README.md                     # Detailed documentation
│
├── train_logD_advanced/              # Advanced LogD training with ensemble methods
│   ├── scripts/                      # Training and update scripts
│   ├── results/                      # Model outputs and predictions
│   ├── plots/                        # Analysis visualizations
│   ├── logs/                         # Training logs
│   └── README.md                     # Detailed documentation
│
├── train_MPPB_coupled_with_logD/     # MPPB training using LogD as additional feature
│   ├── scripts/                      # Training and update scripts
│   ├── results/                      # Model outputs and predictions
│   ├── configs/                      # Model configurations
│   ├── logs/                         # Training logs
│   └── README.md                     # Detailed documentation
│
└── common_logs/                      # Shared logging directory
    └── lightning_logs/               # PyTorch Lightning logs
```

## Quick Start

### 1. Activate Chemprop Environment
From the root directory:
```bash
source ./activate_chemprop.sh
```

### 2. Choose Your Training Approach

#### Option A: Standard Training for All Endpoints
```bash
cd train_all_endpoints/scripts
./run_training.sh full   # Train all 9 endpoints
./run_predictions.sh     # Generate predictions
```

#### Option B: Advanced LogD Training
```bash
cd train_logD_advanced/scripts
./run_advanced_logd.sh full  # Train ensemble model
```

#### Option C: Advanced MPPB with LogD Coupling
```bash
cd train_MPPB_coupled_with_logD/scripts
./run_mppb_advanced.sh full  # Train MPPB with LogD features
```

## Complete Workflow for Best Results

For optimal predictions across all endpoints, follow this sequence:

### Step 1: Train Basic Models
```bash
cd train_all_endpoints/scripts
./run_training.sh full
./run_predictions.sh
```

### Step 2: Train and Apply Advanced LogD
```bash
cd ../../train_logD_advanced/scripts
./run_advanced_logd.sh full

# Update predictions with advanced LogD
python update_logd_predictions.py \
  --input_file ../../../predictions/all_endpoints_predictions.csv
```

### Step 3: Train and Apply Advanced MPPB
```bash
cd ../../train_MPPB_coupled_with_logD/scripts
./run_mppb_advanced.sh full

# Update predictions with advanced MPPB
python update_mppb_predictions.py \
  --input_file ../../../predictions/all_endpoints_predictions_advanced_logd_*.csv
```

### Final Output
The complete predictions file will contain:
- Advanced LogD predictions (ensemble model)
- Advanced MPPB predictions (using LogD as feature)
- Standard predictions for other 7 endpoints
Location: `../predictions/all_endpoints_predictions_latest.csv`

## Training Approaches Overview

### 1. Standard Training (train_all_endpoints)
- **Purpose**: Baseline models for all 9 ADMET endpoints
- **Method**: Standard Chemprop MPNN
- **Endpoints**: LogD, KSOL, HLM/MLM CLint, Caco-2, MPPB, MBPB, MGMB
- **Time**: ~2-4 hours for full training
- **Best for**: Quick results, baseline performance

### 2. Advanced LogD (train_logD_advanced)
- **Purpose**: State-of-the-art LogD prediction
- **Method**: 5-fold CV × 5 seeds ensemble (25 models)
- **Features**: RDKit descriptors + Morgan fingerprints
- **Performance**: MAE <0.11, R² >0.985
- **Time**: 24-48 hours for full ensemble
- **Best for**: High-accuracy LogD predictions with uncertainty

### 3. MPPB Coupled with LogD (train_MPPB_coupled_with_logD)
- **Purpose**: Improved MPPB prediction using physicochemical properties
- **Method**: Multi-modal learning with SMILES + LogD
- **Innovation**: Uses LogD as additional molecular descriptor
- **Performance**: Pearson r = 0.916, Spearman r = 0.910
- **Time**: 12-24 hours for full ensemble
- **Best for**: MPPB predictions when LogD is available

## Performance Summary

| Endpoint | Standard MAE | Advanced MAE | Improvement |
|----------|-------------|--------------|-------------|
| LogD | ~0.15 | <0.11 | 27% |
| MPPB | ~15.0 | 12.43 | 17% |
| KSOL | ~0.20 | - | - |
| HLM CLint | ~8.5 | - | - |
| MLM CLint | ~15.0 | - | - |
| Caco-2 Papp | ~0.30 | - | - |
| Caco-2 Efflux | ~0.25 | - | - |
| MBPB | ~8.0 | - | - |
| MGMB | ~10.0 | - | - |

## Key Features

### Ensemble Methods
- Multiple model training with different seeds
- K-fold cross-validation
- Uncertainty quantification through ensemble disagreement

### Feature Engineering
- RDKit 2D descriptors
- Morgan fingerprint counts
- LogD as additional descriptor for MPPB

### Advanced Architectures
- Deeper networks (6 layers vs 3)
- Stochastic Weight Averaging (SWA)
- Multi-modal input handling

## Output Files

### Model Checkpoints
- Standard models: `../checkpoints/{endpoint}/`
- Advanced models: Within respective `results/` directories

### Predictions
- Individual: `{approach}/results/*/predictions/`
- Combined: `../predictions/all_endpoints_predictions_latest.csv`

### Logs
- Training logs: `{approach}/logs/`
- Lightning logs: `common_logs/lightning_logs/`

## Requirements

- Python 3.8+
- Chemprop v2.0+
- PyTorch with Lightning
- RDKit
- scikit-learn
- pandas, numpy
- tqdm

## Tips for Best Results

1. **Data Quality**: Ensure SMILES are standardized and validated
2. **Sequential Training**: Train LogD first if planning to use MPPB coupling
3. **Resource Management**: Use GPU/MPS for faster training when available
4. **Ensemble Size**: More models generally improve predictions but increase training time
5. **Uncertainty Estimates**: Use ensemble models when confidence intervals are needed

## Troubleshooting

### Memory Issues
Reduce batch size in scripts:
```bash
python train_script.py --batch_size 16
```

### Import Errors
Ensure correct environment:
```bash
conda activate chemprop
```

### GPU/MPS Issues
Disable GPU if needed:
```bash
export CUDA_VISIBLE_DEVICES=""
```

## Citation

If using these training pipelines, please cite:
- Chemprop: [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)
- OpenADMET dataset and benchmarks

## Support

For issues or questions about specific training approaches, refer to the README.md file in each subdirectory for detailed documentation.