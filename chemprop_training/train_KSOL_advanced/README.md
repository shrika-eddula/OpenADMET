# KSOL Training - Production Version

## Overview
Optimized KSOL prediction model using ensemble strategy with SMILES augmentation.

## Performance
- **R²**: 0.702 (3-model ensemble)
- **MAE**: 0.292 log units
- **RMSE**: 0.468 log units
- **79.5%** within ±0.5 log units
- **94.8%** within ±1.0 log units

## Model Architecture
- **Hidden size**: 512
- **Depth**: 4 layers
- **FFN layers**: 3
- **Dropout**: 0.1
- **SMILES augmentation**: 5x
- **Ensemble**: 3 models (seeds: 42, 52, 62)

## Files
- `scripts/train_best.py` - Main training script (DO NOT MODIFY - sensitive to changes)
- `scripts/predict_ksol.py` - Prediction script
- `configs/` - Configuration files
- `production_config.json` - Production configuration

## Trained Models
Models have been moved to: `/Users/khiemnguyen/Desktop/Pedal/OpenADMET/checkpoints/KSOL_advanced_best/`

## Usage

### Training
```bash
cd train_KSOL_advanced
python scripts/train_best.py
```

### Prediction
```bash
python scripts/predict_ksol.py --input <smiles_file> --output <predictions_file>
```

## Important Notes
⚠️ **WARNING**: The model is extremely sensitive to code changes. ANY modification to `train_best.py` causes catastrophic failure (R² drops to negative values) due to UnscaleTransform serialization issues.

⚠️ **HPO Note**: Hyperparameter optimization was attempted but did NOT improve results beyond the manual configuration.

## Data
- Training data: `../../Data/endpoints/train_KSOL.csv`
- Log10 transformation applied to handle wide dynamic range
- 80/20 train/test split with 15% validation from training set

Generated: 2024-11-10
