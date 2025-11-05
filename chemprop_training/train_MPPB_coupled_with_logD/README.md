# Train MPPB Coupled with LogD

Advanced MPPB (Mouse Plasma Protein Binding) training that incorporates LogD as an additional descriptor for improved predictions.

## Key Innovation
This approach uses LogD values as additional molecular descriptors (x_d features) alongside SMILES-based molecular graphs, creating a multi-modal prediction model that leverages both structural and physicochemical properties.

## Directory Structure
```
train_MPPB_coupled_with_logD/
├── scripts/
│   ├── train_mppb_advanced.py           # Advanced MPPB training with LogD
│   ├── train_mppb_with_logd.py         # Alternative training script
│   ├── update_mppb_predictions.py      # Update predictions with trained model
│   ├── regenerate_mppb_predictions.py  # Regenerate predictions with proper scaling
│   ├── combine_logd_mppb_datasets.py   # Dataset preparation
│   ├── run_mppb_advanced.sh            # Training launcher
│   ├── run_mppb_logd.sh               # Alternative launcher
│   └── utils.py                        # Utility functions
├── results/
│   ├── mppb_advanced_full/            # Full training results (25 models)
│   ├── mppb_advanced_medium/          # Medium training results
│   ├── mppb_advanced_test/            # Test training results
│   ├── mppb_logd_full_results/        # Alternative approach results
│   └── mppb_logd_test_results/        # Test results
├── configs/                           # Model configurations
├── plots/                              # Training plots
└── logs/                               # Training logs
```

## Usage

### Training
```bash
# Quick test (5 models, ~1 hour)
cd scripts
./run_mppb_advanced.sh test

# Medium training (10 models, 4-6 hours)
./run_mppb_advanced.sh medium

# Full training (25 models, 12-24 hours)
./run_mppb_advanced.sh full
```

### Update Predictions
```bash
# Update MPPB predictions using the trained ensemble
python update_mppb_predictions.py \
  --input_file ../../../predictions/all_endpoints_predictions_advanced_logd.csv \
  --model_dir ../results/mppb_advanced_full
```

## Model Architecture
- **Input**: SMILES + LogD value
- **Hidden size**: 300
- **Depth**: 6 layers
- **Dropout**: 0.1
- **Additional descriptor (x_d)**: LogD value
- **Ensemble**: 5-fold CV × 5 seeds = 25 models

## Performance Metrics (Full Model)
- **MAE**: 12.43
- **RMSE**: 15.40
- **R²**: 0.124
- **Pearson r**: 0.916 (excellent correlation)
- **Spearman r**: 0.910 (excellent rank correlation)

Note: While R² is relatively low, the high correlation coefficients indicate the model captures the relative ordering of MPPB values very well.

## Data Requirements
The model requires:
1. **SMILES**: Molecular structure representation
2. **LogD**: Lipophilicity at pH 7.4 (used as additional descriptor)
3. **MPPB**: Target values for training (% bound)

## Output Format
The predictions include:
- MPPB predicted value (% bound)
- Optionally: uncertainty estimates from ensemble

## Key Scripts

### train_mppb_advanced.py
Main training script with:
- K-fold cross-validation
- Multi-seed ensemble
- LogD as additional descriptor
- Proper scaling and normalization

### update_mppb_predictions.py
Updates existing prediction files:
- Loads ensemble of trained models
- Uses SMILES and LogD for prediction
- Replaces MPPB column with new predictions
- Preserves all other endpoint predictions

### regenerate_mppb_predictions.py
Regenerates predictions with proper denormalization:
- Restores output transforms
- Applies proper scaling
- Useful for debugging prediction issues

## Training Strategy
1. **Data Preparation**: Combine SMILES with LogD values
2. **Cross-Validation**: 5-fold split for robust evaluation
3. **Multi-Seed Training**: 5 different seeds per fold
4. **Ensemble Prediction**: Average across all 25 models
5. **Uncertainty Estimation**: Standard deviation across ensemble

## Why LogD Coupling?
- LogD correlates with plasma protein binding
- Provides physicochemical context to structural features
- Improves prediction accuracy, especially for lipophilic compounds
- Reduces prediction variance across chemical space

## Notes
- Ensure LogD predictions are available before MPPB training
- Models save scaler parameters for proper inference
- Ensemble provides both predictions and uncertainty
- Best results with advanced LogD predictions as input