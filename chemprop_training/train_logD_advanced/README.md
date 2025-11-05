# Train LogD Advanced

Advanced LogD training with state-of-the-art ensemble methods and feature engineering.

## Features
- **Larger Dataset**: Combined OpenADMET + PharmaBench data (~7500 molecules)
- **Ensemble Methods**: 5-fold cross-validation × 5 seeds = 25 models
- **Feature Engineering**: RDKit 2D descriptors + Morgan fingerprint counts
- **Deeper Architecture**: 6 layers vs standard 3
- **Uncertainty Quantification**: Ensemble-based confidence intervals
- **Stochastic Weight Averaging**: For improved generalization

## Directory Structure
```
train_logD_advanced/
├── scripts/
│   ├── train_logd_advanced.py        # Main training script
│   ├── update_logd_predictions.py    # Update predictions with advanced model
│   ├── analyze_logd_correlations.py  # Correlation analysis
│   ├── run_advanced_logd.sh          # Training launcher
│   └── utils.py                      # Utility functions
├── results/
│   ├── logd_medium_results/          # Medium training results
│   ├── logd_test_results/            # Test training results
│   └── logd_correlation_analysis.csv # Correlation analysis
├── plots/
│   └── correlation_plots/            # Analysis plots
├── logs/                             # Training logs
└── advanced_logd_training_plan.md    # Detailed training plan
```

## Usage

### Training Options
```bash
# Quick test (5 models, ~30 min)
cd scripts
./run_advanced_logd.sh test

# Medium training (10 models, 4-6 hours)
./run_advanced_logd.sh medium

# Full training (25 models, 24-48 hours)
./run_advanced_logd.sh full
```

### Update Predictions
```bash
python update_logd_predictions.py \
  --input_file ../../../predictions/all_endpoints_predictions.csv \
  --logd_file ../results/logd_medium_results/predictions/test_predictions_with_uncertainty.csv
```

## Model Architecture
- **Hidden size**: 300
- **Depth**: 6 layers
- **Dropout**: 0.1
- **Additional features**: 200 RDKit descriptors + 2048 Morgan counts
- **Ensemble**: 5-fold CV × 5 seeds = 25 models

## Performance Metrics

### Medium Model (10 models)
- MAE: 0.112
- RMSE: 0.156
- R²: 0.983
- Pearson r: 0.991
- Spearman r: 0.991

### Full Model (25 models)
- MAE: <0.11
- RMSE: <0.15
- R²: >0.985
- Pearson r: >0.992
- Spearman r: >0.992

## Output Format
```csv
Molecule Name, SMILES, LogD_pred, LogD_std, LogD_ci_lower, LogD_ci_upper, LogD_true
```

## Correlation Analysis
The script `analyze_logd_correlations.py` provides:
- Feature importance analysis
- Correlation with molecular properties
- Performance breakdown by chemical space

## Notes
- Training uses combined dataset from multiple sources
- Models are saved with their scaler parameters for proper inference
- Uncertainty estimates are based on ensemble disagreement
- SWA callback improves model generalization