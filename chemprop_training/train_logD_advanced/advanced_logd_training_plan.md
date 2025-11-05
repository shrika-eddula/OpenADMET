# Advanced LogD Training Plan

## Overview
Comprehensive training strategy for LogD prediction using Chemprop with advanced techniques to maximize performance on the expanded dataset.

## Dataset
- **Training Data**: `Data/endpoints/train_big_LogD.csv` (combined OpenADMET + PharmaBench)
- **Test Data**: `Data/test.csv`
- **Validation**: Scaffold-based K-fold cross-validation

## Architecture & Hyperparameters

### Base Model: D-MPNN (Directed Message Passing Neural Network)
- **Depth**: 6 (deeper than default 3)
- **Hidden Size**: 600 (wider than default 300)
- **FFN Layers**: 3
- **FFN Hidden Size**: 600
- **Dropout**: 0.10 (mild regularization)
- **Batch Size**: 64
- **Aggregation**: mean

### Training Parameters
- **Epochs**: 120
- **Warmup Epochs**: 5
- **Loss Function**: Huber (robust to outliers)
- **Optimizer**: AdamW with weight decay

### Learning Rate Schedule
- **1-Cycle Policy**:
  - Initial LR: 1.5e-4
  - Max LR: 2.5e-3
  - Final LR: 1e-5

## Feature Engineering

### 1. Molecular Representations
```bash
--features_generator rdkit_2d_normalized,morgan_count
```
- **RDKit 2D Descriptors**: Normalized physicochemical properties
  - MolWeight, LogP, TPSA, NumHDonors, NumHAcceptors, etc.
- **Morgan Fingerprint Counts**: ECFP feature counts (not bits)

### 2. Additional Features (Optional)
Create `features_LogD.csv` with:
- Calculated pKa (strongest acidic and basic)
- cLogP (calculated LogP)
- Molecular complexity metrics
- Aromatic ring count
- Fraction sp3 carbons

## Training Strategy

### 1. Scaffold K-Fold Cross-Validation
- **K = 5 folds**
- Ensures chemical diversity in validation
- Better generalization estimates

### 2. Multi-Seed Ensemble
- **Seeds**: [42, 123, 456, 789, 2024]
- 5 models per fold = 25 total models
- Average predictions for final output

### 3. Stochastic Weight Averaging (SWA)
- **Start**: Epoch 90 (75% of training)
- **SWA LR**: 1e-4
- **Update Frequency**: Every batch
- Improves generalization

### 4. Advanced Techniques

#### Outlier Handling
- **Huber Loss**: δ = 1.0 (reduces impact of outliers)
- **Winsorization**: Clip extreme values at 1st/99th percentiles
- **Data Cleaning**: Remove chemically unreasonable structures

#### Uncertainty Quantification
- **MC-Dropout**:
  - Keep dropout active during inference
  - 20 forward passes
  - Calculate mean and std for uncertainty
- **Deep Ensembles**: Natural from multi-seed training
- **Confidence Intervals**: 95% CI from ensemble predictions

#### Calibration
- **Isotonic Regression**: Post-hoc calibration on validation set
- **Temperature Scaling**: Alternative calibration method
- **Calibration Plots**: Reliability diagrams

## Implementation Components

### 1. Main Training Script
`train_logd_advanced.py`
- Orchestrates entire pipeline
- Handles K-fold CV
- Manages multi-seed training
- Combines predictions

### 2. Feature Generation
`generate_features_logd.py`
- Computes RDKit descriptors
- Calculates Morgan counts
- Generates additional features CSV

### 3. Model Architecture
`models/advanced_mpnn.py`
- Extended D-MPNN with SWA
- MC-Dropout implementation
- Feature concatenation layer

### 4. Evaluation & Metrics
`evaluate_logd.py`
- Comprehensive metrics (MAE, RMSE, R², Spearman)
- Uncertainty metrics (calibration error, sharpness)
- Chemical space analysis
- Error analysis by molecular properties

### 5. Prediction Pipeline
`predict_logd_advanced.py`
- Ensemble predictions
- Uncertainty estimates
- Calibrated outputs
- Confidence intervals

## Evaluation Metrics

### Primary Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **Spearman ρ**: Rank correlation

### Uncertainty Metrics
- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **Sharpness**: Average prediction interval width
- **Coverage**: % of true values in 95% CI

### Chemical Space Analysis
- Performance by molecular weight bins
- Performance by LogD ranges
- Scaffold diversity metrics
- Outlier analysis

## Output Structure

```
logd_advanced_results/
├── models/
│   ├── fold_0/
│   │   ├── seed_42/
│   │   │   ├── checkpoint.pt
│   │   │   ├── swa_model.pt
│   │   │   └── metrics.json
│   │   └── ...
│   └── ...
├── predictions/
│   ├── train_cv_predictions.csv
│   ├── test_predictions.csv
│   ├── test_predictions_calibrated.csv
│   └── test_predictions_with_ci.csv
├── evaluation/
│   ├── metrics_summary.json
│   ├── calibration_plots.png
│   ├── parity_plots.png
│   ├── error_analysis.csv
│   └── chemical_space_performance.csv
└── logs/
    └── training.log
```

## Expected Performance Improvements

### Baseline (Current)
- MAE: ~0.15
- RMSE: ~0.21
- R²: ~0.97

### Target (With Advanced Training)
- MAE: <0.12
- RMSE: <0.18
- R²: >0.98
- Well-calibrated uncertainty estimates
- Robust to outliers

## Implementation Priority

1. **Phase 1**: Core improvements
   - Larger dataset (train_big_LogD.csv)
   - Deeper/wider architecture
   - Feature augmentation

2. **Phase 2**: Ensemble & CV
   - K-fold cross-validation
   - Multi-seed ensemble
   - Basic ensemble predictions

3. **Phase 3**: Advanced techniques
   - SWA implementation
   - Huber loss
   - MC-Dropout uncertainty

4. **Phase 4**: Calibration & Analysis
   - Isotonic calibration
   - Comprehensive evaluation
   - Error analysis

## Command Examples

### Training with all features:
```bash
python train_logd_advanced.py \
    --data_path Data/endpoints/train_big_LogD.csv \
    --test_path Data/test.csv \
    --features_generator rdkit_2d_normalized,morgan_count \
    --features_path features_LogD.csv \
    --depth 6 \
    --hidden_size 600 \
    --ffn_num_layers 3 \
    --ffn_hidden_size 600 \
    --dropout 0.10 \
    --batch_size 64 \
    --epochs 120 \
    --warmup_epochs 5 \
    --loss_function huber \
    --init_lr 1.5e-4 \
    --max_lr 2.5e-3 \
    --final_lr 1e-5 \
    --ensemble_size 5 \
    --num_folds 5 \
    --use_swa \
    --swa_start_epoch 90 \
    --use_mc_dropout \
    --calibrate \
    --save_dir logd_advanced_results
```

### Quick test (reduced settings):
```bash
python train_logd_advanced.py \
    --data_path Data/endpoints/train_big_LogD.csv \
    --test_path Data/test.csv \
    --features_generator rdkit_2d_normalized \
    --depth 4 \
    --hidden_size 400 \
    --epochs 10 \
    --num_folds 2 \
    --ensemble_size 2 \
    --save_dir test_results
```

## Notes & Considerations

1. **Computational Requirements**:
   - Full pipeline: ~24-48 hours on GPU
   - Memory: >16GB RAM recommended
   - Storage: ~5GB for all models and results

2. **Chemprop Version**:
   - Ensure Chemprop v2.0+ for all features
   - Some features may need custom implementation

3. **Data Quality**:
   - Check for duplicates and conflicts
   - Validate SMILES strings
   - Consider experimental uncertainty in labels

4. **Reproducibility**:
   - Fix all random seeds
   - Log all hyperparameters
   - Version control code and data