# Train All Endpoints

This directory contains scripts and utilities for training all 9 ADMET endpoints using standard Chemprop models.

## Endpoints Covered
- LogD (Lipophilicity at pH 7.4)
- KSOL (Kinetic Solubility)
- HLM_CLint (Human Liver Microsome Clearance)
- MLM_CLint (Mouse Liver Microsome Clearance)
- Caco-2 Permeability Papp A>B
- Caco-2 Permeability Efflux
- MPPB (Mouse Plasma Protein Binding)
- MBPB (Mouse Brain Protein Binding)
- MGMB (Mouse Gut Microbiome Binding)

## Directory Structure
```
train_all_endpoints/
├── scripts/
│   ├── train_all_endpoints.py     # Main training script
│   ├── generate_all_predictions.py # Generate predictions
│   ├── run_training.sh            # Training launcher script
│   ├── run_predictions.sh         # Predictions launcher
│   └── utils.py                   # Utility functions
├── configs/
│   └── config.json                # Model configurations
├── results/
│   └── training_summary.json      # Training results summary
└── logs/
    └── outputs/                   # Training logs
```

## Usage

### Training
```bash
# Quick test (1 epoch)
cd scripts
./run_training.sh test

# Full training (100 epochs)
./run_training.sh full
```

### Generating Predictions
```bash
cd scripts
./run_predictions.sh
```

### Custom Training
```bash
python train_all_endpoints.py \
  --endpoints LogD KSOL MPPB \
  --max_epochs 50 \
  --batch_size 32
```

## Configuration
Model parameters are defined in `configs/config.json`:
- Hidden size: 300
- Depth: 3 layers
- Dropout: 0.0
- Learning rate: 1e-4
- Batch size: 32
- Max epochs: 100

## Output
- Model checkpoints: `../../checkpoints/{endpoint}/`
- Predictions: `../../predictions/`
- Training logs: `logs/outputs/`

## Expected Performance
| Endpoint | Typical MAE | Typical R² |
|----------|------------|------------|
| LogD | ~0.15 | ~0.97 |
| KSOL | ~0.20 | ~0.85 |
| HLM_CLint | ~8.5 | ~0.75 |
| MLM_CLint | ~15.0 | ~0.70 |
| Caco-2 Papp | ~0.30 | ~0.80 |
| Caco-2 Efflux | ~0.25 | ~0.75 |
| MPPB | ~15.0 | ~0.10 |
| MBPB | ~8.0 | ~0.15 |
| MGMB | ~10.0 | ~0.20 |