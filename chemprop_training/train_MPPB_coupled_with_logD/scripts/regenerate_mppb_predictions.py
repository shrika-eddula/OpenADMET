#!/usr/bin/env python3
"""
Regenerate MPPB predictions with proper denormalization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

from chemprop import data, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset

def main():
    # Load ensemble info
    results_dir = Path('mppb_advanced_test')

    with open(results_dir / 'ensemble_info.json', 'r') as f:
        ensemble_info = json.load(f)

    print(f"Found {len(ensemble_info['models'])} models in ensemble")

    # Load test data
    test_df = pd.read_csv('../Data/paired_endpoints/train_LogD_MPPB.csv')
    print(f"Loaded {len(test_df)} test molecules")

    # Create test datapoints with LogD
    test_datapoints = []
    mean_logd = test_df['LogD'].mean()

    for _, row in test_df.iterrows():
        x_d = np.array([row['LogD']], dtype=np.float32) if not pd.isna(row['LogD']) else np.array([mean_logd], dtype=np.float32)
        dp = MoleculeDatapoint.from_smi(row['SMILES'], None, x_d=x_d)
        test_datapoints.append(dp)

    test_dset = MoleculeDataset(test_datapoints)
    test_loader = data.build_dataloader(test_dset, batch_size=32, num_workers=0, shuffle=False)

    all_predictions = []

    # Get predictions from each model
    for model_info in tqdm(ensemble_info['models'], desc="Loading models"):
        # Load model
        model = models.MPNN.load_from_checkpoint(model_info['best_checkpoint'], strict=False)

        # CRITICAL: Restore the output transform
        scaler = StandardScaler()
        scaler.mean_ = np.array([model_info['scaler_mean']])
        scaler.scale_ = np.array([model_info['scaler_std']])
        model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        # Also restore X_d transform if available
        if model_info.get('x_d_scaler_mean') is not None:
            x_d_scaler = StandardScaler()
            x_d_scaler.mean_ = np.array([model_info['x_d_scaler_mean']])
            x_d_scaler.scale_ = np.array([model_info['x_d_scaler_std']])
            model.X_d_transform = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

        # Get predictions
        import lightning.pytorch as pl
        trainer = pl.Trainer(accelerator="auto", devices=1, enable_progress_bar=False, logger=False)
        predictions = trainer.predict(model, test_loader)

        # Concatenate predictions
        preds = []
        for batch_preds in predictions:
            preds.extend(batch_preds.cpu().numpy())

        all_predictions.append(np.array(preds).flatten())

    # Calculate ensemble statistics
    all_predictions = np.array(all_predictions)
    ensemble_mean = np.mean(all_predictions, axis=0)
    ensemble_std = np.std(all_predictions, axis=0)

    print(f"\nPrediction statistics:")
    print(f"  Mean: {ensemble_mean.mean():.2f}")
    print(f"  Std: {ensemble_mean.std():.2f}")
    print(f"  Range: [{ensemble_mean.min():.2f}, {ensemble_mean.max():.2f}]")

    # Evaluate
    y_true = test_df['MPPB'].values
    mae = mean_absolute_error(y_true, ensemble_mean)
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_mean))
    r2 = r2_score(y_true, ensemble_mean)
    pearson_r, _ = pearsonr(y_true, ensemble_mean)
    spearman_r, _ = spearmanr(y_true, ensemble_mean)

    print(f"\n{'='*60}")
    print("CORRECTED EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman r: {spearman_r:.4f}")

    # Save corrected predictions
    output_df = test_df[['Molecule Name', 'SMILES', 'LogD']].copy()
    output_df['MPPB_true'] = test_df['MPPB']
    output_df['MPPB_pred'] = ensemble_mean
    output_df['MPPB_std'] = ensemble_std

    output_df.to_csv(results_dir / 'predictions' / 'test_predictions_corrected.csv', index=False)
    print(f"\nCorrected predictions saved to {results_dir}/predictions/test_predictions_corrected.csv")

if __name__ == "__main__":
    main()