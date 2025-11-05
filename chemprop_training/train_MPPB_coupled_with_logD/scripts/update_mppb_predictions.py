#!/usr/bin/env python3
"""
Update MPPB predictions in the all endpoints file using the trained advanced model.
The model requires both SMILES and LogD values as input.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
import os
import sys
sys.path.append('/Users/khiemnguyen/Desktop/Pedal/OpenADMET')

from chemprop import data, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from sklearn.preprocessing import StandardScaler
import lightning.pytorch as pl
from pathlib import Path
from tqdm import tqdm

def load_ensemble_models(ensemble_info_path):
    """Load all models from the ensemble."""
    with open(ensemble_info_path, 'r') as f:
        ensemble_info = json.load(f)

    models_list = []
    for model_info in ensemble_info['models']:
        checkpoint_path = model_info['best_checkpoint']
        if os.path.exists(checkpoint_path):
            models_list.append({
                'path': checkpoint_path,
                'scaler_mean': model_info['scaler_mean'],
                'scaler_std': model_info['scaler_std'],
                'x_d_scaler_mean': model_info.get('x_d_scaler_mean'),
                'x_d_scaler_std': model_info.get('x_d_scaler_std')
            })

    return models_list

def predict_with_ensemble(smiles_list, logd_values, models_info):
    """Make predictions using the ensemble of models."""
    all_predictions = []

    # Create test datapoints with LogD
    test_datapoints = []
    mean_logd = np.mean([l for l in logd_values if not pd.isna(l)])

    for smiles, logd in zip(smiles_list, logd_values):
        x_d = np.array([logd], dtype=np.float32) if not pd.isna(logd) else np.array([mean_logd], dtype=np.float32)
        dp = MoleculeDatapoint.from_smi(smiles, None, x_d=x_d)
        test_datapoints.append(dp)

    test_dset = MoleculeDataset(test_datapoints)
    test_loader = data.build_dataloader(test_dset, batch_size=32, num_workers=0, shuffle=False)

    for model_info in tqdm(models_info, desc="Loading models"):
        # Load model with strict=False to handle state dict mismatches
        model = models.MPNN.load_from_checkpoint(model_info['path'], strict=False)

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

        # Get predictions using Lightning trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            logger=False
        )
        predictions = trainer.predict(model, test_loader)

        # Concatenate predictions
        preds = []
        for batch_preds in predictions:
            preds.extend(batch_preds.cpu().numpy())

        all_predictions.append(np.array(preds).flatten())

    # Calculate ensemble statistics
    all_predictions = np.array(all_predictions)
    mean_predictions = np.mean(all_predictions, axis=0)

    return mean_predictions

def main():
    parser = argparse.ArgumentParser(description='Update MPPB predictions with advanced model results')
    parser.add_argument('--input_file', type=str,
                        default='../predictions/all_endpoints_predictions_advanced_logd_20251030_230827.csv',
                        help='Input predictions file with all endpoints including LogD')
    parser.add_argument('--model_dir', type=str,
                        default='mppb_advanced_full',
                        help='Directory containing the trained MPPB model')
    parser.add_argument('--output_file', type=str,
                        default=None,
                        help='Output file path (if not specified, creates new with timestamp)')

    args = parser.parse_args()

    # Load the original predictions
    print(f"Loading original predictions from: {args.input_file}")
    original_df = pd.read_csv(args.input_file)
    print(f"  Shape: {original_df.shape}")
    print(f"  Columns: {list(original_df.columns)}")

    # Check required columns
    if 'SMILES' not in original_df.columns:
        raise ValueError("SMILES column not found in input file")
    if 'LogD' not in original_df.columns:
        raise ValueError("LogD column not found in input file")

    # Create a copy of original dataframe
    updated_df = original_df.copy()

    # Check if MPPB column exists
    if 'MPPB' not in updated_df.columns:
        print("Warning: MPPB column not found in original file, creating new column")
        updated_df['MPPB'] = None

    # Store original MPPB values for comparison
    original_mppb = updated_df['MPPB'].copy()

    # Filter out rows with missing SMILES or LogD
    valid_mask = updated_df['SMILES'].notna() & updated_df['LogD'].notna()
    valid_indices = updated_df[valid_mask].index

    if len(valid_indices) == 0:
        raise ValueError("No valid rows with both SMILES and LogD values found")

    print(f"\nFound {len(valid_indices)} valid molecules with both SMILES and LogD values")

    # Extract SMILES and LogD for valid molecules
    smiles_list = updated_df.loc[valid_indices, 'SMILES'].tolist()
    logd_values = updated_df.loc[valid_indices, 'LogD'].tolist()

    # Load ensemble models
    ensemble_info_path = os.path.join(args.model_dir, 'ensemble_info.json')
    print(f"\nLoading ensemble models from: {ensemble_info_path}")
    models_info = load_ensemble_models(ensemble_info_path)
    print(f"  Found {len(models_info)} models in ensemble")

    # Make predictions
    print("\nGenerating MPPB predictions...")
    mean_preds = predict_with_ensemble(smiles_list, logd_values, models_info)

    # Update the dataframe with predictions
    updated_df.loc[valid_indices, 'MPPB'] = mean_preds

    # Generate output filename if not specified
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../predictions/all_endpoints_predictions_advanced_mppb_{timestamp}.csv"
    else:
        output_file = args.output_file

    # Save the updated predictions
    print(f"\nSaving updated predictions to: {output_file}")
    updated_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("UPDATE SUMMARY")
    print("="*60)

    print(f"\nUpdated MPPB predictions for {len(valid_indices)} molecules")
    print(f"MPPB statistics:")
    print(f"  Mean:   {updated_df.loc[valid_indices, 'MPPB'].mean():.2f}")
    print(f"  Median: {updated_df.loc[valid_indices, 'MPPB'].median():.2f}")
    print(f"  Std:    {updated_df.loc[valid_indices, 'MPPB'].std():.2f}")
    print(f"  Min:    {updated_df.loc[valid_indices, 'MPPB'].min():.2f}")
    print(f"  Max:    {updated_df.loc[valid_indices, 'MPPB'].max():.2f}")

    # Compare old vs new MPPB values if they existed
    if original_mppb.notna().any():
        valid_comparison = updated_df[original_mppb.notna() & valid_mask]
        if len(valid_comparison) > 0:
            print("\nMPPB Value Changes (for molecules with previous predictions):")
            print(f"  Original mean: {original_mppb[original_mppb.notna()].mean():.2f}")
            print(f"  Updated mean:  {updated_df.loc[valid_indices, 'MPPB'].mean():.2f}")

            # Show some examples of changes
            print("\n  Sample changes (first 10):")
            for i in range(min(10, len(valid_comparison))):
                idx = valid_comparison.index[i]
                mol_name = updated_df.loc[idx, 'Molecule Name']
                old_val = original_mppb.loc[idx]
                new_val = updated_df.loc[idx, 'MPPB']
                logd_val = updated_df.loc[idx, 'LogD']
                if pd.notna(old_val) and pd.notna(new_val):
                    print(f"    {mol_name} (LogD={logd_val:.2f}): {old_val:.2f} -> {new_val:.2f} (Δ = {new_val-old_val:+.2f})")

    print(f"\nFinal dataset shape: {updated_df.shape}")
    print(f"Output saved to: {output_file}")

    # Also save a "latest" version for easy access
    latest_file = "../predictions/all_endpoints_predictions_latest.csv"
    updated_df.to_csv(latest_file, index=False)
    print(f"Also saved as: {latest_file}")

if __name__ == "__main__":
    main()