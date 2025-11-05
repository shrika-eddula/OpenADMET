#!/usr/bin/env python3
"""
Generate predictions for all 9 ADMET endpoints on test set molecules.
Combines all predictions into a single CSV file with the same format as train.csv.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from lightning import pytorch as pl
from tqdm import tqdm

# Add chemprop to path if needed
sys.path.insert(0, str(Path.cwd()))

# Import Chemprop modules
from chemprop import data, featurizers, models, nn

# Define endpoints
ENDPOINTS = [
    'LogD',
    'KSOL',
    'HLM_CLint',
    'MLM_CLint',
    'Caco_2_Permeability_Papp_AtoB',
    'Caco_2_Permeability_Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]

# Map endpoint names to column names in CSV
ENDPOINT_TO_COLUMN = {
    'LogD': 'LogD',
    'KSOL': 'KSOL',
    'HLM_CLint': 'HLM CLint',
    'MLM_CLint': 'MLM CLint',
    'Caco_2_Permeability_Papp_AtoB': 'Caco-2 Permeability Papp A>B',
    'Caco_2_Permeability_Efflux': 'Caco-2 Permeability Efflux',
    'MPPB': 'MPPB',
    'MBPB': 'MBPB',
    'MGMB': 'MGMB'
}

def find_best_checkpoint(endpoint):
    """Find the best checkpoint for an endpoint."""
    checkpoint_dir = Path(f"checkpoints/{endpoint}")
    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None

    # Sort by modification time and get the most recent
    # Or look for "best" or "last" in filename
    best_ckpt = None
    for ckpt in ckpt_files:
        if "last" in ckpt.name.lower():
            return str(ckpt)
        if best_ckpt is None or ckpt.stat().st_mtime > Path(best_ckpt).stat().st_mtime:
            best_ckpt = str(ckpt)

    return best_ckpt

def load_scaler_params(endpoint):
    """Load scaler parameters from model info."""
    model_info_path = Path(f"checkpoints/{endpoint}/model_info.json")
    if model_info_path.exists():
        with open(model_info_path, 'r') as f:
            info = json.load(f)
            if 'scaler_params' in info:
                return info['scaler_params']
    return None

def predict_endpoint(endpoint, test_df, batch_size=32):
    """Generate predictions for a single endpoint."""
    print(f"\nProcessing {endpoint}...")

    # Find checkpoint
    checkpoint_path = find_best_checkpoint(endpoint)
    if not checkpoint_path:
        print(f"  ⚠ No checkpoint found for {endpoint}")
        return None

    print(f"  Using checkpoint: {checkpoint_path}")

    # Load model with strict=False to handle minor mismatches
    try:
        model = models.MPNN.load_from_checkpoint(checkpoint_path, strict=False)
        model.eval()
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None

    # Create featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # Process test molecules
    test_smiles = test_df['SMILES'].values
    test_datapoints = []
    valid_indices = []

    for i, smi in enumerate(test_smiles):
        try:
            dp = data.MoleculeDatapoint.from_smi(smi)
            test_datapoints.append(dp)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Warning: Error processing SMILES at index {i}: {e}")
            continue

    if not test_datapoints:
        print(f"  ❌ No valid molecules for {endpoint}")
        return None

    print(f"  Processing {len(test_datapoints)} molecules")

    # Create dataset and dataloader
    test_dataset = data.MoleculeDataset(test_datapoints, featurizer=featurizer)
    test_dataloader = data.build_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    # Use Lightning Trainer for predictions
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        logger=False
    )

    # Make predictions using trainer
    predictions = trainer.predict(model, test_dataloader)

    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0).flatten()

    # Create result array with NaN for invalid molecules
    result = np.full(len(test_df), np.nan)
    result[valid_indices] = predictions

    print(f"  ✓ Generated {len(predictions)} predictions")
    print(f"  Range: [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]")

    return result

def main():
    """Main function to generate all predictions."""
    print("="*60)
    print("Generating Predictions for All ADMET Endpoints")
    print("="*60)

    # Load test data
    test_file = Path("Data/test.csv")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    test_df = pd.read_csv(test_file)
    print(f"\nLoaded {len(test_df)} test molecules")

    # Initialize results dataframe with molecule info
    results_df = test_df[['Molecule Name', 'SMILES']].copy()

    # Process each endpoint
    successful_endpoints = []
    failed_endpoints = []

    for endpoint in ENDPOINTS:
        column_name = ENDPOINT_TO_COLUMN[endpoint]

        # Check if we should skip this endpoint
        checkpoint_exists = find_best_checkpoint(endpoint) is not None
        if not checkpoint_exists:
            print(f"\n⏩ Skipping {endpoint} (no trained model)")
            failed_endpoints.append(endpoint)
            results_df[column_name] = np.nan
            continue

        # Generate predictions
        predictions = predict_endpoint(endpoint, test_df)

        if predictions is not None:
            results_df[column_name] = predictions
            successful_endpoints.append(endpoint)
        else:
            results_df[column_name] = np.nan
            failed_endpoints.append(endpoint)

    # Save results
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"all_endpoints_predictions_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)

    # Also save as latest for easy access
    latest_file = output_dir / "all_endpoints_predictions_latest.csv"
    results_df.to_csv(latest_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total molecules: {len(results_df)}")
    print(f"Successful endpoints ({len(successful_endpoints)}): {', '.join(successful_endpoints)}")
    if failed_endpoints:
        print(f"Failed/Skipped endpoints ({len(failed_endpoints)}): {', '.join(failed_endpoints)}")

    print(f"\n✓ Predictions saved to:")
    print(f"  {output_file}")
    print(f"  {latest_file}")

    # Show sample of results
    print("\nSample of predictions (first 5 molecules):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.head())

    # Show completeness statistics
    print("\nCompleteness by endpoint:")
    for endpoint in ENDPOINTS:
        column_name = ENDPOINT_TO_COLUMN[endpoint]
        if column_name in results_df.columns:
            non_nan = results_df[column_name].notna().sum()
            pct = (non_nan / len(results_df)) * 100
            print(f"  {endpoint:30} {non_nan:4}/{len(results_df)} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()