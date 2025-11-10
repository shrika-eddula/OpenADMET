#!/usr/bin/env python3
"""
Production inference script for KSOL prediction with uncertainty quantification.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import torch
import lightning.pytorch as pl

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from chemprop import data, featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset

warnings.filterwarnings('ignore')


class KSOLPredictor:
    """Production KSOL predictor with ensemble and TTA."""

    def __init__(self, model_dir: str):
        """Initialize predictor with trained models."""
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.models = []
        self.scalers = []
        self.config = None

        # Load configuration
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded configuration from {config_path}")

        # Load models from all folds and seeds
        self.load_models()

    def load_models(self) -> None:
        """Load all trained models from directory."""
        print("Loading models...")

        # Find all model checkpoints
        model_paths = list(self.model_dir.glob("fold_*/seed_*/best*.ckpt"))

        if not model_paths:
            # Try simpler structure
            model_paths = list(self.model_dir.glob("*.ckpt"))

        if not model_paths:
            raise FileNotFoundError(f"No model checkpoints found in {self.model_dir}")

        print(f"Found {len(model_paths)} model checkpoints")

        for model_path in model_paths:
            try:
                # Load model
                model = models.MPNN.load_from_checkpoint(str(model_path))
                self.models.append(model)
                print(f"  Loaded: {model_path.name}")
            except Exception as e:
                print(f"  Failed to load {model_path}: {e}")

        if not self.models:
            raise RuntimeError("No models could be loaded")

        print(f"Successfully loaded {len(self.models)} models")

    def randomize_smiles(self, smiles: str, n: int = 10) -> List[str]:
        """Generate randomized SMILES for TTA."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        randomized = [smiles]  # Include original
        for _ in range(n - 1):
            try:
                new_smiles = Chem.MolToSmiles(mol, doRandom=True)
                if new_smiles not in randomized:
                    randomized.append(new_smiles)
            except:
                continue

        return randomized

    def predict_single(self, smiles: str, use_tta: bool = True,
                      tta_rounds: int = 10) -> Dict:
        """
        Predict KSOL for a single SMILES.

        Args:
            smiles: SMILES string
            use_tta: Whether to use test-time augmentation
            tta_rounds: Number of TTA rounds

        Returns:
            Dictionary with predictions and uncertainty
        """
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'smiles': smiles,
                'valid': False,
                'error': 'Invalid SMILES'
            }

        # Canonicalize
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Generate SMILES variants for TTA
        if use_tta:
            smiles_variants = self.randomize_smiles(canonical_smiles, tta_rounds)
        else:
            smiles_variants = [canonical_smiles]

        # Collect predictions from all models and variants
        all_predictions = []

        for model in self.models:
            model_preds = []

            for smi_variant in smiles_variants:
                # Create datapoint
                dp = MoleculeDatapoint.from_smi(smi_variant, y=None)

                # Create dataset
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                dataset = MoleculeDataset([dp], featurizer)

                # Create dataloader
                dataloader = data.build_dataloader(dataset, batch_size=1, shuffle=False)

                # Predict
                trainer = pl.Trainer(
                    accelerator="auto",
                    devices=1,
                    logger=False,
                    enable_progress_bar=False
                )

                preds = trainer.predict(model, dataloader)
                pred_value = preds[0].cpu().numpy().item()
                model_preds.append(pred_value)

            # Average over TTA variants for this model
            all_predictions.append(np.mean(model_preds))

        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions)
        std_pred = np.std(all_predictions)

        # Convert from log10 to µM
        ksol_uM = 10 ** mean_pred
        ksol_lower = 10 ** (mean_pred - 1.96 * std_pred)
        ksol_upper = 10 ** (mean_pred + 1.96 * std_pred)

        return {
            'smiles': canonical_smiles,
            'valid': True,
            'log10_ksol': mean_pred,
            'log10_ksol_std': std_pred,
            'ksol_uM': ksol_uM,
            'ksol_uM_lower_95CI': ksol_lower,
            'ksol_uM_upper_95CI': ksol_upper,
            'n_models': len(self.models),
            'tta_rounds': len(smiles_variants)
        }

    def predict_batch(self, smiles_list: List[str], use_tta: bool = True,
                     tta_rounds: int = 10, batch_size: int = 32) -> pd.DataFrame:
        """
        Predict KSOL for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings
            use_tta: Whether to use test-time augmentation
            tta_rounds: Number of TTA rounds
            batch_size: Batch size for inference

        Returns:
            DataFrame with predictions
        """
        print(f"Predicting KSOL for {len(smiles_list)} molecules...")

        if use_tta:
            print(f"Using test-time augmentation ({tta_rounds} rounds)")
        print(f"Using ensemble of {len(self.models)} models")

        results = []
        for smiles in tqdm(smiles_list, desc="Predicting"):
            result = self.predict_single(smiles, use_tta, tta_rounds)
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add classification
        df['solubility_class'] = pd.cut(
            df['ksol_uM'],
            bins=[0, 10, 100, float('inf')],
            labels=['Low', 'Medium', 'High']
        )

        return df

    def predict_from_file(self, input_file: str, output_file: str = None,
                         use_tta: bool = True, tta_rounds: int = 10) -> pd.DataFrame:
        """
        Predict KSOL from CSV file.

        Args:
            input_file: Path to CSV with SMILES column
            output_file: Path to save predictions
            use_tta: Whether to use test-time augmentation
            tta_rounds: Number of TTA rounds

        Returns:
            DataFrame with predictions
        """
        # Load input data
        df_input = pd.read_csv(input_file)

        # Find SMILES column
        smiles_col = None
        for col in ['SMILES', 'smiles', 'Smiles']:
            if col in df_input.columns:
                smiles_col = col
                break

        if smiles_col is None:
            raise ValueError(f"No SMILES column found in {input_file}")

        # Get SMILES list
        smiles_list = df_input[smiles_col].tolist()

        # Make predictions
        df_pred = self.predict_batch(smiles_list, use_tta, tta_rounds)

        # Merge with input data
        df_result = pd.concat([df_input, df_pred], axis=1)

        # Save if output file specified
        if output_file:
            df_result.to_csv(output_file, index=False)
            print(f"\nPredictions saved to: {output_file}")

        return df_result


class KSOLAnalyzer:
    """Analyze and visualize KSOL predictions."""

    @staticmethod
    def summarize_predictions(df: pd.DataFrame) -> None:
        """Print summary statistics of predictions."""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)

        # Overall statistics
        print(f"\nTotal molecules: {len(df)}")
        print(f"Valid SMILES: {df['valid'].sum()} ({df['valid'].mean()*100:.1f}%)")

        if df['valid'].sum() > 0:
            valid_df = df[df['valid']]

            print(f"\nKSOL Statistics (µM):")
            print(f"  Median: {valid_df['ksol_uM'].median():.2f}")
            print(f"  Mean:   {valid_df['ksol_uM'].mean():.2f}")
            print(f"  Min:    {valid_df['ksol_uM'].min():.2f}")
            print(f"  Max:    {valid_df['ksol_uM'].max():.2f}")

            print(f"\nLog10(KSOL) Statistics:")
            print(f"  Median: {valid_df['log10_ksol'].median():.2f}")
            print(f"  Mean:   {valid_df['log10_ksol'].mean():.2f}")
            print(f"  Std:    {valid_df['log10_ksol'].std():.2f}")

            print(f"\nSolubility Classes:")
            if 'solubility_class' in valid_df.columns:
                class_counts = valid_df['solubility_class'].value_counts()
                for cls, count in class_counts.items():
                    print(f"  {cls}: {count} ({count/len(valid_df)*100:.1f}%)")

            print(f"\nPrediction Uncertainty:")
            print(f"  Mean uncertainty (log10): {valid_df['log10_ksol_std'].mean():.3f}")
            print(f"  Max uncertainty (log10):  {valid_df['log10_ksol_std'].max():.3f}")

    @staticmethod
    def identify_outliers(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Identify molecules with high prediction uncertainty."""
        if 'log10_ksol_std' not in df.columns:
            return pd.DataFrame()

        high_uncertainty = df[df['log10_ksol_std'] > threshold].copy()
        high_uncertainty = high_uncertainty.sort_values('log10_ksol_std', ascending=False)

        if len(high_uncertainty) > 0:
            print(f"\nMolecules with high uncertainty (std > {threshold}):")
            print(high_uncertainty[['smiles', 'ksol_uM', 'log10_ksol_std']].head(10))

        return high_uncertainty


def main():
    parser = argparse.ArgumentParser(description='KSOL Prediction')

    # Input/Output
    parser.add_argument('--input', '-i', required=True, help='Input CSV file or SMILES string')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--model-dir', '-m', default='results_advanced',
                       help='Directory containing trained models')

    # Prediction options
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable test-time augmentation')
    parser.add_argument('--tta-rounds', type=int, default=10,
                       help='Number of TTA rounds')

    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                       help='Show detailed analysis of predictions')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.5,
                       help='Threshold for high uncertainty')

    args = parser.parse_args()

    # Initialize predictor
    predictor = KSOLPredictor(args.model_dir)

    # Check if input is a file or SMILES
    if Path(args.input).exists():
        # Predict from file
        df_results = predictor.predict_from_file(
            args.input,
            args.output,
            use_tta=not args.no_tta,
            tta_rounds=args.tta_rounds
        )
    else:
        # Assume it's a SMILES string
        result = predictor.predict_single(
            args.input,
            use_tta=not args.no_tta,
            tta_rounds=args.tta_rounds
        )
        df_results = pd.DataFrame([result])

        # Print single prediction
        if result['valid']:
            print(f"\nPrediction for: {result['smiles']}")
            print(f"  KSOL: {result['ksol_uM']:.2f} µM")
            print(f"  95% CI: [{result['ksol_uM_lower_95CI']:.2f}, {result['ksol_uM_upper_95CI']:.2f}] µM")
            print(f"  Log10(KSOL): {result['log10_ksol']:.3f} ± {result['log10_ksol_std']:.3f}")
        else:
            print(f"\nError: {result['error']}")

    # Analyze results if requested
    if args.analyze and len(df_results) > 1:
        analyzer = KSOLAnalyzer()
        analyzer.summarize_predictions(df_results)
        analyzer.identify_outliers(df_results, args.uncertainty_threshold)

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()