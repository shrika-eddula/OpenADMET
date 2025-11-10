#!/usr/bin/env python3
"""
Best training strategy for KSOL - single split with ensemble.
Train multiple models on the same 80% split for ensemble predictions.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from chemprop import data, featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings('ignore')


class BestKSOLTrainer:
    """Optimized KSOL trainer based on what actually works."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results_best"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_and_split_data(self, data_path: str, test_size: float = 0.2):
        """Load data and create a single train/val/test split."""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)

        # Rename columns
        if 'SMILES' in df.columns:
            df = df.rename(columns={'SMILES': 'smiles'})
        if 'KSOL' in df.columns:
            df = df.rename(columns={'KSOL': 'ksol_uM'})

        # Log transform
        df['log10_ksol'] = np.log10(df['ksol_uM'].clip(lower=1e-10))

        print(f"Total molecules: {len(df)}")
        print(f"Target range: {df['log10_ksol'].min():.2f} to {df['log10_ksol'].max():.2f}")

        # Single split - FIXED seed for reproducibility
        np.random.seed(42)
        test_idx = np.random.choice(len(df), int(test_size * len(df)), replace=False)
        train_val_idx = np.array([i for i in range(len(df)) if i not in test_idx])

        # Split train_val into train (85%) and val (15%)
        val_size = int(0.15 * len(train_val_idx))
        val_idx = np.random.choice(train_val_idx, val_size, replace=False)
        train_idx = np.array([i for i in train_val_idx if i not in val_idx])

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        print(f"Split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

        return train_df, val_df, test_df

    def augment_smiles(self, df: pd.DataFrame, factor: int = 5) -> pd.DataFrame:
        """Simple SMILES augmentation."""
        if factor <= 1:
            return df

        from rdkit import Chem
        augmented_rows = []

        print(f"Augmenting training data {factor}x...")
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                augmented_rows.append(row)
                continue

            # Add original
            augmented_rows.append(row)

            # Add augmented versions
            for i in range(factor - 1):
                new_row = row.copy()
                # Randomize SMILES
                new_smiles = Chem.MolToSmiles(mol, doRandom=True)
                new_row['smiles'] = new_smiles
                augmented_rows.append(new_row)

        aug_df = pd.DataFrame(augmented_rows)
        print(f"Augmented size: {len(aug_df)}")
        return aug_df

    def train_single_model(self, train_df, val_df, test_df,
                          model_idx: int = 0, augment: int = 5,
                          epochs: int = 50, hidden_size: int = 512):
        """Train a single model with given configuration."""
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx + 1}")
        print(f"{'='*60}")

        # Set seed for reproducibility
        pl.seed_everything(42 + model_idx * 10)

        # Augment training data
        if augment > 1:
            train_df = self.augment_smiles(train_df, augment)

        # Create datapoints
        train_points = [
            MoleculeDatapoint.from_smi(row['smiles'], y=[row['log10_ksol']])
            for _, row in train_df.iterrows()
        ]
        val_points = [
            MoleculeDatapoint.from_smi(row['smiles'], y=[row['log10_ksol']])
            for _, row in val_df.iterrows()
        ]
        test_points = [
            MoleculeDatapoint.from_smi(row['smiles'], y=[row['log10_ksol']])
            for _, row in test_df.iterrows()
        ]

        # Create featurizer
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Create datasets
        train_dataset = MoleculeDataset(train_points, featurizer)
        val_dataset = MoleculeDataset(val_points, featurizer)
        test_dataset = MoleculeDataset(test_points, featurizer)

        # Normalize targets
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
        test_dataset.normalize_targets(scaler)

        # Create dataloaders
        train_loader = data.build_dataloader(train_dataset, batch_size=64, shuffle=True)
        val_loader = data.build_dataloader(val_dataset, batch_size=64, shuffle=False)
        test_loader = data.build_dataloader(test_dataset, batch_size=64, shuffle=False)

        # Build model
        message_passing = nn.BondMessagePassing(
            d_h=hidden_size,
            depth=4,
            dropout=0.1
        )

        aggregation = nn.MeanAggregation()

        predictor = nn.RegressionFFN(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        model = models.MPNN(
            message_passing=message_passing,
            agg=aggregation,
            predictor=predictor,
            batch_norm=True,
            metrics=[nn.MAE(), nn.RMSE()],
            warmup_epochs=2,
            init_lr=1e-3,
            max_lr=1e-3,
            final_lr=1e-4
        )

        # Set output transform
        model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        # Callbacks
        checkpoint_dir = self.results_dir / f"model_{model_idx}"
        checkpoint_dir.mkdir(exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min"
            )
        ]

        # Train
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False
        )

        trainer.fit(model, train_loader, val_loader)

        # Test predictions
        test_preds = trainer.predict(model, test_loader)
        test_preds = np.concatenate([p.cpu().numpy() for p in test_preds]).flatten()

        return test_preds, model

    def train_ensemble(self, n_models: int = 5, augment_factor: int = 5, epochs: int = 50):
        """Train ensemble of models on the same split."""
        # Load and split data ONCE
        train_df, val_df, test_df = self.load_and_split_data(
            "../../Data/endpoints/train_KSOL.csv"
        )

        # Get true test values
        test_true = test_df['log10_ksol'].values

        # Train multiple models
        all_predictions = []
        models = []

        for i in range(n_models):
            preds, model = self.train_single_model(
                train_df, val_df, test_df,
                model_idx=i,
                augment=augment_factor,
                epochs=epochs,
                hidden_size=512
            )
            all_predictions.append(preds)
            models.append(model)

            # Show individual model performance
            mae = mean_absolute_error(test_true, preds)
            r2 = r2_score(test_true, preds)
            print(f"Model {i+1} - MAE: {mae:.4f}, R²: {r2:.4f}")

        # Ensemble predictions
        all_predictions = np.array(all_predictions)
        ensemble_preds = np.mean(all_predictions, axis=0)
        ensemble_std = np.std(all_predictions, axis=0)

        # Calculate ensemble metrics
        mae = mean_absolute_error(test_true, ensemble_preds)
        rmse = np.sqrt(mean_squared_error(test_true, ensemble_preds))
        r2 = r2_score(test_true, ensemble_preds)
        spearman = spearmanr(test_true, ensemble_preds)[0]

        within_05 = np.mean(np.abs(test_true - ensemble_preds) <= 0.5)
        within_10 = np.mean(np.abs(test_true - ensemble_preds) <= 1.0)

        print("\n" + "="*60)
        print("ENSEMBLE PERFORMANCE (Average of all models)")
        print("="*60)
        print(f"MAE:      {mae:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        print(f"R²:       {r2:.4f}")
        print(f"Spearman: {spearman:.4f}")
        print(f"Within 0.5 log: {within_05*100:.1f}%")
        print(f"Within 1.0 log: {within_10*100:.1f}%")
        print("="*60)

        # Save results
        results_df = pd.DataFrame({
            'smiles': test_df['smiles'].values,
            'true_log10': test_true,
            'pred_log10': ensemble_preds,
            'pred_std': ensemble_std,
            'true_ksol_uM': 10**test_true,
            'pred_ksol_uM': 10**ensemble_preds
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"ensemble_predictions_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Save metrics
        metrics = {
            'n_models': n_models,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'spearman': spearman,
            'within_0.5': within_05,
            'within_1.0': within_10
        }

        metrics_path = self.results_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return results_df, metrics


def main():
    parser = argparse.ArgumentParser(description='Best KSOL Training Strategy')
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in ensemble')
    parser.add_argument('--augment', type=int, default=5, help='Augmentation factor')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')

    args = parser.parse_args()

    trainer = BestKSOLTrainer()
    results_df, metrics = trainer.train_ensemble(
        n_models=args.n_models,
        augment_factor=args.augment,
        epochs=args.epochs
    )

    # Create visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(results_df['true_log10'], results_df['pred_log10'], alpha=0.5, s=10)
    plt.plot([results_df['true_log10'].min(), results_df['true_log10'].max()],
             [results_df['true_log10'].min(), results_df['true_log10'].max()], 'r--')
    plt.xlabel('True log10(KSOL)')
    plt.ylabel('Predicted log10(KSOL)')
    plt.title(f"Ensemble Predictions (R² = {metrics['r2']:.3f})")

    plt.subplot(1, 2, 2)
    errors = results_df['pred_log10'] - results_df['true_log10']
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f"Error Distribution (MAE = {metrics['mae']:.3f})")

    plt.tight_layout()
    plot_path = trainer.results_dir / "ensemble_performance.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()