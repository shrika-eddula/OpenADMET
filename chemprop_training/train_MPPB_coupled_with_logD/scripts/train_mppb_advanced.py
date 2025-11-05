#!/usr/bin/env python3
"""
Advanced MPPB Training with LogD Feature
Implements: K-fold CV, multi-seed ensemble, uncertainty quantification
Sequential prediction: SMILES + LogD → MPPB
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
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Import Chemprop
from chemprop import data, featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data import make_split_indices, split_data_by_indices

# Set up logging
def setup_logging(save_dir):
    """Set up comprehensive logging."""
    log_dir = Path(save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class AdvancedMPPBTrainer:
    """Advanced trainer for MPPB with LogD feature and ensemble methods."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config['save_dir'])
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.save_dir / "models"
        self.predictions_dir = self.save_dir / "predictions"
        self.evaluation_dir = self.save_dir / "evaluation"

        for dir_path in [self.models_dir, self.predictions_dir, self.evaluation_dir]:
            dir_path.mkdir(exist_ok=True)

        self.logger.info(f"Initialized AdvancedMPPBTrainer with config: {config}")

    def load_data(self):
        """Load and prepare data."""
        self.logger.info(f"Loading data from {self.config['data_path']}")

        # Load training data
        df_train = pd.read_csv(self.config['data_path'])
        self.logger.info(f"Loaded {len(df_train)} training molecules")

        # Load test data
        df_test = pd.read_csv(self.config['test_path'])
        self.logger.info(f"Loaded {len(df_test)} test molecules")

        # Check LogD-MPPB correlation
        corr = df_train['LogD'].corr(df_train['MPPB'])
        self.logger.info(f"LogD-MPPB correlation in training: {corr:.4f}")

        return df_train, df_test

    def create_datapoints_with_features(self, df, target_col='MPPB', mean_logd=None):
        """Create molecule datapoints with LogD as additional feature."""
        datapoints = []
        valid_indices = []

        if mean_logd is None:
            mean_logd = df['LogD'].mean() if 'LogD' in df.columns else 2.0

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating datapoints"):
            try:
                # Get target value
                if target_col in df.columns and not pd.isna(row[target_col]):
                    y = [row[target_col]]
                else:
                    y = None

                # Extract LogD as additional feature
                if 'LogD' in df.columns and not pd.isna(row['LogD']):
                    x_d = np.array([row['LogD']], dtype=np.float32)
                else:
                    x_d = np.array([mean_logd], dtype=np.float32)

                # Create molecule datapoint with external features
                dp = MoleculeDatapoint.from_smi(row['SMILES'], y, x_d=x_d)

                datapoints.append(dp)
                valid_indices.append(i)

            except Exception as e:
                self.logger.warning(f"Error processing SMILES at index {i}: {e}")
                continue

        return datapoints, valid_indices

    def build_model(self):
        """Build MPNN model that accepts LogD features."""
        self.logger.info("Building advanced MPNN model with LogD integration")

        # Message Passing
        mp = nn.BondMessagePassing(
            d_h=self.config['hidden_size'],
            depth=self.config['depth'],
            dropout=self.config['dropout']
        )

        # Aggregation
        agg = nn.MeanAggregation()

        # Feed-forward network (receives hidden_size + 1 for LogD)
        ffn = nn.RegressionFFN(
            input_dim=self.config['hidden_size'] + 1,  # +1 for LogD
            hidden_dim=self.config['ffn_hidden_size'],
            n_layers=self.config['ffn_num_layers'],
            dropout=self.config['dropout']
        )

        # Metrics
        metric_list = [
            nn.metrics.MAE(),
            nn.metrics.RMSE(),
            nn.metrics.R2Score()
        ]

        # Build MPNN
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)

        return mpnn

    def train_fold(self, fold_idx, train_data, val_data, seed):
        """Train a single model for one fold with a specific seed."""
        self.logger.info(f"Training fold {fold_idx} with seed {seed}")

        # Set random seed
        pl.seed_everything(seed)

        # Create datasets
        train_dset = MoleculeDataset(train_data)
        val_dset = MoleculeDataset(val_data)

        # Normalize targets
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        # Normalize LogD features
        x_d_scaler = train_dset.normalize_inputs('X_d')
        val_dset.normalize_inputs('X_d', x_d_scaler)

        # Create dataloaders
        train_loader = data.build_dataloader(
            train_dset,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 0),
            shuffle=True
        )

        val_loader = data.build_dataloader(
            val_dset,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 0),
            shuffle=False
        )

        # Build model
        model = self.build_model()

        # Set transforms
        model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        if x_d_scaler is not None:
            model.X_d_transform = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

        # Configure callbacks
        callbacks = []

        # Model checkpoint
        checkpoint_dir = self.models_dir / f"fold_{fold_idx}" / f"seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.get('patience', 20),
            mode="min"
        )
        callbacks.append(early_stopping)

        # Stochastic Weight Averaging if specified
        if self.config.get('use_swa', False):
            swa = StochasticWeightAveraging(
                swa_lrs=self.config.get('swa_lr', 1e-4),
                swa_epoch_start=self.config.get('swa_start_epoch', 0.8)
            )
            callbacks.append(swa)

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=pl.loggers.CSVLogger(checkpoint_dir),
            gradient_clip_val=1.0,
            enable_model_summary=False
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Save model info
        best_checkpoint = checkpoint_callback.best_model_path
        model_info = {
            'fold': fold_idx,
            'seed': seed,
            'best_checkpoint': best_checkpoint,
            'best_val_loss': float(checkpoint_callback.best_model_score),
            'epochs_trained': trainer.current_epoch,
            'scaler_mean': float(scaler.mean_[0]),
            'scaler_std': float(scaler.scale_[0]),
            'x_d_scaler_mean': float(x_d_scaler.mean_[0]) if x_d_scaler else None,
            'x_d_scaler_std': float(x_d_scaler.scale_[0]) if x_d_scaler else None,
            'train_size': len(train_data),
            'val_size': len(val_data)
        }

        info_file = checkpoint_dir / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        return model_info

    def train_ensemble(self):
        """Train ensemble with K-fold CV and multiple seeds."""
        # Load data
        df_train, df_test = self.load_data()

        # Create datapoints
        train_datapoints, _ = self.create_datapoints_with_features(df_train)
        mean_logd = df_train['LogD'].mean()

        # K-fold cross-validation
        num_folds = self.config['num_folds']
        ensemble_size = self.config['ensemble_size']
        seeds = self.config.get('seeds', [42, 123, 456, 789, 2024][:ensemble_size])

        # Get molecules for splitting
        mols = [dp.mol for dp in train_datapoints]

        all_models = []

        for fold_idx in range(num_folds):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
            self.logger.info(f"{'='*60}")

            # Create scaffold split for this fold
            split_indices = make_split_indices(
                mols, "scaffold_balanced", (0.8, 0.2, 0.0), seed=fold_idx
            )
            train_indices = split_indices[0][0]
            val_indices = split_indices[1][0]

            # Split data
            train_data, val_data, _ = split_data_by_indices(
                train_datapoints, [train_indices], [val_indices], []
            )

            # Train multiple seeds for this fold
            for seed in seeds:
                model_info = self.train_fold(
                    fold_idx, train_data[0], val_data[0], seed
                )
                all_models.append(model_info)

        # Save ensemble info
        ensemble_info = {
            'num_folds': num_folds,
            'ensemble_size': ensemble_size,
            'total_models': len(all_models),
            'models': all_models
        }

        ensemble_file = self.save_dir / "ensemble_info.json"
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_info, f, indent=2)

        self.logger.info(f"\nEnsemble training completed: {len(all_models)} models")

        # Generate ensemble predictions on test set
        self.generate_ensemble_predictions(df_test, all_models, mean_logd)

        return ensemble_info

    def generate_ensemble_predictions(self, test_df, model_infos, mean_logd):
        """Generate predictions from ensemble with uncertainty estimates."""
        self.logger.info("Generating ensemble predictions...")

        # Create test datapoints
        test_datapoints, _ = self.create_datapoints_with_features(test_df, mean_logd=mean_logd)
        test_dset = MoleculeDataset(test_datapoints)

        # Create dataloader
        test_loader = data.build_dataloader(
            test_dset,
            batch_size=self.config['batch_size'],
            num_workers=0,
            shuffle=False
        )

        all_predictions = []

        # Get predictions from each model
        for model_info in tqdm(model_infos, desc="Loading models"):
            # Load model
            checkpoint_path = model_info['best_checkpoint']
            model = models.MPNN.load_from_checkpoint(checkpoint_path, strict=False)

            # CRITICAL: Restore the output transform for denormalization
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array([model_info['scaler_mean']])
            scaler.scale_ = np.array([model_info['scaler_std']])
            model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

            # Use trainer for predictions
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
                logger=False
            )

            # Get predictions
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
        ensemble_ci_lower = np.percentile(all_predictions, 2.5, axis=0)
        ensemble_ci_upper = np.percentile(all_predictions, 97.5, axis=0)

        # Evaluate if we have true values
        if 'MPPB' in test_df.columns:
            y_true = test_df['MPPB'].values
            valid_mask = ~np.isnan(y_true)

            if np.any(valid_mask):
                y_true_valid = y_true[valid_mask]
                y_pred_valid = ensemble_mean[valid_mask]

                # Calculate metrics
                mae = mean_absolute_error(y_true_valid, y_pred_valid)
                rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                r2 = r2_score(y_true_valid, y_pred_valid)
                pearson_r, _ = pearsonr(y_true_valid, y_pred_valid)
                spearman_r, _ = spearmanr(y_true_valid, y_pred_valid)

                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'pearson_r': pearson_r,
                    'spearman_r': spearman_r,
                    'n_samples': len(y_true_valid)
                }

                self.logger.info("\n" + "="*60)
                self.logger.info("ENSEMBLE TEST SET EVALUATION")
                self.logger.info("="*60)
                self.logger.info(f"Number of models: {len(all_predictions)}")
                self.logger.info(f"Test samples: {len(y_true_valid)}")
                self.logger.info(f"\nMetrics:")
                self.logger.info(f"  MAE:  {mae:.4f}")
                self.logger.info(f"  RMSE: {rmse:.4f}")
                self.logger.info(f"  R²:   {r2:.4f}")
                self.logger.info(f"  Pearson r:  {pearson_r:.4f}")
                self.logger.info(f"  Spearman r: {spearman_r:.4f}")

                # Save metrics
                metrics_file = self.evaluation_dir / "ensemble_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

        # Save predictions with uncertainty
        output_df = test_df[['Molecule Name', 'SMILES']].copy()
        output_df['LogD_input'] = test_df['LogD'] if 'LogD' in test_df.columns else mean_logd
        output_df['MPPB_pred'] = ensemble_mean
        output_df['MPPB_std'] = ensemble_std
        output_df['MPPB_ci_lower'] = ensemble_ci_lower
        output_df['MPPB_ci_upper'] = ensemble_ci_upper
        if 'MPPB' in test_df.columns:
            output_df['MPPB_true'] = test_df['MPPB']

        output_file = self.predictions_dir / "test_predictions_ensemble.csv"
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"Ensemble predictions saved to {output_file}")

        # Analyze uncertainty
        self.logger.info("\n" + "="*60)
        self.logger.info("UNCERTAINTY ANALYSIS")
        self.logger.info("="*60)
        self.logger.info(f"Mean uncertainty (std): {ensemble_std.mean():.4f}")
        self.logger.info(f"Max uncertainty: {ensemble_std.max():.4f}")
        self.logger.info(f"Min uncertainty: {ensemble_std.min():.4f}")

        # Check calibration if we have true values
        if 'MPPB' in test_df.columns and np.any(valid_mask):
            in_ci = ((y_true_valid >= ensemble_ci_lower[valid_mask]) &
                    (y_true_valid <= ensemble_ci_upper[valid_mask])).mean()
            self.logger.info(f"Coverage (95% CI): {in_ci*100:.1f}%")

        return ensemble_mean, ensemble_std

def main():
    parser = argparse.ArgumentParser(description='Advanced MPPB training with LogD feature')
    parser.add_argument('--data_path', type=str,
                        default='../Data/endpoints/train_big_LogD_MPPB.csv',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str,
                        default='../Data/paired_endpoints/train_LogD_MPPB.csv',
                        help='Path to test data')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--hidden_size', type=int, default=400,
                        help='Hidden dimension size')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of FFN layers')
    parser.add_argument('--ffn_hidden_size', type=int, default=400,
                        help='FFN hidden size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--num_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--ensemble_size', type=int, default=3,
                        help='Number of seeds per fold')
    parser.add_argument('--use_swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start_epoch', type=int, default=40,
                        help='Epoch to start SWA')
    parser.add_argument('--save_dir', type=str, default='mppb_advanced_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create config
    config = vars(args)

    # Initialize trainer
    trainer = AdvancedMPPBTrainer(config)

    # Train ensemble
    ensemble_info = trainer.train_ensemble()

    print("\n" + "="*60)
    print("ADVANCED TRAINING COMPLETED")
    print("="*60)
    print(f"✓ Trained {ensemble_info['total_models']} models")
    print(f"✓ Configuration: {ensemble_info['num_folds']} folds × {ensemble_info['ensemble_size']} seeds")
    print(f"✓ Results saved to: {config['save_dir']}/")

if __name__ == "__main__":
    main()