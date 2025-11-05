#!/usr/bin/env python3
"""
Train MPPB model using LogD as an additional feature
Sequential prediction approach: SMILES + LogD → MPPB
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

class MPPBwithLogDTrainer:
    """Trainer for MPPB using LogD as additional feature."""

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

        self.logger.info(f"Initialized MPPBwithLogDTrainer with config: {config}")

    def load_data(self):
        """Load and prepare data with LogD features."""
        self.logger.info(f"Loading data from {self.config['data_path']}")

        # Load training data
        df_train = pd.read_csv(self.config['data_path'])
        self.logger.info(f"Loaded {len(df_train)} training molecules")

        # Load test data
        df_test = pd.read_csv(self.config['test_path'])
        self.logger.info(f"Loaded {len(df_test)} test molecules")

        # Check LogD availability
        logd_train = df_train['LogD'].notna().sum()
        logd_test = df_test['LogD'].notna().sum()
        self.logger.info(f"LogD available: Train={logd_train}/{len(df_train)}, Test={logd_test}/{len(df_test)}")

        return df_train, df_test

    def create_datapoints_with_features(self, df, target_col='MPPB'):
        """Create molecule datapoints with LogD as additional feature."""
        datapoints = []
        valid_indices = []

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
                    # Use mean LogD if missing (for test set without LogD)
                    x_d = np.array([self.config.get('mean_logd', 2.0)], dtype=np.float32)

                # Create molecule datapoint with external features
                dp = MoleculeDatapoint.from_smi(row['SMILES'], y, x_d=x_d)

                datapoints.append(dp)
                valid_indices.append(i)

            except Exception as e:
                self.logger.warning(f"Error processing SMILES at index {i}: {e}")
                continue

        return datapoints, valid_indices

    def build_model_with_features(self, input_dim=1):
        """Build MPNN model that accepts additional scalar features."""
        self.logger.info("Building MPNN model with LogD feature integration")

        # Message Passing
        mp = nn.BondMessagePassing(
            d_h=self.config['hidden_size'],
            depth=self.config['depth'],
            dropout=self.config['dropout']
        )

        # Aggregation
        agg = nn.MeanAggregation()

        # Feed-forward network with additional feature dimension
        # The FFN will receive hidden_size (from MPNN) + input_dim (LogD) dimensions
        ffn = nn.RegressionFFN(
            input_dim=self.config['hidden_size'] + input_dim,  # Add LogD dimension
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
        batch_norm = self.config.get('batch_norm', True)

        # Don't set X_d_transform here, will be set after normalization
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=metric_list)

        return mpnn

    def train_model(self, train_df, val_df=None):
        """Train the model with LogD features."""
        # Set random seed
        pl.seed_everything(self.config.get('seed', 42))

        # Create datapoints with LogD features
        train_datapoints, _ = self.create_datapoints_with_features(train_df)

        # Create train/val split if val_df not provided
        if val_df is None:
            # Create scaffold split
            mols = [dp.mol for dp in train_datapoints]
            split_indices = make_split_indices(
                mols, "scaffold_balanced", (0.8, 0.2, 0.0), seed=42
            )
            train_indices = split_indices[0][0]
            val_indices = split_indices[1][0]

            # Split data
            train_data, val_data, _ = split_data_by_indices(
                train_datapoints, [train_indices], [val_indices], []
            )
            train_data = train_data[0]
            val_data = val_data[0]
        else:
            train_data = train_datapoints
            val_datapoints, _ = self.create_datapoints_with_features(val_df)
            val_data = val_datapoints

        # Create datasets (LogD features are already in datapoints)
        train_dset = MoleculeDataset(train_data)
        val_dset = MoleculeDataset(val_data)

        # Normalize targets
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        # Normalize LogD features (X_d is the key for external features)
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
        model = self.build_model_with_features(input_dim=1)  # 1 for LogD

        # Set the transforms for predictions and inputs
        model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        if x_d_scaler is not None:
            model.X_d_transform = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

        # Configure callbacks
        callbacks = []

        # Model checkpoint
        checkpoint_dir = self.models_dir / "best_model"
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
            patience=self.config.get('patience', 10),
            mode="min"
        )
        callbacks.append(early_stopping)

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=pl.loggers.CSVLogger(self.save_dir / "logs"),
            gradient_clip_val=1.0
        )

        # Train model
        self.logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)

        # Save model info
        model_info = {
            'best_checkpoint': checkpoint_callback.best_model_path,
            'best_val_loss': float(checkpoint_callback.best_model_score),
            'epochs_trained': trainer.current_epoch,
            'scaler_mean': float(scaler.mean_[0]),
            'scaler_std': float(scaler.scale_[0]),
            'x_d_scaler_mean': float(x_d_scaler.mean_[0]) if x_d_scaler else None,
            'x_d_scaler_std': float(x_d_scaler.scale_[0]) if x_d_scaler else None,
            'config': self.config
        }

        info_file = self.models_dir / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        self.logger.info(f"Training completed. Best model: {checkpoint_callback.best_model_path}")
        self.logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

        return model, scaler, x_d_scaler, model_info

    def evaluate_model(self, model, test_df):
        """Evaluate the model on test set."""
        # Create test datapoints with LogD
        test_datapoints, _ = self.create_datapoints_with_features(test_df)

        # Create dataset
        test_dset = MoleculeDataset(test_datapoints)

        # Create dataloader
        test_loader = data.build_dataloader(
            test_dset,
            batch_size=self.config['batch_size'],
            num_workers=0,
            shuffle=False
        )

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
        all_preds = []
        for batch_preds in predictions:
            all_preds.extend(batch_preds.cpu().numpy())

        y_pred = np.array(all_preds).flatten()

        # Get true values if available
        if 'MPPB' in test_df.columns:
            y_true = test_df['MPPB'].values
            valid_mask = ~np.isnan(y_true)

            if np.any(valid_mask):
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

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
                self.logger.info("TEST SET EVALUATION")
                self.logger.info("="*60)
                self.logger.info(f"Samples: {len(y_true_valid)}")
                self.logger.info(f"MAE:  {mae:.4f}")
                self.logger.info(f"RMSE: {rmse:.4f}")
                self.logger.info(f"R²:   {r2:.4f}")
                self.logger.info(f"Pearson r:  {pearson_r:.4f}")
                self.logger.info(f"Spearman r: {spearman_r:.4f}")

                # Save metrics
                metrics_file = self.evaluation_dir / "test_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            else:
                metrics = None
        else:
            metrics = None

        # Save predictions
        output_df = test_df[['Molecule Name', 'SMILES']].copy()
        output_df['LogD_input'] = test_df['LogD']
        output_df['MPPB_pred'] = y_pred
        if 'MPPB' in test_df.columns:
            output_df['MPPB_true'] = test_df['MPPB']

        output_file = self.predictions_dir / "test_predictions.csv"
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions saved to {output_file}")

        return y_pred, metrics

def main():
    parser = argparse.ArgumentParser(description='Train MPPB model with LogD as feature')
    parser.add_argument('--data_path', type=str,
                        default='../Data/endpoints/train_big_LogD_MPPB.csv',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str,
                        default='../Data/paired_endpoints/train_LogD_MPPB.csv',
                        help='Path to test data (for evaluation)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing layers')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden dimension size')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of FFN layers')
    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='FFN hidden size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='mppb_logd_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create config
    config = vars(args)

    # Initialize trainer
    trainer = MPPBwithLogDTrainer(config)

    # Load data
    train_df, test_df = trainer.load_data()

    # Calculate correlation in training data
    corr = train_df['LogD'].corr(train_df['MPPB'])
    print(f"\nLogD-MPPB correlation in training: {corr:.4f}")

    # Train model
    model, scaler, x_d_scaler, model_info = trainer.train_model(train_df)

    # Evaluate on test set
    if len(test_df) > 0:
        predictions, metrics = trainer.evaluate_model(model, test_df)

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"✓ Model saved to: {config['save_dir']}/models/")
    print(f"✓ Predictions saved to: {config['save_dir']}/predictions/")
    if metrics:
        print(f"✓ Test MAE: {metrics['mae']:.4f}")
        print(f"✓ Test R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()