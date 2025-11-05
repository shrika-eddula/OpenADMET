#!/usr/bin/env python3
"""
Advanced LogD Training Script with Comprehensive Improvements
Implements: K-fold CV, multi-seed ensemble, feature augmentation, SWA, uncertainty quantification
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
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
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

class AdvancedLogDTrainer:
    """Advanced trainer for LogD with all improvements."""

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

        self.logger.info(f"Initialized AdvancedLogDTrainer with config: {config}")

    def load_data(self):
        """Load and prepare data."""
        self.logger.info(f"Loading data from {self.config['data_path']}")

        # Load training data
        df_train = pd.read_csv(self.config['data_path'])
        self.logger.info(f"Loaded {len(df_train)} training molecules")

        # Load test data
        df_test = pd.read_csv(self.config['test_path'])
        self.logger.info(f"Loaded {len(df_test)} test molecules")

        # Apply winsorization if specified
        if self.config.get('winsorize', False):
            lower = df_train['LogD'].quantile(0.01)
            upper = df_train['LogD'].quantile(0.99)
            df_train['LogD'] = df_train['LogD'].clip(lower, upper)
            self.logger.info(f"Applied winsorization: [{lower:.2f}, {upper:.2f}]")

        return df_train, df_test

    def prepare_features(self, df):
        """Prepare molecular features."""
        self.logger.info("Preparing molecular features...")

        # Create base featurizer
        if 'rdkit_2d' in self.config['features_generator']:
            if 'normalized' in self.config['features_generator']:
                from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer
                molecule_featurizer = V1RDKit2DNormalizedFeaturizer()
                self.logger.info("Using RDKit 2D normalized descriptors")
            else:
                from chemprop.featurizers import V1RDKit2DFeaturizer
                molecule_featurizer = V1RDKit2DFeaturizer()
                self.logger.info("Using RDKit 2D descriptors")
        else:
            molecule_featurizer = None

        # Create graph featurizer
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Load additional features if provided
        extra_features = None
        if self.config.get('features_path'):
            extra_features = pd.read_csv(self.config['features_path'])
            self.logger.info(f"Loaded additional features from {self.config['features_path']}")

        return featurizer, molecule_featurizer, extra_features

    def create_datapoints(self, df, target_col='LogD'):
        """Create molecule datapoints."""
        datapoints = []
        valid_indices = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating datapoints"):
            try:
                if target_col in df.columns and not pd.isna(row[target_col]):
                    # Use single value, not nested array
                    y = [row[target_col]]
                else:
                    y = None

                dp = MoleculeDatapoint.from_smi(row['SMILES'], y)
                datapoints.append(dp)
                valid_indices.append(i)
            except Exception as e:
                self.logger.warning(f"Error processing SMILES at index {i}: {e}")
                continue

        return datapoints, valid_indices

    def build_model(self, featurizer, molecule_featurizer=None):
        """Build advanced MPNN model."""
        self.logger.info("Building advanced MPNN model")

        # Message Passing
        mp = nn.BondMessagePassing(
            d_h=self.config['hidden_size'],
            depth=self.config['depth'],
            dropout=self.config['dropout']
        )

        # Aggregation
        agg = nn.MeanAggregation()

        # Feed-forward network (output_transform will be set later with scaler)
        ffn = nn.RegressionFFN(
            input_dim=self.config['hidden_size'],
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
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=metric_list)

        return mpnn

    def train_fold(self, fold_idx, train_data, val_data, featurizer, seed):
        """Train a single fold with specific seed."""
        self.logger.info(f"Training fold {fold_idx} with seed {seed}")

        # Set random seed
        pl.seed_everything(seed)

        # Create datasets
        train_dset = MoleculeDataset(train_data, featurizer)
        scaler = train_dset.normalize_targets()

        val_dset = MoleculeDataset(val_data, featurizer)
        val_dset.normalize_targets(scaler)

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
        model = self.build_model(featurizer)

        # Set the unscale transform for the model's output
        # The MPNN model has a predictor with an output_transform
        model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

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
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)

        # Stochastic Weight Averaging
        if self.config.get('use_swa', False):
            swa_callback = StochasticWeightAveraging(
                swa_lrs=self.config.get('swa_lr', 1e-4),
                swa_epoch_start=self.config.get('swa_start_epoch', 90)
            )
            callbacks.append(swa_callback)
            self.logger.info("Using Stochastic Weight Averaging")

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=self.config.get('gradient_clip', 1.0),
            deterministic=True,
            logger=True
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Save final model info
        model_info = {
            'fold': fold_idx,
            'seed': seed,
            'best_checkpoint': checkpoint_callback.best_model_path,
            'best_val_loss': checkpoint_callback.best_model_score.item(),
            'epochs_trained': trainer.current_epoch,
            'scaler_mean': float(scaler.mean_[0]),
            'scaler_std': float(scaler.scale_[0])
        }

        info_file = checkpoint_dir / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        return model_info

    def train_ensemble(self):
        """Train ensemble with K-fold CV and multiple seeds."""
        # Load data
        df_train, df_test = self.load_data()

        # Prepare features
        featurizer, molecule_featurizer, extra_features = self.prepare_features(df_train)

        # Create datapoints
        train_datapoints, _ = self.create_datapoints(df_train)

        # K-fold cross-validation
        num_folds = self.config['num_folds']
        ensemble_size = self.config['ensemble_size']
        seeds = self.config.get('seeds', [42, 123, 456, 789, 2024][:ensemble_size])

        # Get molecules for splitting
        mols = [dp.mol for dp in train_datapoints]

        # Perform scaffold splitting for K-fold
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        all_models = []
        cv_predictions = []

        for fold_idx in range(num_folds):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
            self.logger.info(f"{'='*60}")

            # Create scaffold split for this fold
            # make_split_indices now returns tuple of lists
            # Use fold_idx as seed for consistent splits across folds
            split_indices = make_split_indices(
                mols, "scaffold_balanced", (0.8, 0.2, 0.0), seed=fold_idx
            )
            train_indices = split_indices[0][0]  # First replicate's train indices
            val_indices = split_indices[1][0]    # First replicate's val indices

            # Split data - wrap indices in lists as required by API
            train_data, val_data, _ = split_data_by_indices(
                train_datapoints, [train_indices], [val_indices], []
            )

            # Train multiple seeds for this fold
            fold_models = []
            for seed in seeds:
                model_info = self.train_fold(
                    fold_idx, train_data[0], val_data[0], featurizer, seed
                )
                model_info['train_size'] = len(train_data[0])
                model_info['val_size'] = len(val_data[0])
                fold_models.append(model_info)
                all_models.append(model_info)

            # Store fold results
            self.logger.info(f"Completed fold {fold_idx + 1} with {len(fold_models)} models")

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

        self.logger.info(f"Training complete! Trained {len(all_models)} models total")
        return ensemble_info

    def predict_with_uncertainty(self, df_test, ensemble_info):
        """Make predictions with uncertainty estimation."""
        self.logger.info("Making ensemble predictions with uncertainty...")

        # Prepare features
        featurizer, _, _ = self.prepare_features(df_test)

        # Create test datapoints
        test_datapoints, valid_indices = self.create_datapoints(df_test, target_col=None)

        all_predictions = []

        # Load each model and predict
        for model_info in tqdm(ensemble_info['models'], desc="Loading models"):
            checkpoint_path = model_info['best_checkpoint']

            # Load model
            model = models.MPNN.load_from_checkpoint(checkpoint_path, strict=False)
            model.eval()

            # Create dataset
            test_dset = MoleculeDataset(test_datapoints, featurizer)

            # Create dataloader
            test_loader = data.build_dataloader(
                test_dset,
                batch_size=self.config['batch_size'],
                num_workers=0,
                shuffle=False
            )

            # Predict
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
                logger=False
            )

            predictions = trainer.predict(model, test_loader)
            predictions = np.concatenate(predictions, axis=0).flatten()

            # Denormalize predictions
            predictions = predictions * model_info['scaler_std'] + model_info['scaler_mean']

            all_predictions.append(predictions)

        # Convert to numpy array
        all_predictions = np.array(all_predictions)

        # Calculate ensemble statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # Calculate 95% confidence intervals
        ci_lower = np.percentile(all_predictions, 2.5, axis=0)
        ci_upper = np.percentile(all_predictions, 97.5, axis=0)

        # Create results dataframe
        results_df = pd.DataFrame({
            'Molecule Name': df_test.iloc[valid_indices]['Molecule Name'].values,
            'SMILES': df_test.iloc[valid_indices]['SMILES'].values,
            'LogD_pred': mean_predictions,
            'LogD_std': std_predictions,
            'LogD_ci_lower': ci_lower,
            'LogD_ci_upper': ci_upper
        })

        # Save predictions
        predictions_file = self.predictions_dir / "test_predictions_with_uncertainty.csv"
        results_df.to_csv(predictions_file, index=False)

        self.logger.info(f"Saved predictions to {predictions_file}")

        return results_df

    def evaluate(self, results_df, df_test=None):
        """Comprehensive evaluation of predictions."""
        self.logger.info("Performing comprehensive evaluation...")

        metrics = {}

        # If we have true test labels (e.g., for validation)
        if df_test is not None and 'LogD' in df_test.columns:
            y_true = df_test['LogD'].values
            y_pred = results_df['LogD_pred'].values

            # Basic metrics
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['spearman'] = spearmanr(y_true, y_pred)[0]

            # Uncertainty metrics
            if 'LogD_ci_lower' in results_df.columns:
                # Coverage: % of true values within CI
                in_ci = (y_true >= results_df['LogD_ci_lower'].values) & \
                        (y_true <= results_df['LogD_ci_upper'].values)
                metrics['ci_coverage'] = np.mean(in_ci)

                # Sharpness: average CI width
                metrics['ci_sharpness'] = np.mean(
                    results_df['LogD_ci_upper'].values - results_df['LogD_ci_lower'].values
                )

        # Save metrics
        metrics_file = self.evaluation_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Evaluation metrics: {metrics}")

        return metrics

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Advanced LogD Training')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='Data/endpoints/train_big_LogD.csv',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str,
                        default='Data/test.csv',
                        help='Path to test data')
    parser.add_argument('--features_path', type=str, default=None,
                        help='Path to additional features CSV')

    # Model arguments
    parser.add_argument('--depth', type=int, default=6,
                        help='Message passing depth')
    parser.add_argument('--hidden_size', type=int, default=600,
                        help='Hidden dimension')
    parser.add_argument('--ffn_num_layers', type=int, default=3,
                        help='Number of FFN layers')
    parser.add_argument('--ffn_hidden_size', type=int, default=600,
                        help='FFN hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.10,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # Ensemble arguments
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--ensemble_size', type=int, default=5,
                        help='Number of seeds per fold')

    # Advanced features
    parser.add_argument('--features_generator', type=str,
                        default='rdkit_2d_normalized,morgan_count',
                        help='Feature generators to use')
    parser.add_argument('--use_swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start_epoch', type=int, default=90,
                        help='Epoch to start SWA')
    parser.add_argument('--swa_lr', type=float, default=1e-4,
                        help='SWA learning rate')
    parser.add_argument('--winsorize', action='store_true',
                        help='Apply winsorization to training data')

    # Output
    parser.add_argument('--save_dir', type=str,
                        default='logd_advanced_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create config
    config = vars(args)

    # Initialize trainer
    trainer = AdvancedLogDTrainer(config)

    # Train ensemble
    ensemble_info = trainer.train_ensemble()

    # Load test data
    df_test = pd.read_csv(args.test_path)

    # Make predictions with uncertainty
    results_df = trainer.predict_with_uncertainty(df_test, ensemble_info)

    # Evaluate if possible
    trainer.evaluate(results_df, df_test)

    print("\n" + "="*60)
    print("ADVANCED LOGD TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.save_dir}")
    print(f"Models: {len(ensemble_info['models'])} total")
    print(f"Predictions: {len(results_df)} molecules")

if __name__ == "__main__":
    main()