#!/usr/bin/env python3
"""
Train Chemprop models for all ADMET endpoints sequentially.

This script trains MPNN models for each of the 9 ADMET endpoints:
- LogD
- HLM_CLint
- MLM_CLint
- Caco_2_Permeability_Papp_AtoB
- Caco_2_Permeability_Efflux
- KSOL
- MBPB
- MGMB
- MPPB

Each endpoint can be skipped using the skip flags in the configuration.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

# Import Chemprop modules
from chemprop import data, featurizers, models, nn

# Set up logging
def setup_logging(log_dir: Path):
    """Set up logging configuration."""
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


# Endpoint configurations
ENDPOINT_CONFIGS = {
    'LogD': {
        'skip': True,  # Skip LogD as requested
        'target_column': 'LogD',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'HLM_CLint': {
        'skip': False,
        'target_column': 'HLM CLint',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'MLM_CLint': {
        'skip': False,
        'target_column': 'MLM CLint',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'Caco_2_Permeability_Papp_AtoB': {
        'skip': False,
        'target_column': 'Caco-2 Permeability Papp A>B',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'Caco_2_Permeability_Efflux': {
        'skip': False,
        'target_column': 'Caco-2 Permeability Efflux',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'KSOL': {
        'skip': False,
        'target_column': 'KSOL',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'MBPB': {
        'skip': False,
        'target_column': 'MBPB',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'MGMB': {
        'skip': False,
        'target_column': 'MGMB',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    },
    'MPPB': {
        'skip': False,
        'target_column': 'MPPB',
        'task_type': 'regression',
        'metrics': ['mae', 'rmse', 'r2']
    }
}


# Default model hyperparameters
DEFAULT_MODEL_CONFIG = {
    'smiles_column': 'SMILES',
    'molecule_column': 'Molecule Name',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'batch_size': 32,
    'num_workers': 0,  # Set to 0 to avoid multiprocessing issues
    'max_epochs': 100,
    'hidden_size': 300,
    'depth': 3,
    'dropout': 0.0,
    'ffn_hidden_size': 300,
    'ffn_num_layers': 2,
    'learning_rate': 1e-3,
    'split_type': 'scaffold_balanced',  # Better for drug discovery
    'early_stopping_patience': 20,
    'gradient_clip_val': 1.0
}


class EndpointTrainer:
    """Trainer class for a single ADMET endpoint."""

    def __init__(self, endpoint: str, config: Dict, base_dir: Path, logger: logging.Logger):
        """
        Initialize the EndpointTrainer.

        Args:
            endpoint: Name of the endpoint to train
            config: Configuration dictionary
            base_dir: Base directory for the project
            logger: Logger instance
        """
        self.endpoint = endpoint
        self.config = config
        self.base_dir = base_dir
        self.logger = logger

        # Set up directories
        self.data_dir = base_dir / "Data"
        self.endpoints_dir = self.data_dir / "endpoints"
        self.checkpoint_dir = base_dir / "checkpoints" / endpoint
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize variables
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None
        self.model = None
        self.trainer = None

    def load_data(self) -> pd.DataFrame:
        """Load training data for the endpoint."""
        train_file = self.endpoints_dir / f"train_{self.endpoint}.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")

        df = pd.read_csv(train_file)
        self.logger.info(f"Loaded {len(df)} molecules for {self.endpoint}")

        # Log data statistics
        target_col = ENDPOINT_CONFIGS[self.endpoint]['target_column']
        self.logger.info(f"{target_col} range: [{df[target_col].min():.3f}, {df[target_col].max():.3f}]")
        self.logger.info(f"Mean {target_col}: {df[target_col].mean():.3f} ± {df[target_col].std():.3f}")

        return df

    def prepare_datapoints(self, df: pd.DataFrame) -> List:
        """Create MoleculeDatapoints from dataframe."""
        smiles = df[self.config['smiles_column']].values
        targets = df[ENDPOINT_CONFIGS[self.endpoint]['target_column']].values.reshape(-1, 1)

        datapoints = []
        failed_molecules = 0

        for smi, y in tqdm(zip(smiles, targets), total=len(smiles),
                           desc=f"Processing {self.endpoint} molecules"):
            try:
                dp = data.MoleculeDatapoint.from_smi(smi, y)
                datapoints.append(dp)
            except Exception as e:
                failed_molecules += 1
                self.logger.debug(f"Failed to process SMILES {smi}: {e}")

        if failed_molecules > 0:
            self.logger.warning(f"Failed to process {failed_molecules} molecules")

        self.logger.info(f"Successfully created {len(datapoints)} datapoints")
        return datapoints

    def split_data(self, datapoints: List) -> Tuple:
        """Split data into train/val/test sets."""
        mols = [d.mol for d in datapoints]

        split_sizes = (
            self.config['train_ratio'],
            self.config['val_ratio'],
            self.config['test_ratio']
        )

        self.logger.info(f"Performing {self.config['split_type']} split with ratios {split_sizes}")

        train_indices, val_indices, test_indices = data.make_split_indices(
            mols, self.config['split_type'], split_sizes
        )

        train_data, val_data, test_data = data.split_data_by_indices(
            datapoints, train_indices, val_indices, test_indices
        )

        self.logger.info(f"Split sizes - Train: {len(train_data[0])}, "
                        f"Val: {len(val_data[0])}, Test: {len(test_data[0])}")

        return train_data, val_data, test_data

    def create_datasets_and_loaders(self, train_data, val_data, test_data):
        """Create datasets and data loaders."""
        # Create featurizer
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Create training dataset and normalize
        train_dset = data.MoleculeDataset(train_data[0], featurizer)
        self.scaler = train_dset.normalize_targets()

        # Create validation and test datasets with same scaler
        val_dset = data.MoleculeDataset(val_data[0], featurizer)
        val_dset.normalize_targets(self.scaler)

        test_dset = data.MoleculeDataset(test_data[0], featurizer)
        test_dset.normalize_targets(self.scaler)

        # Create dataloaders
        train_loader = data.build_dataloader(
            train_dset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=True
        )

        val_loader = data.build_dataloader(
            val_dset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )

        test_loader = data.build_dataloader(
            test_dset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )

        return train_loader, val_loader, test_loader, featurizer

    def build_model(self, featurizer):
        """Build the MPNN model."""
        # Message Passing
        mp = nn.BondMessagePassing(
            d_h=self.config['hidden_size'],
            depth=self.config['depth'],
            dropout=self.config['dropout']
        )

        # Aggregation
        agg = nn.MeanAggregation()

        # Output transform to unscale predictions
        output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler)

        # Feed-Forward Network
        ffn = nn.RegressionFFN(
            input_dim=self.config['hidden_size'],
            hidden_dim=self.config['ffn_hidden_size'],
            n_layers=self.config['ffn_num_layers'],
            dropout=self.config['dropout'],
            output_transform=output_transform
        )

        # Metrics
        metric_list = [
            nn.metrics.RMSE(),
            nn.metrics.MAE(),
            nn.metrics.R2Score()
        ]

        # Build MPNN with metrics passed as parameter
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)

        self.logger.info(f"Built MPNN model for {self.endpoint}")
        return mpnn

    def train(self, train_loader, val_loader):
        """Train the model."""
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"{self.endpoint}-{{epoch:03d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=False
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config['early_stopping_patience'],
            mode="min",
            verbose=False
        )

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=True,
            gradient_clip_val=self.config['gradient_clip_val'],
            deterministic=True
        )

        # Train model
        self.logger.info(f"Starting training for {self.endpoint}...")
        self.trainer.fit(self.model, train_loader, val_loader)

        # Store best checkpoint path
        self.best_checkpoint = checkpoint_callback.best_model_path
        self.logger.info(f"Best checkpoint saved at: {self.best_checkpoint}")

        return checkpoint_callback.best_model_path

    def evaluate(self, test_loader):
        """Evaluate the model on test set."""
        self.logger.info(f"Evaluating {self.endpoint} on test set...")
        test_results = self.trainer.test(self.model, dataloaders=test_loader)

        # Log results
        for key, value in test_results[0].items():
            self.logger.info(f"{key}: {value:.4f}")

        return test_results[0]

    def predict_test_set(self, featurizer):
        """Make predictions on the competition test set."""
        # Load test data
        test_file = self.data_dir / "test.csv"
        df_test = pd.read_csv(test_file)

        test_smiles = df_test[self.config['smiles_column']].values
        test_mol_names = df_test[self.config['molecule_column']].values

        # Create datapoints
        test_datapoints = []
        valid_indices = []

        for i, smi in enumerate(tqdm(test_smiles, desc=f"Processing test SMILES for {self.endpoint}")):
            try:
                dp = data.MoleculeDatapoint.from_smi(smi)
                test_datapoints.append(dp)
                valid_indices.append(i)
            except Exception as e:
                self.logger.debug(f"Failed to process test SMILES {smi}: {e}")
                continue

        if not test_datapoints:
            self.logger.warning(f"No valid test molecules for {self.endpoint}")
            return None

        # Create dataset and dataloader
        test_dataset = data.MoleculeDataset(test_datapoints, featurizer=featurizer)
        test_dataloader = data.build_dataloader(
            test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )

        # Load best model (strict=False to handle metrics mismatch)
        best_model = models.MPNN.load_from_checkpoint(self.best_checkpoint, strict=False)
        best_model.eval()

        # Make predictions
        with torch.inference_mode():
            trainer_predict = pl.Trainer(
                accelerator="auto",
                devices=1,
                enable_progress_bar=True,
                logger=False
            )
            predictions = trainer_predict.predict(best_model, test_dataloader)

        # Process predictions
        predictions = np.concatenate(predictions, axis=0).flatten()

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Molecule Name': test_mol_names[valid_indices],
            'SMILES': test_smiles[valid_indices],
            f'{self.endpoint}_pred': predictions
        })

        # Save predictions
        pred_dir = self.base_dir / "predictions" / "chemprop"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"test_{self.endpoint}_predictions.csv"
        predictions_df.to_csv(pred_file, index=False)

        self.logger.info(f"Saved {len(predictions)} predictions to {pred_file}")

        return predictions_df

    def save_model_info(self, test_results):
        """Save model information and metrics."""
        model_info = {
            'endpoint': self.endpoint,
            'model_type': 'Chemprop MPNN',
            'training_config': self.config,
            'data_splits': {
                'train': len(self.train_data[0]) if self.train_data else 0,
                'val': len(self.val_data[0]) if self.val_data else 0,
                'test': len(self.test_data[0]) if self.test_data else 0
            },
            'test_performance': {k: float(v) for k, v in test_results.items()},
            'best_checkpoint': str(self.best_checkpoint) if hasattr(self, 'best_checkpoint') else None,
            'scaler_params': {
                'mean': float(self.scaler.mean_[0]),
                'std': float(self.scaler.scale_[0])
            } if self.scaler else None
        }

        # Save to JSON
        info_file = self.checkpoint_dir / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        self.logger.info(f"Model info saved to {info_file}")

        return model_info

    def run_training_pipeline(self):
        """Run the complete training pipeline for the endpoint."""
        try:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Starting training pipeline for {self.endpoint}")
            self.logger.info(f"{'='*60}")

            # 1. Load data
            df = self.load_data()

            # 2. Prepare datapoints
            datapoints = self.prepare_datapoints(df)

            # 3. Split data
            self.train_data, self.val_data, self.test_data = self.split_data(datapoints)

            # 4. Create datasets and loaders
            train_loader, val_loader, test_loader, featurizer = self.create_datasets_and_loaders(
                self.train_data, self.val_data, self.test_data
            )

            # 5. Build model
            self.model = self.build_model(featurizer)

            # 6. Train model
            best_checkpoint = self.train(train_loader, val_loader)

            # 7. Evaluate on test set
            test_results = self.evaluate(test_loader)

            # 8. Make predictions on competition test set
            predictions = self.predict_test_set(featurizer)

            # 9. Save model information
            model_info = self.save_model_info(test_results)

            self.logger.info(f"Successfully completed training for {self.endpoint}")
            return True, model_info

        except Exception as e:
            self.logger.error(f"Error training {self.endpoint}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, None


def main(args):
    """Main function to train all endpoints."""
    # Set up base directory
    base_dir = Path(args.base_dir)

    # Set up logging
    log_dir = base_dir / "chemprop_training" / "logs"
    logger = setup_logging(log_dir)

    logger.info("Starting Chemprop training for all ADMET endpoints")
    logger.info(f"Base directory: {base_dir}")

    # Load custom config if provided
    config = DEFAULT_MODEL_CONFIG.copy()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Update config with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs

    # Track results
    results_summary = {}
    successful_endpoints = []
    failed_endpoints = []

    # Train each endpoint
    for endpoint, endpoint_config in ENDPOINT_CONFIGS.items():
        if endpoint_config['skip']:
            logger.info(f"Skipping {endpoint} (skip flag is True)")
            continue

        if args.endpoints and endpoint not in args.endpoints:
            logger.info(f"Skipping {endpoint} (not in specified endpoints)")
            continue

        trainer = EndpointTrainer(endpoint, config, base_dir, logger)
        success, model_info = trainer.run_training_pipeline()

        if success:
            successful_endpoints.append(endpoint)
            results_summary[endpoint] = model_info
        else:
            failed_endpoints.append(endpoint)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful endpoints ({len(successful_endpoints)}): {', '.join(successful_endpoints)}")
    if failed_endpoints:
        logger.info(f"Failed endpoints ({len(failed_endpoints)}): {', '.join(failed_endpoints)}")

    # Save summary
    summary_file = base_dir / "chemprop_training" / "training_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'successful_endpoints': successful_endpoints,
            'failed_endpoints': failed_endpoints,
            'results': results_summary,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"\nTraining summary saved to {summary_file}")

    return len(failed_endpoints) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chemprop models for ADMET endpoints")
    parser.add_argument('--base_dir', type=str, default=".",
                       help='Base directory for the project')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to custom configuration JSON file')
    parser.add_argument('--endpoints', nargs='+', default=None,
                       help='Specific endpoints to train (default: all non-skipped)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of training epochs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Perform a dry run without actual training')

    args = parser.parse_args()

    # Run main
    success = main(args)
    sys.exit(0 if success else 1)