"""
Utility functions for Chemprop training pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_endpoint_config(config_file: str = None) -> Dict:
    """
    Load endpoint configuration from file or use defaults.

    Args:
        config_file: Path to configuration JSON file

    Returns:
        Configuration dictionary
    """
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Return default configuration
        return {
            'endpoints': {
                'LogD': {'skip': True},
                'HLM_CLint': {'skip': False},
                'MLM_CLint': {'skip': False},
                'Caco_2_Permeability_Papp_AtoB': {'skip': False},
                'Caco_2_Permeability_Efflux': {'skip': False},
                'KSOL': {'skip': False},
                'MBPB': {'skip': False},
                'MGMB': {'skip': False},
                'MPPB': {'skip': False}
            }
        }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     task_type: str = 'regression') -> Dict:
    """
    Calculate evaluation metrics based on task type.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if task_type == 'regression':
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)

        # Calculate additional statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

    elif task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['auc'] = roc_auc_score(y_true, y_pred)

    return metrics


def create_parity_plot(y_true: np.ndarray, y_pred: np.ndarray,
                      endpoint: str, save_path: Path = None) -> None:
    """
    Create a parity plot for predictions.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        endpoint: Name of the endpoint
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{endpoint} Parity Plot\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}')
    ax1.grid(True, alpha=0.3)

    # Error distribution
    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (Predicted - True)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Error Distribution\nMean: {errors.mean():.3f}, Std: {errors.std():.3f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close()


def merge_predictions(predictions_dir: Path, output_file: Path = None) -> pd.DataFrame:
    """
    Merge all endpoint predictions into a single dataframe.

    Args:
        predictions_dir: Directory containing individual prediction files
        output_file: Path to save merged predictions

    Returns:
        Merged predictions dataframe
    """
    prediction_files = list(predictions_dir.glob("test_*_predictions.csv"))

    if not prediction_files:
        logging.warning(f"No prediction files found in {predictions_dir}")
        return None

    # Load first file as base
    merged_df = pd.read_csv(prediction_files[0])
    base_columns = ['Molecule Name', 'SMILES']

    # Merge remaining files
    for pred_file in prediction_files[1:]:
        df = pd.read_csv(pred_file)
        # Get prediction column (last column that's not in base_columns)
        pred_cols = [col for col in df.columns if col not in base_columns]

        # Merge on molecule name
        merged_df = merged_df.merge(
            df[['Molecule Name'] + pred_cols],
            on='Molecule Name',
            how='outer'
        )

    if output_file:
        merged_df.to_csv(output_file, index=False)
        logging.info(f"Merged predictions saved to {output_file}")

    return merged_df


def create_training_report(results_summary: Dict, report_path: Path) -> None:
    """
    Create a comprehensive training report.

    Args:
        results_summary: Dictionary containing training results for all endpoints
        report_path: Path to save the report
    """
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("CHEMPROP TRAINING REPORT")
    report_lines.append("="*60)
    report_lines.append("")

    for endpoint, info in results_summary.items():
        report_lines.append(f"\n{endpoint}")
        report_lines.append("-"*40)

        if info:
            # Data splits
            if 'data_splits' in info:
                splits = info['data_splits']
                report_lines.append(f"Training samples: {splits.get('train', 'N/A')}")
                report_lines.append(f"Validation samples: {splits.get('val', 'N/A')}")
                report_lines.append(f"Test samples: {splits.get('test', 'N/A')}")

            # Performance metrics
            if 'test_performance' in info:
                perf = info['test_performance']
                report_lines.append("\nTest Performance:")
                for metric, value in perf.items():
                    report_lines.append(f"  {metric}: {value:.4f}")

            # Model info
            if 'best_checkpoint' in info:
                report_lines.append(f"\nBest checkpoint: {Path(info['best_checkpoint']).name}")
        else:
            report_lines.append("Training failed or skipped")

    report_lines.append("\n" + "="*60)

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logging.info(f"Training report saved to {report_path}")


def validate_data_files(data_dir: Path, endpoints: List[str]) -> Dict[str, bool]:
    """
    Validate that all required data files exist.

    Args:
        data_dir: Data directory path
        endpoints: List of endpoint names

    Returns:
        Dictionary mapping endpoint to availability status
    """
    endpoints_dir = data_dir / "endpoints"
    availability = {}

    for endpoint in endpoints:
        train_file = endpoints_dir / f"train_{endpoint}.csv"
        availability[endpoint] = train_file.exists()

        if not availability[endpoint]:
            logging.warning(f"Training file not found for {endpoint}: {train_file}")

    # Check test file
    test_file = data_dir / "test.csv"
    if not test_file.exists():
        logging.warning(f"Test file not found: {test_file}")

    return availability


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_statistics(df: pd.DataFrame, target_column: str) -> Dict:
    """
    Calculate statistics for the target column.

    Args:
        df: Dataframe with data
        target_column: Name of the target column

    Returns:
        Dictionary of statistics
    """
    values = df[target_column].dropna()

    stats = {
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'q25': values.quantile(0.25),
        'median': values.median(),
        'q75': values.quantile(0.75),
        'missing': df[target_column].isna().sum()
    }

    return stats


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"