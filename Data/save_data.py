"""
Data saving utilities for OpenADMET ExpansionRx Challenge.

This module provides functions to save training, test, and teaser datasets
to local files in various formats (CSV, pickle, parquet).
"""

from pathlib import Path
from typing import Optional, Literal

import pandas as pd
import yaml
from constants import CONFIG_PATH


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        Dictionary containing configuration data.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Try looking in parent directory if not found in current directory
        config_file = Path("..") / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_teaser(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    overwrite: bool = False,
    config_path: str = "config.yaml"
) -> None:
    """
    Save teaser dataset to a local file.
    
    Args:
        df: DataFrame containing the teaser dataset to save.
        output_path: Path where the file will be saved. If None, uses default from config.
        format: File format to save. Options: "csv", "pickle", "parquet". Defaults to "csv".
        overwrite: If True, overwrite existing file. If False, raise error if file exists.
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        ValueError: If an unsupported format is specified.
    """
    if output_path is None:
        config = load_config(config_path)
        data_root = config['constants']['data_root']
        filename = config['constants']['output_filenames']['teaser']
        output_path = f"{data_root}/{filename}"
    
    _save_data(df, output_path, format, overwrite, "teaser", config_path)


def save_train(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    overwrite: bool = False,
    config_path: str = CONFIG_PATH
) -> None:
    """
    Save training dataset to a local file.
    
    Args:
        df: DataFrame containing the training dataset to save.
        output_path: Path where the file will be saved. If None, uses default from config.
        format: File format to save. Options: "csv", "pickle", "parquet". Defaults to "csv".
        overwrite: If True, overwrite existing file. If False, raise error if file exists.
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        ValueError: If an unsupported format is specified.
    """
    if output_path is None:
        config = load_config(config_path)
        data_root = config['constants']['data_root']
        filename = config['constants']['output_filenames']['train']
        output_path = f"{data_root}/{filename}"
    
    _save_data(df, output_path, format, overwrite, "training", config_path)


def save_test(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    overwrite: bool = False,
    config_path: str = CONFIG_PATH
) -> None:
    """
    Save test dataset to a local file.
    
    Args:
        df: DataFrame containing the test dataset to save.
        output_path: Path where the file will be saved. If None, uses default from config.
        format: File format to save. Options: "csv", "pickle", "parquet". Defaults to "csv".
        overwrite: If True, overwrite existing file. If False, raise error if file exists.
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        ValueError: If an unsupported format is specified.
    """
    if output_path is None:
        config = load_config(config_path)
        data_root = config['constants']['data_root']
        filename = config['constants']['output_filenames']['test']
        output_path = f"{data_root}/{filename}"
    
    _save_data(df, output_path, format, overwrite, "test", config_path)


def _save_data(
    df: pd.DataFrame,
    output_path: str,
    format: str,
    overwrite: bool,
    dataset_name: str,
    config_path: str = CONFIG_PATH
) -> None:
    """
    Internal helper function to save data in specified format.
    
    Args:
        df: DataFrame to save.
        output_path: Path where the file will be saved.
        format: File format ("csv", "pickle", or "parquet").
        overwrite: Whether to overwrite existing files.
        dataset_name: Name of the dataset for logging purposes.
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        ValueError: If an unsupported format is specified.
    """
    output_file = Path(output_path)
    
    # Check if file exists
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"File '{output_path}' already exists. "
            "Set overwrite=True to replace it."
        )
    
    # Create parent directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on format
    if format == "csv":
        df.to_csv(output_file, index=False)
    elif format == "pickle":
        df.to_pickle(output_file)
    elif format == "parquet":
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError(
            f"Unsupported format '{format}'. "
            "Supported formats: 'csv', 'pickle', 'parquet'"
        )
    
    print(f"✓ {dataset_name.capitalize()} data saved to '{output_path}' ({format} format)")
    print(f"  Shape: {df.shape}")
    print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")


def save_all_datasets(
    df_teaser: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Optional[str] = None,
    format: Literal["csv", "pickle", "parquet"] = "csv",
    overwrite: bool = False,
    config_path: str = CONFIG_PATH
) -> None:
    """
    Save all three datasets (teaser, train, test) at once.
    
    Args:
        df_teaser: DataFrame containing the teaser dataset.
        df_train: DataFrame containing the training dataset.
        df_test: DataFrame containing the test dataset.
        output_dir: Directory where files will be saved. If None, uses default from config.
        format: File format to save. Options: "csv", "pickle", "parquet". Defaults to "csv".
        overwrite: If True, overwrite existing files. If False, raise error if files exist.
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Raises:
        FileExistsError: If any file already exists and overwrite is False.
        ValueError: If an unsupported format is specified.
    """
    config = load_config(config_path)
    
    if output_dir is None:
        output_dir = config['constants']['data_root']
    
    output_path = Path(output_dir)
    
    # Get file extension from config
    extension = config['constants']['extensions'][format]
    
    print("=" * 60)
    print("Saving All Datasets")
    print("=" * 60)
    
    # Save each dataset
    save_teaser(df_teaser, str(output_path / f"teaser{extension}"), format, overwrite, config_path)
    save_train(df_train, str(output_path / f"train{extension}"), format, overwrite, config_path)
    save_test(df_test, str(output_path / f"test{extension}"), format, overwrite, config_path)
    
    print("=" * 60)
    print("✓ All datasets saved successfully")
    print("=" * 60)


def main() -> None:
    """Main function to demonstrate saving datasets."""
    # Import load functions
    try:
        from load_data import load_teaser, load_train, load_test
    except ImportError:
        print("Error: Could not import load_data module.")
        print("Make sure load_data.py exists in the same directory.")
        return
    
    print("=" * 60)
    print("OpenADMET ExpansionRx Challenge - Data Saver")
    print("=" * 60)
    
    try:
        # Load all datasets
        print("\n📊 Loading datasets...")
        df_teaser = load_teaser()
        df_train = load_train()
        df_test = load_test()
        print("✓ All datasets loaded successfully\n")
        
        # Save all datasets (using defaults from config)
        save_all_datasets(
            df_teaser,
            df_train,
            df_test,
            format="csv",
            overwrite=True
        )
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
