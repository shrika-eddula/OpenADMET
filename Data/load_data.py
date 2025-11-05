"""
Data loading utilities for OpenADMET ExpansionRx Challenge.

This module provides functions to load training, test, and teaser datasets
from HuggingFace using URLs defined in the configuration file.
"""

from pathlib import Path
from typing import Optional

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


def load_teaser(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Load teaser dataset for initial exploration.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        DataFrame containing the teaser dataset.
        
    Raises:
        KeyError: If the teaser URL is not found in the config.
        Exception: If the data cannot be loaded from HuggingFace.
    """
    config = load_config(config_path)
    teaser_url = config['data_links']['teaser_url']
    teaser_csv = config['constants']['teaser_csv']
    
    # Construct the full path to the CSV file
    csv_path = f"{teaser_url}/{teaser_csv}"
    df = pd.read_csv(csv_path)
    
    return df


def load_train(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Load training dataset with labeled examples.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        DataFrame containing the training dataset.
        
    Raises:
        KeyError: If the train URL is not found in the config.
        Exception: If the data cannot be loaded from HuggingFace.
    """
    config = load_config(config_path)
    train_url = config['data_links']['train_url']
    train_csv = config['constants']['train_csv']
    
    # Construct the full path to the CSV file
    csv_path = f"{train_url}/{train_csv}"
    df = pd.read_csv(csv_path)
    
    return df


def load_test(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Load test dataset with blinded labels for evaluation.
    
    Args:
        config_path: Path to the configuration file. Defaults to "config.yaml".
        
    Returns:
        DataFrame containing the test dataset.
        
    Raises:
        KeyError: If the test URL is not found in the config.
        Exception: If the data cannot be loaded from HuggingFace.
    """
    config = load_config(config_path)
    test_url = config['data_links']['test_url']
    test_csv = config['constants']['test_csv']
    
    # Construct the full path to the CSV file
    csv_path = f"{test_url}/{test_csv}"
    df = pd.read_csv(csv_path)
    
    return df


def main() -> None:
    """Main function to demonstrate loading all datasets."""
    print("=" * 60)
    print("OpenADMET ExpansionRx Challenge - Data Loader")
    print("=" * 60)
    
    # Load teaser data
    print("\n📊 Loading teaser data...")
    try:
        df_teaser = load_teaser()
        print(f"✓ Teaser data loaded successfully")
        print(f"  Shape: {df_teaser.shape}")
        print(f"  Columns: {list(df_teaser.columns)}")
        print(f"\nFirst few rows:")
        print(df_teaser.head())
    except Exception as e:
        print(f"✗ Error loading teaser data: {e}")
    
    # Load training data
    print("\n📊 Loading training data...")
    try:
        df_train = load_train()
        print(f"✓ Training data loaded successfully")
        print(f"  Shape: {df_train.shape}")
        print(f"  Columns: {list(df_train.columns)}")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
    
    # Load test data
    print("\n📊 Loading test data...")
    try:
        df_test = load_test()
        print(f"✓ Test data loaded successfully")
        print(f"  Shape: {df_test.shape}")
        print(f"  Columns: {list(df_test.columns)}")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
