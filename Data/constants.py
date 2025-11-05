"""
Constants for OpenADMET ExpansionRx Challenge.

This module centralizes path constants used across the data loading and saving modules.
"""

from pathlib import Path

# Configuration file path
# Relative to the Data directory, the config file is in the parent directory
CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")
