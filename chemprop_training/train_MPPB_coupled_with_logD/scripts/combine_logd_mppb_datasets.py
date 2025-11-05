#!/usr/bin/env python3
"""
Combine PharmaBench LogD and PPB datasets with OpenADMET MPPB data
to create a large training dataset for sequential prediction
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

def standardize_smiles(smiles):
    """Standardize SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

def combine_pharmabench_data():
    """Combine PharmaBench LogD and PPB datasets."""

    print("Loading PharmaBench datasets...")

    # Load LogD data
    logd_df = pd.read_csv('../PharmaBench/data/final_datasets/logd_reg_final_data.csv')
    print(f"PharmaBench LogD: {len(logd_df)} molecules")
    print(f"  Columns: {list(logd_df.columns)}")

    # Load PPB data (Plasma Protein Binding)
    ppb_df = pd.read_csv('../PharmaBench/data/final_datasets/ppb_reg_final_data.csv')
    print(f"PharmaBench PPB: {len(ppb_df)} molecules")
    print(f"  Columns: {list(ppb_df.columns)}")

    # Rename columns to standard names
    logd_df = logd_df.rename(columns={'Smiles_unify': 'smiles', 'value': 'y'})
    ppb_df = ppb_df.rename(columns={'Smiles_unify': 'smiles', 'value': 'y'})

    # Standardize SMILES in both datasets
    print("\nStandardizing SMILES...")
    logd_df['canonical_smiles'] = logd_df['smiles'].apply(standardize_smiles)
    ppb_df['canonical_smiles'] = ppb_df['smiles'].apply(standardize_smiles)

    # Remove invalid SMILES
    logd_df = logd_df[logd_df['canonical_smiles'].notna()]
    ppb_df = ppb_df[ppb_df['canonical_smiles'].notna()]

    print(f"After SMILES standardization:")
    print(f"  LogD: {len(logd_df)} molecules")
    print(f"  PPB: {len(ppb_df)} molecules")

    # First, let's check if we should match on original SMILES or canonical SMILES
    # Check how many molecules match using original SMILES
    original_match = pd.merge(
        logd_df[['smiles', 'y']].rename(columns={'y': 'LogD'}),
        ppb_df[['smiles', 'y']].rename(columns={'y': 'PPB'}),
        on='smiles',
        how='inner'
    )
    print(f"Molecules matching on original SMILES: {len(original_match)}")

    # Check how many match on canonical SMILES
    canonical_match = pd.merge(
        logd_df[['canonical_smiles', 'y']].rename(columns={'y': 'LogD'}),
        ppb_df[['canonical_smiles', 'y']].rename(columns={'y': 'PPB'}),
        on='canonical_smiles',
        how='inner'
    )
    print(f"Molecules matching on canonical SMILES: {len(canonical_match)}")

    # Use the best matching method
    if len(original_match) >= len(canonical_match):
        print("\nUsing original SMILES for matching...")
        merged_df = pd.merge(
            logd_df[['smiles', 'canonical_smiles', 'y']].rename(columns={'y': 'LogD'}),
            ppb_df[['smiles', 'y']].rename(columns={'y': 'PPB'}),
            on='smiles',
            how='inner'
        )
    else:
        print("\nUsing canonical SMILES for matching...")
        merged_df = pd.merge(
            logd_df[['canonical_smiles', 'smiles', 'y']].rename(columns={'y': 'LogD', 'smiles': 'original_smiles'}),
            ppb_df[['canonical_smiles', 'y']].rename(columns={'y': 'PPB'}),
            on='canonical_smiles',
            how='inner'
        )

    print(f"Merged dataset: {len(merged_df)} molecules with both LogD and PPB")

    # PPB is given as percentage bound (0-100), MPPB is given as percentage unbound (0-100)
    # MPPB = 100 - PPB
    # However, let's check the scale first
    print(f"\nPPB value statistics from PharmaBench:")
    print(f"  Min: {merged_df['PPB'].min():.2f}")
    print(f"  Max: {merged_df['PPB'].max():.2f}")
    print(f"  Mean: {merged_df['PPB'].mean():.2f}")

    # If PPB values are in fraction (0-1) scale, convert to percentage first
    if merged_df['PPB'].max() <= 1.0:
        print("  PPB appears to be in fraction scale (0-1), converting to percentage...")
        merged_df['PPB'] = merged_df['PPB'] * 100

    # Now convert PPB (% bound) to MPPB (% unbound)
    merged_df['MPPB'] = 100 - merged_df['PPB']

    print(f"\nConverted MPPB statistics:")
    print(f"  Min: {merged_df['MPPB'].min():.2f}")
    print(f"  Max: {merged_df['MPPB'].max():.2f}")
    print(f"  Mean: {merged_df['MPPB'].mean():.2f}")

    # Create molecule names (generated IDs)
    merged_df['Molecule Name'] = ['PB-' + str(i).zfill(7) for i in range(len(merged_df))]

    # Rename columns to match format
    pharmabench_combined = merged_df[['Molecule Name', 'canonical_smiles', 'LogD', 'MPPB']].rename(
        columns={'canonical_smiles': 'SMILES'}
    )

    print(f"\nPharmaBench combined dataset:")
    print(f"  Shape: {pharmabench_combined.shape}")
    print(f"  LogD range: [{pharmabench_combined['LogD'].min():.2f}, {pharmabench_combined['LogD'].max():.2f}]")
    print(f"  MPPB range: [{pharmabench_combined['MPPB'].min():.2f}, {pharmabench_combined['MPPB'].max():.2f}]")

    return pharmabench_combined

def combine_with_openadmet():
    """Combine PharmaBench data with OpenADMET paired data."""

    # Get PharmaBench combined data
    pharmabench_df = combine_pharmabench_data()

    # Load OpenADMET paired data
    print("\n" + "="*60)
    print("Loading OpenADMET paired data...")
    openadmet_df = pd.read_csv('../Data/paired_endpoints/train_LogD_MPPB.csv')
    print(f"OpenADMET LogD-MPPB: {len(openadmet_df)} molecules")
    print(f"  Columns: {list(openadmet_df.columns)}")

    # Check for overlapping molecules
    print("\nChecking for overlapping molecules...")
    openadmet_df['canonical_smiles'] = openadmet_df['SMILES'].apply(standardize_smiles)
    pharmabench_df['canonical_smiles'] = pharmabench_df['SMILES'].apply(standardize_smiles)

    overlap = set(openadmet_df['canonical_smiles']) & set(pharmabench_df['canonical_smiles'])
    print(f"Overlapping molecules: {len(overlap)}")

    if len(overlap) > 0:
        # Remove overlapping molecules from PharmaBench to avoid duplicates
        print("Removing overlapping molecules from PharmaBench...")
        pharmabench_df = pharmabench_df[~pharmabench_df['canonical_smiles'].isin(overlap)]
        print(f"PharmaBench after deduplication: {len(pharmabench_df)} molecules")

    # Combine datasets
    print("\nCombining datasets...")
    # Keep original SMILES column for final dataset
    combined_df = pd.concat([
        openadmet_df[['Molecule Name', 'SMILES', 'LogD', 'MPPB']],
        pharmabench_df[['Molecule Name', 'SMILES', 'LogD', 'MPPB']]
    ], ignore_index=True)

    print(f"\n" + "="*60)
    print("COMBINED DATASET STATISTICS")
    print("="*60)
    print(f"Total molecules: {len(combined_df)}")
    print(f"  From OpenADMET: {len(openadmet_df)}")
    print(f"  From PharmaBench: {len(pharmabench_df)}")

    # Check for NaN values
    print(f"\nMissing values:")
    print(f"  LogD: {combined_df['LogD'].isna().sum()}")
    print(f"  MPPB: {combined_df['MPPB'].isna().sum()}")

    # Remove any rows with NaN
    combined_df = combined_df.dropna()
    print(f"After removing NaN: {len(combined_df)} molecules")

    # Statistics
    print(f"\nValue ranges:")
    print(f"  LogD: [{combined_df['LogD'].min():.2f}, {combined_df['LogD'].max():.2f}]")
    print(f"  MPPB: [{combined_df['MPPB'].min():.2f}, {combined_df['MPPB'].max():.2f}]")

    print(f"\nDistribution statistics:")
    print(f"  LogD: mean={combined_df['LogD'].mean():.2f}, std={combined_df['LogD'].std():.2f}")
    print(f"  MPPB: mean={combined_df['MPPB'].mean():.2f}, std={combined_df['MPPB'].std():.2f}")

    # Calculate correlation
    correlation = combined_df['LogD'].corr(combined_df['MPPB'])
    print(f"\nCorrelation between LogD and MPPB: {correlation:.4f}")

    # Save combined dataset
    output_dir = Path('../Data/endpoints')
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'train_big_LogD_MPPB.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined dataset to: {output_file}")

    # Also save a version with canonical SMILES for reference
    combined_df['canonical_smiles'] = combined_df['SMILES'].apply(standardize_smiles)
    stats_file = output_dir / 'train_big_LogD_MPPB_stats.csv'

    # Create summary statistics
    stats_df = pd.DataFrame({
        'Dataset': ['OpenADMET', 'PharmaBench', 'Combined'],
        'N_molecules': [len(openadmet_df), len(pharmabench_df), len(combined_df)],
        'LogD_mean': [
            openadmet_df['LogD'].mean(),
            pharmabench_df['LogD'].mean() if len(pharmabench_df) > 0 else 0,
            combined_df['LogD'].mean()
        ],
        'LogD_std': [
            openadmet_df['LogD'].std(),
            pharmabench_df['LogD'].std() if len(pharmabench_df) > 0 else 0,
            combined_df['LogD'].std()
        ],
        'MPPB_mean': [
            openadmet_df['MPPB'].mean(),
            pharmabench_df['MPPB'].mean() if len(pharmabench_df) > 0 else 0,
            combined_df['MPPB'].mean()
        ],
        'MPPB_std': [
            openadmet_df['MPPB'].std(),
            pharmabench_df['MPPB'].std() if len(pharmabench_df) > 0 else 0,
            combined_df['MPPB'].std()
        ]
    })

    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics to: {stats_file}")

    return combined_df

def check_pharmabench_structure():
    """Quick check of PharmaBench data structure."""

    print("Checking PharmaBench data structure...")
    print("\nLogD dataset:")
    logd_df = pd.read_csv('../PharmaBench/data/final_datasets/logd_reg_final_data.csv', nrows=5)
    print(f"  Columns: {list(logd_df.columns)}")
    print(f"  Sample data:")
    print(logd_df.head(3))

    print("\nPPB dataset:")
    ppb_df = pd.read_csv('../PharmaBench/data/final_datasets/ppb_reg_final_data.csv', nrows=5)
    print(f"  Columns: {list(ppb_df.columns)}")
    print(f"  Sample data:")
    print(ppb_df.head(3))

def main():
    # First check data structure
    print("="*60)
    print("CHECKING DATA STRUCTURE")
    print("="*60)
    check_pharmabench_structure()

    # Combine datasets
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)
    combined_df = combine_with_openadmet()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✓ Successfully created train_big_LogD_MPPB.csv")
    print(f"✓ Total molecules: {len(combined_df)}")
    print(f"✓ Ready for sequential training with LogD as input feature")

    # Show sample of combined data
    print("\nSample of combined dataset:")
    print(combined_df.head())

if __name__ == "__main__":
    main()