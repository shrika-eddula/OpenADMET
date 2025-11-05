#!/usr/bin/env python3
"""
Update LogD predictions in the all endpoints file with advanced model predictions
"""

import pandas as pd
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Update LogD predictions with advanced model results')
    parser.add_argument('--input_file', type=str,
                        default='../predictions/all_endpoints_predictions_20251029_145847.csv',
                        help='Input predictions file with all endpoints')
    parser.add_argument('--logd_file', type=str,
                        default='logd_medium_results/predictions/test_predictions_with_uncertainty.csv',
                        help='Advanced LogD predictions file')
    parser.add_argument('--output_file', type=str,
                        default=None,
                        help='Output file path (if not specified, creates new with timestamp)')

    args = parser.parse_args()

    # Load the original predictions
    print(f"Loading original predictions from: {args.input_file}")
    original_df = pd.read_csv(args.input_file)
    print(f"  Shape: {original_df.shape}")
    print(f"  Columns: {list(original_df.columns)}")

    # Load the advanced LogD predictions
    print(f"\nLoading advanced LogD predictions from: {args.logd_file}")
    logd_df = pd.read_csv(args.logd_file)
    print(f"  Shape: {logd_df.shape}")
    print(f"  Columns: {list(logd_df.columns)}")

    # Create a copy of original dataframe
    updated_df = original_df.copy()

    # Check if LogD column exists
    if 'LogD' not in updated_df.columns:
        print("Warning: LogD column not found in original file")
        updated_df['LogD'] = None

    # Store original LogD values for comparison
    original_logd = updated_df['LogD'].copy()

    # Merge on Molecule Name
    print("\nMerging predictions...")

    # Create mapping from molecule name to LogD prediction
    logd_mapping = dict(zip(logd_df['Molecule Name'], logd_df['LogD_pred']))

    # Update LogD values
    updated_count = 0
    not_found = []

    for idx, row in updated_df.iterrows():
        mol_name = row['Molecule Name']
        if mol_name in logd_mapping:
            updated_df.at[idx, 'LogD'] = logd_mapping[mol_name]
            updated_count += 1
        else:
            not_found.append(mol_name)

    print(f"  Updated {updated_count} LogD values")
    if not_found:
        print(f"  Warning: {len(not_found)} molecules not found in LogD predictions")
        if len(not_found) <= 10:
            print(f"    Missing: {not_found}")

    # Add uncertainty columns if desired (optional)
    # Default: don't add uncertainty columns to keep the same format
    add_uncertainty = False

    if add_uncertainty:
        print("Adding uncertainty columns...")
        # Create mappings for uncertainty metrics
        std_mapping = dict(zip(logd_df['Molecule Name'], logd_df['LogD_std']))
        ci_lower_mapping = dict(zip(logd_df['Molecule Name'], logd_df['LogD_ci_lower']))
        ci_upper_mapping = dict(zip(logd_df['Molecule Name'], logd_df['LogD_ci_upper']))

        # Add new columns
        updated_df['LogD_std'] = updated_df['Molecule Name'].map(std_mapping)
        updated_df['LogD_ci_lower'] = updated_df['Molecule Name'].map(ci_lower_mapping)
        updated_df['LogD_ci_upper'] = updated_df['Molecule Name'].map(ci_upper_mapping)

        # Reorder columns to put LogD-related columns together
        cols = list(updated_df.columns)
        logd_cols = ['LogD', 'LogD_std', 'LogD_ci_lower', 'LogD_ci_upper']
        other_cols = [c for c in cols if c not in logd_cols and c != 'Molecule Name' and c != 'SMILES']
        updated_df = updated_df[['Molecule Name', 'SMILES'] + logd_cols + other_cols]

    # Generate output filename if not specified
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../predictions/all_endpoints_predictions_advanced_logd_{timestamp}.csv"
    else:
        output_file = args.output_file

    # Save the updated predictions
    print(f"\nSaving updated predictions to: {output_file}")
    updated_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("UPDATE SUMMARY")
    print("="*60)

    # Compare old vs new LogD values if they existed
    if original_logd.notna().any():
        valid_comparison = updated_df[original_logd.notna()]
        if len(valid_comparison) > 0:
            print("\nLogD Value Changes (for molecules with previous predictions):")
            print(f"  Original mean: {original_logd.mean():.3f}")
            print(f"  Updated mean:  {updated_df['LogD'].mean():.3f}")
            print(f"  Original std:  {original_logd.std():.3f}")
            print(f"  Updated std:   {updated_df['LogD'].std():.3f}")

            # Show some examples of changes
            print("\n  Sample changes (first 5):")
            for i in range(min(5, len(valid_comparison))):
                idx = valid_comparison.index[i]
                mol_name = updated_df.loc[idx, 'Molecule Name']
                old_val = original_logd.loc[idx]
                new_val = updated_df.loc[idx, 'LogD']
                if pd.notna(old_val) and pd.notna(new_val):
                    print(f"    {mol_name}: {old_val:.3f} -> {new_val:.3f} (Δ = {new_val-old_val:+.3f})")

    print(f"\nFinal dataset shape: {updated_df.shape}")
    print(f"Output saved to: {output_file}")

    # Also save a "latest" version for easy access
    latest_file = "../predictions/all_endpoints_predictions_latest.csv"
    updated_df.to_csv(latest_file, index=False)
    print(f"Also saved as: {latest_file}")

if __name__ == "__main__":
    main()