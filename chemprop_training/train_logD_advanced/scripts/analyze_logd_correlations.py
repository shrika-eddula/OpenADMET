#!/usr/bin/env python3
"""
Analyze correlations between LogD and other ADMET endpoints
to determine optimal sequential prediction order
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_paired_datasets(train_df):
    """Create datasets for molecules with LogD and each other endpoint."""

    # Define all endpoints (excluding LogD)
    endpoints = ['KSOL', 'HLM CLint', 'MLM CLint',
                'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
                'MPPB', 'MBPB', 'MGMB']

    paired_datasets = {}

    for endpoint in endpoints:
        # Get molecules with both LogD and this endpoint
        mask = train_df['LogD'].notna() & train_df[endpoint].notna()
        paired_df = train_df[mask][['Molecule Name', 'SMILES', 'LogD', endpoint]]
        paired_datasets[endpoint] = paired_df

    return paired_datasets

def analyze_correlations(paired_datasets):
    """Calculate correlation metrics for each endpoint pair with LogD."""

    correlation_results = []

    for endpoint, df in paired_datasets.items():
        if len(df) > 1:
            logd_values = df['LogD'].values
            endpoint_values = df[endpoint].values

            # Calculate correlations
            pearson_r, pearson_p = pearsonr(logd_values, endpoint_values)
            spearman_r, spearman_p = spearmanr(logd_values, endpoint_values)

            # Calculate R-squared
            r_squared = pearson_r ** 2

            # Store results
            correlation_results.append({
                'Endpoint': endpoint,
                'N_samples': len(df),
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'R_squared': r_squared,
                'LogD_mean': logd_values.mean(),
                'LogD_std': logd_values.std(),
                'Endpoint_mean': endpoint_values.mean(),
                'Endpoint_std': endpoint_values.std()
            })

    return pd.DataFrame(correlation_results)

def create_scatter_plots(paired_datasets, output_dir='correlation_plots'):
    """Create scatter plots for each endpoint vs LogD."""

    Path(output_dir).mkdir(exist_ok=True)

    # Create a figure with subplots
    n_endpoints = len(paired_datasets)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (endpoint, df) in enumerate(paired_datasets.items()):
        if len(df) > 1:
            ax = axes[idx]

            # Create scatter plot
            ax.scatter(df['LogD'], df[endpoint], alpha=0.5, s=10)

            # Add trend line
            z = np.polyfit(df['LogD'].values, df[endpoint].values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['LogD'].min(), df['LogD'].max(), 100)
            ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)

            # Calculate correlation
            r, _ = pearsonr(df['LogD'].values, df[endpoint].values)

            # Labels and title
            ax.set_xlabel('LogD')
            ax.set_ylabel(endpoint)
            ax.set_title(f'{endpoint}\n(r = {r:.3f}, n = {len(df)})')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/logd_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Scatter plots saved to {output_dir}/logd_correlations.png")

def save_paired_datasets(paired_datasets, output_dir='../Data/paired_endpoints'):
    """Save paired datasets to CSV files."""

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for endpoint, df in paired_datasets.items():
        # Clean endpoint name for filename
        clean_name = endpoint.replace(' ', '_').replace('-', '_')
        output_file = f"{output_dir}/train_LogD_{clean_name}.csv"

        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} samples to {output_file}")

def main():
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('../Data/train.csv')
    print(f"Total training samples: {len(train_df)}")

    # Check LogD availability
    logd_available = train_df['LogD'].notna().sum()
    print(f"Samples with LogD: {logd_available}")

    # Create paired datasets
    print("\n" + "="*60)
    print("CREATING PAIRED DATASETS")
    print("="*60)
    paired_datasets = create_paired_datasets(train_df)

    # Analyze correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS: LogD vs Other Endpoints")
    print("="*60)

    correlation_df = analyze_correlations(paired_datasets)

    # Sort by absolute Pearson correlation
    correlation_df['Abs_Pearson'] = correlation_df['Pearson_r'].abs()
    correlation_df = correlation_df.sort_values('Abs_Pearson', ascending=False)

    # Display results
    print("\nCorrelation Summary (sorted by |Pearson r|):")
    print("-"*60)

    for _, row in correlation_df.iterrows():
        print(f"\n{row['Endpoint']}:")
        print(f"  Samples: {row['N_samples']:,}")
        print(f"  Pearson r: {row['Pearson_r']:.4f} (p={row['Pearson_p']:.2e})")
        print(f"  Spearman r: {row['Spearman_r']:.4f} (p={row['Spearman_p']:.2e})")
        print(f"  R²: {row['R_squared']:.4f}")

        # Interpret correlation strength
        abs_r = abs(row['Pearson_r'])
        if abs_r > 0.7:
            strength = "STRONG"
        elif abs_r > 0.5:
            strength = "MODERATE"
        elif abs_r > 0.3:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"

        direction = "POSITIVE" if row['Pearson_r'] > 0 else "NEGATIVE"
        print(f"  Interpretation: {strength} {direction} correlation")

    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_scatter_plots(paired_datasets)

    # Create correlation heatmap
    correlation_matrix = pd.DataFrame(index=['LogD'],
                                     columns=correlation_df['Endpoint'].values)
    correlation_matrix.loc['LogD'] = correlation_df['Pearson_r'].values

    plt.figure(figsize=(10, 2))
    sns.heatmap(correlation_matrix.astype(float), annot=True, fmt='.3f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title('LogD Correlations with ADMET Endpoints')
    plt.tight_layout()
    plt.savefig('correlation_plots/logd_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Heatmap saved to correlation_plots/logd_correlation_heatmap.png")

    # Save paired datasets
    print("\n" + "="*60)
    print("SAVING PAIRED DATASETS")
    print("="*60)
    save_paired_datasets(paired_datasets)

    # Save correlation results
    correlation_df.to_csv('logd_correlation_analysis.csv', index=False)
    print(f"\nCorrelation analysis saved to logd_correlation_analysis.csv")

    # Recommendation for sequential prediction
    print("\n" + "="*60)
    print("RECOMMENDATION FOR SEQUENTIAL PREDICTION")
    print("="*60)

    best_endpoint = correlation_df.iloc[0]
    print(f"\nBest correlated endpoint: {best_endpoint['Endpoint']}")
    print(f"Pearson correlation: {best_endpoint['Pearson_r']:.4f}")
    print(f"R-squared: {best_endpoint['R_squared']:.4f}")
    print(f"Sample size: {best_endpoint['N_samples']:,}")

    print("\nSuggested sequential prediction order (by correlation strength):")
    for i, row in correlation_df.iterrows():
        print(f"  {i+1}. {row['Endpoint']:<40} (r = {row['Pearson_r']:+.4f})")

    # Physical interpretation
    print("\n" + "="*60)
    print("PHYSICAL INTERPRETATION")
    print("="*60)
    print("\nLogD (lipophilicity at pH 7.4) is expected to correlate with:")
    print("  • KSOL (solubility) - NEGATIVE correlation expected")
    print("  • Permeability (Caco-2) - POSITIVE correlation expected")
    print("  • Protein binding (MPPB/MBPB/MGMB) - POSITIVE correlation expected")
    print("  • Metabolic clearance (HLM/MLM) - Complex relationship")

    # Check if correlations match expectations
    for _, row in correlation_df.iterrows():
        endpoint = row['Endpoint']
        r = row['Pearson_r']

        if 'KSOL' in endpoint and r < 0:
            print(f"\n✓ {endpoint}: Negative correlation ({r:.3f}) matches expectation")
        elif 'Permeability' in endpoint and 'Efflux' not in endpoint and r > 0:
            print(f"\n✓ {endpoint}: Positive correlation ({r:.3f}) matches expectation")
        elif any(x in endpoint for x in ['MPPB', 'MBPB', 'MGMB']) and r > 0:
            print(f"\n✓ {endpoint}: Positive correlation ({r:.3f}) matches expectation")

if __name__ == "__main__":
    main()