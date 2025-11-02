import pandas as pd
import os
from pathlib import Path

def load_pharmabench_datasets(datasets_dir="final_datasets"):
    """
    Load all PharmaBench datasets and convert to a format similar to expansion_data_train_raw.csv
    
    Args:
        datasets_dir: Directory containing PharmaBench datasets
        
    Returns:
        DataFrame in the format similar to expansion_data_train_raw.csv
    """
    # Define property mapping from PharmaBench to expansion_data format
    property_mapping = {
        'logd': 'LogD',
        'human_mic_cl': 'HLM CLint',
        'mouse_mic_cl': 'MLM CLint',
    }
    
    # List all CSV files in the datasets directory
    dataset_files = list(Path(datasets_dir).glob("*_final_data.csv"))
    print(f"Found {len(dataset_files)} PharmaBench datasets")
    
    # Create a dictionary to store data for each SMILES
    smiles_dict = {}
    
    # Process each dataset
    for file_path in dataset_files:
        # Load dataset
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            print(f"Skipping empty file: {file_path.name}")
            continue
            
        # Get the property name from the data itself
        if 'property' in df.columns:
            # Use the first property value in the dataset
            property_name = df['property'].iloc[0]
            
            # Skip if property not in mapping
            if property_name not in property_mapping:
                print(f"Skipping {file_path.name} - property {property_name} not in mapping")
                continue
                
            # Map property to expansion_data format
            target_property = property_mapping[property_name]
            print(f"Processing {file_path.name}: {len(df)} compounds for {target_property}")
        else:
            print(f"Skipping {file_path.name} - no property column found")
            continue
        
        # Process each row
        for _, row in df.iterrows():
            smiles = row['Smiles_unify']
            value = row['value']
            
            # Initialize dict for this SMILES if not exists
            if smiles not in smiles_dict:
                smiles_dict[smiles] = {
                    'SMILES': smiles,
                    'Molecule Name': f"PB-{len(smiles_dict) + 1:06d}"  # Generate molecule name
                }
            
            # Add property value
            smiles_dict[smiles][target_property] = value
            
            # Add modifier (all values are exact in PharmaBench)
            smiles_dict[smiles][f"{target_property} modifier"] = '='
    
    # Convert dictionary to DataFrame
    result_df = pd.DataFrame.from_dict(smiles_dict, orient='index').reset_index(drop=True)
    
    # Add INCHIKEY column (empty for now, would need RDKit to generate)
    result_df['INCHIKEY'] = ''
    
    # Ensure all original columns are present (even if empty)
    original_columns = [
        'Molecule Name', 'SMILES', 'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
        'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
        'MPPB', 'MBPB', 'MGMB',
        'LogD modifier', 'KSOL modifier', 'HLM CLint modifier', 'MLM CLint modifier',
        'Caco-2 Permeability Papp A>B modifier', 'Caco-2 Permeability Efflux modifier',
        'MPPB modifier', 'MBPB modifier', 'MGMB modifier', 'INCHIKEY'
    ]
    
    # Add PharmaBench-specific columns
    # Remove the code that adds PharmaBench-specific columns
    # Delete lines 91-96 completely:
    # Add PharmaBench-specific columns
    for prop in ['RLMC', 'PPB', 'CYP 2C9', 'CYP 2D6', 'CYP 3A4', 'BBB', 'AMES']:
        if prop not in result_df.columns:
            result_df[prop] = None
        if f"{prop} modifier" not in result_df.columns:
            result_df[f"{prop} modifier"] = None
    
    # Replace the column reordering code to use only the original columns
    # Replace lines 98-115 with:
    # Ensure all columns from expansion_data are present
    for col in original_columns:
        if col not in result_df.columns:
            result_df[col] = ''
    
    # Reorder columns to match expansion_data_train_raw.csv exactly
    result_df = result_df[original_columns]
    
    # Replace NaN with empty strings to match expansion_data format
    result_df = result_df.fillna('')
    
    return result_df

def save_combined_dataset(pharmabench_df, expansion_df_path, output_path):
    """
    Combine PharmaBench data with existing expansion data
    
    Args:
        pharmabench_df: DataFrame with PharmaBench data
        expansion_df_path: Path to expansion_data_train_raw.csv
        output_path: Path to save combined dataset
    """
    # Load expansion data
    expansion_df = pd.read_csv(expansion_df_path)
    print(f"Loaded expansion data: {len(expansion_df)} compounds")
    
    # Combine datasets
    combined_df = pd.concat([expansion_df, pharmabench_df], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} compounds")
    
    # Save combined dataset
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")
    
    return combined_df

if __name__ == "__main__":
    # Load PharmaBench datasets
    pharmabench_df = load_pharmabench_datasets()
    
    # Print summary
    print("\nPharmaBench Dataset Summary:")
    print(f"Total compounds: {len(pharmabench_df)}")
    
    # Count non-empty values for each property
    for col in pharmabench_df.columns:
        if col not in ['Molecule Name', 'SMILES', 'INCHIKEY'] and not col.endswith(' modifier'):
            non_empty = (pharmabench_df[col] != '').sum()
            if non_empty > 0:
                print(f"{col}: {non_empty} values")
    
    # Save PharmaBench data in expansion_data format
    pharmabench_df.to_csv("pharmabench_train_raw.csv", index=False)
    print(f"Saved PharmaBench data to pharmabench_train_raw.csv")
    
    # Optionally combine with expansion data
    # expansion_path = "../models/CheMeleon/expansion_data_train_raw.csv"
    # if os.path.exists(expansion_path):
    #     combined_df = save_combined_dataset(
    #         pharmabench_df, 
    #         expansion_path, 
    #         "combined_train_raw.csv"
    #     )