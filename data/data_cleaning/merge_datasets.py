import pandas as pd

# Load the three datasets
expansion_data = pd.read_csv('/home/jeanshe/orcd/pool/OpenADMET/Data/splits/Expansion_Log10_Transformed.csv')
polaris_data = pd.read_csv('/home/jeanshe/orcd/pool/OpenADMET/Data/splits/Polaris_Log10_Transformed.csv')
pharmabench_data = pd.read_csv('/home/jeanshe/orcd/pool/OpenADMET/Data/pharmabench_train_raw.csv')

# Concatenate all datasets
# combined_data = pd.concat([expansion_data, polaris_data, pharmabench_data], ignore_index=True)
combined_data = pd.concat([expansion_data, pharmabench_data], ignore_index=True)

# Remove duplicates based on SMILES (if a molecule appears in multiple datasets)
# Keep the first occurrence (priority: Expansion > Polaris > PharmaBench)
combined_data = combined_data.drop_duplicates(subset=['SMILES'], keep='first')

# Save the combined dataset
combined_data.to_csv('/home/jeanshe/orcd/pool/OpenADMET/Data/splits/Combined_Log10_Transformed_Canon_SMILES.csv', index=False)

# Print summary statistics
print(f"Total compounds in combined dataset: {len(combined_data)}")
print(f"Compounds from Expansion data: {len(expansion_data)}")
print(f"Compounds from Polaris data: {len(polaris_data)}")
print(f"Compounds from PharmaBench data: {len(pharmabench_data)}")

# Count all properties (excluding identifiers and modifiers)
print("\nProperty coverage in combined dataset:")
property_columns = [col for col in combined_data.columns 
                   if col not in ['Molecule Name', 'SMILES', 'INCHIKEY'] 
                   and not col.endswith('modifier')]

for col in property_columns:
    count = combined_data[col].notna().sum()
    print(f"{col}: {count} values")