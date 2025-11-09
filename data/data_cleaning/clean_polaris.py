import pandas as pd
import re

def clean_cxsmiles(cxsmiles):
    """Extract basic SMILES from CXSMILES by removing the extended part"""
    # Remove everything after the first space or pipe
    basic_smiles = re.split(r'[ |]', cxsmiles)[0]
    return basic_smiles

# Load the polaris data
polaris_df = pd.read_csv("polaris_data.csv")

# Create a new DataFrame with the desired structure
result_df = pd.DataFrame()

# Map columns
result_df['Molecule Name'] = polaris_df['Molecule Name']
result_df['SMILES'] = polaris_df['CXSMILES'].apply(clean_cxsmiles)
result_df['LogD'] = polaris_df['LogD']
result_df['KSOL'] = polaris_df['KSOL']
result_df['HLM CLint'] = polaris_df['HLM']
result_df['MLM CLint'] = polaris_df['MLM']

# Add MDR1-MDCKII as Caco-2 Permeability Papp A>B (closest match)
result_df['Caco-2 Permeability Papp A>B'] = ''

# Initialize other columns from expansion_data_train_raw.csv with empty values
result_df['Caco-2 Permeability Efflux'] = ''
result_df['MPPB'] = ''
result_df['MBPB'] = ''
result_df['MGMB'] = ''

# Add modifier columns (assuming all values are exact measurements)
for col in ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 
            'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
            'MPPB', 'MBPB', 'MGMB']:
    result_df[f'{col} modifier'] = ''
    # Set modifier to '=' where values exist
    result_df.loc[result_df[col].notna() & (result_df[col] != ''), f'{col} modifier'] = '='

# Add INCHIKEY column (empty as it's not in the source data)
result_df['INCHIKEY'] = ''

# Filter to keep only training set if needed
# result_df = result_df[polaris_df['Set'] == 'Train']

# Save the converted data
result_df.to_csv("polaris_data_converted.csv", index=False)

print(f"Converted {len(result_df)} compounds to expansion_data format")