import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

# Load data
input_filepath = "data/datasets/FINAL_DATA.csv"
df = pd.read_csv(input_filepath)
print(f"Data successfully loaded from {input_filepath}")

# Function to generate Murcko Scaffold SMILES from row of ADMET data
def produce_scaffold(row):
    """
    Takes as input a row from df.iterrows(), isolates the SMILE, and returns the MurckoScaffoldSMILE.
    
    Args:
        row: return value of df.iterrows(), a tuple of the df index and pandas Series

    Returns:
        a numpy array, sized (2048,) of the CheMeleon fingerprint
    """
    curr_smile = row["SMILES"]
    return MurckoScaffoldSmilesFromSmiles(curr_smile, includeChirality=True)
    
# Add the representation column to the dataframe
df["Murcko Scaffold"] = df.apply(lambda row: produce_scaffold(row), axis=1)

# Save scaffolds+df as a new CSV file, append embeddings to the existing ADMET data
output_filepath = "data/data_splitting/combined_data_with_murcko_scaffolds_chiral_specific.csv"
df.to_csv(output_filepath)

print(f"Dataframe with representations saved to {output_filepath}")