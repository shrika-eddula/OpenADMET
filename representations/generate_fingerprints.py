# Script to generate embeddings for each of our SMILES data
# Imports
import pandas as pd
from chemeleon_fingerprint import CheMeleonFingerprint

# Initialize CheMeleon
chemeleon_fingerprint = CheMeleonFingerprint()
print(f"CheMeleon fingerprint successfully initialized")

# Load data
input_filepath = "data/datasets/FINAL_DATA.csv"
data = pd.read_csv(input_filepath)
print(f"Data successfully loaded from {input_filepath}")

# Function to produce CheMeleon fingerprint from row of ADMET data
def produce_fingerprint(row):
    """
    Takes as input a row from df.iterrows(), isolates the SMILE, and returns a CheMeleon fingerprint.
    
    Args:
        row: return value of df.iterrows(), a tuple of the df index and pandas Series

    Returns:
        a numpy array of the CheMeleon fingerprint
    """
    curr_smile = row["SMILES"]
    return chemeleon_fingerprint(curr_smile)

# Add the representation column to the dataframe
data["CheMeleon Fingerprint"] = data.apply(lambda row: produce_fingerprint(row), axis=1)

# Save embeddings as a new CSV file, append embeddings to the existing ADMET data
output_filepath = "representations/combined_data_with_chemeleon_fingerprints.csv"
data.to_csv(output_filepath)
print(f"Dataframe with representations saved to {output_filepath}")