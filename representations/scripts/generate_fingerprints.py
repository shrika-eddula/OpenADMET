# Script to generate embeddings for each of our SMILES data
# Imports
import pandas as pd
import numpy as np
import pickle
from chemeleon_fingerprint import CheMeleonFingerprint

# Initialize CheMeleon
chemeleon_fingerprint = CheMeleonFingerprint()
print(f"CheMeleon fingerprint successfully initialized")

# Load data
input_filepath = "data/data_splitting/unique_scaffolds.csv"
data = pd.read_csv(input_filepath)
print(f"Data successfully loaded from {input_filepath}")

# Function to produce CheMeleon fingerprint from row of ADMET data
def produce_fingerprint(row):
    """
    Takes as input a row from df.iterrows(), isolates the SMILE, and returns a CheMeleon fingerprint.
    
    Args:
        row: return value of df.iterrows(), a tuple of the df index and pandas Series

    Returns:
        a numpy array, sized (2048,) of the CheMeleon fingerprint
    """
    # curr_smile = row["SMILES"]
    curr_smile = row["Murcko Scaffold"]
    fingerprint = chemeleon_fingerprint([curr_smile])[0]
    # Ensure it's a numpy array and has the correct shape
    fingerprint = np.asarray(fingerprint, dtype=np.float32)
    if fingerprint.shape != (2048,):
        raise ValueError(f"Expected fingerprint shape (2048,), got {fingerprint.shape}")
    return fingerprint

# Add the representation column to the dataframe
data["CheMeleon Fingerprint"] = data.apply(lambda row: produce_fingerprint(row), axis=1)

# Save embeddings+df as a new PKL file, append embeddings to the existing ADMET data
# PKL stores the column of the df as numpy arrays, while .csv will convert to strings
output_pkl = "data/data_splitting/unique_scaffolds_chemeleon_fingerprints.pkl"
with open(output_pkl, "wb") as file:
    pickle.dump(data, file)

print(f"Dataframe with representations saved to {output_pkl}")

# Save embeddings as a NPY file
output_npy = "data/data_splitting/unique_scaffolds_chemeleon_fingerprints.npy"
fingerprint_array = data["CheMeleon Fingerprint"].to_numpy()
with open(output_npy, "wb") as file:
    np.save(fingerprint_array, file)

print(f"Numpy array with representations saved to {output_npy} with shape {output_npy.shape}")