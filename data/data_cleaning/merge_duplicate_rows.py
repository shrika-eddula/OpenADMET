import pandas as pd
import numpy as np

# Load in the data
data = pd.read_csv("Combined_Log10_Transformed_Canon_SMILES_ML_Ready.csv")

# Identify all the duplicated rows
duplicates = data[data.duplicated(subset=["SMILES"], keep=False)]
print(f"Total duplicate rows found: {duplicates.shape[0]}")

# Separate them from the rest of the dataframe
uniques = data[~data["SMILES"].duplicated(keep=False)]
print(f"Total unique rows found (excluding any duplicate rows): {uniques.shape[0]}")

# For each unique SMILE in the duplicates
duplicate_smiles = duplicates["SMILES"].unique().tolist()
print(f"Total unique duplicate SMILES found: {len(duplicate_smiles)}")

for smi in duplicate_smiles:
    # Grab all rows with that unique SMILE
    curr_smi = duplicates[duplicates["SMILES"] == smi]
    curr_dict = {}

    # Merge all values in that subset of rows
    for col in curr_smi.columns:

        curr_dict["SMILES"] = curr_smi.iloc[0]["SMILES"]
        curr_dict["Molecule Name"] = curr_smi.iloc[0]["Molecule Name"]

        if col == "SMILES" or col == "Molecule Name":
            continue
        else:
            values = curr_smi[col].dropna().tolist()

            # If there are numerical values:
                # Replace with average
            if values:
                avg = np.average(values)
            # If the values are all NAN:
                # Save as NAN
            else:
                avg = None

            curr_dict[col] = avg

    curr_df = pd.DataFrame([curr_dict])
    print(f"New merged row for SMILE {smi}: /n {curr_df}")

    # Concatenate the new, unique rows with the rest of the dataframe
    pd.concat([uniques, curr_df], ignore_index=True)
    print(f"Row merged with the final dataset")

# Save 
output_filepath = "Combined_Log10_Transformed_Canon_SMILES_Duplicates_Merged.csv"
uniques.to_csv(output_filepath, index=False)
print(f"Merged dataset saved at {output_filepath}. Merging complete.")