import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv")

print(df.head())

# Count missing values
print('This is how many missing values are in the dataset')
print(df.isnull().sum())

# Count how many molecules have each property filled in
print('This is how many molecules have each property filled in')
print(df.count())

# Summarize numeric columns (mean, std, min, max)
print(df.describe())


# Convert SMILES to molecule objects
mols = [Chem.MolFromSmiles(s) for s in df['SMILES'][:5]]

# Draw molecules
Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200,200))