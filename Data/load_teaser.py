# scripts/load_teaser.py
import pandas as pd

def load_teaser():
    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv")
    return df

if __name__ == "__main__":
    print('Loading teaser data...')
    df = load_teaser()
    print(df.head())
    print(df.shape)
