# load and run inference using the trained models
import sys
from datetime import datetime

import pandas as pd
import joblib
from pathlib import Path

from common import clean_smiles
from tqdm import tqdm

if __name__ == "__main__":
    try:
        model_dir = Path(sys.argv[1])
    except:
        print("Usage: python inference.py <model_directory>")
        exit(1)

    data_cache_f = Path("expansion_data_test_blinded.csv")
    if data_cache_f.exists():
        df = pd.read_csv(data_cache_f)
    else:
        df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv")
        df.to_csv(data_cache_f, index=False)

    test_smiles = list(map(clean_smiles, df["SMILES"]))

    out_data = {
        "Molecule Name": df["Molecule Name"],
        "SMILES": df["SMILES"],  # original smiles for output, cleaned ones for inference
    }

    # targets = list(model_dir.glob("Log*"))
    targets = [d for d in model_dir.iterdir() if d.is_dir() and d.name != "cache.db"]
    pbar = tqdm(total=len(targets))
    for target in targets:
        target_name = target.stem.replace("_", " ")
        pbar.set_description(f"Predicting '{target_name}'")
        pipe = joblib.load(target / "final_model.joblib")
        pred = pipe.predict(test_smiles)
        
        # Apply appropriate inverse transformations based on check_dataset_scale.ipynb
        if target_name == "LogD":
            # LogD wasn't transformed (log_scale=False)
            pass
        elif target_name == "KSOL" or target_name == "Caco-2 Permeability Papp A>B":
            # These had log_scale=True and multiplier=1e-6
            # Original transformation: log10((x+1)*1e-6)
            # Inverse: (10^pred)/1e-6 - 1
            pred = (10**pred) / 1e-6 - 1
        else:
            # All other metrics had log_scale=True and multiplier=1
            # Original transformation: log10(x+1)
            # Inverse: 10^pred - 1
            pred = 10**pred - 1
        
        out_data[target_name] = pred
        pbar.update(1)
    pbar.close()

    # timestamped result file
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(
        model_dir / f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )