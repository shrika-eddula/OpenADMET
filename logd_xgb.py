# scripts/logd_xgb.py  (fixed)
import os
import numpy as np, pandas as pd, xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


# --- descriptor set (kept same) ---
DESCS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.FractionCSP3,
    Descriptors.RingCount,
]

# --- helpers ---
def smiles_to_mol(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol

def ecfp4(mol):
    fp = _morgan.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_feats(mol):
    vals = [f(mol) for f in DESCS]
    return np.array(vals, dtype=np.float32)

def scaffold_from_smiles(smiles):
    try:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return "NA"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else "NA"
    except Exception:
        return "NA"

# --- IO + standardize columns from teaser CSV ---
COLMAP = {
    "Molecule Name": "molecule_id",
    "SMILES": "smiles",
    "LogD": "LogD",
    "KSOL": "KSOL",
    "HLM CLint": "HLM_Clint",
    "MLM CLint": "MLM_Clint",
    "Caco-2 Permeability Papp A>B": "Caco2_Papp_A2B",
    "Caco-2 Permeability Efflux": "Caco2_Efflux_Ratio",
    "MPPB": "MPPB_Unbound",
    "MBPB": "MBPB_Unbound",
    "MGMB": "MGMB_Unbound",
}

def load_df():
    # You can also use huggingface_hub or datasets; this works since you already got a head printout.
    df = pd.read_csv(
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv"
    )
    # Standardize column names
    df = df.rename(columns=COLMAP)

    # Ensure LogD is numeric
    df["LogD"] = pd.to_numeric(df["LogD"], errors="coerce")

    # Select only rows with LogD present + the three columns we need
    df = df[df["LogD"].notna()][["molecule_id", "smiles", "LogD"]].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows with numeric LogD were found after coercion.")
    return df

def featurize(df):
    mols = [smiles_to_mol(s) for s in tqdm(df.smiles, desc="mol")]
    keep = [i for i, m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    ecfp = np.vstack([ecfp4(m) for m in tqdm(mols, desc="ecfp4")])
    desc = np.vstack([rdkit_feats(m) for m in tqdm(mols, desc="desc")])
    y = df["LogD"].values.astype(np.float32)

    # scale only the dense descriptor block; leave binary ECFP alone
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc)
    X = np.concatenate([ecfp, desc_scaled], axis=1)
    return df, X, y, scaler

def scaffold_split(df, test_size=0.2, seed=13):
    scafs = df.smiles.map(scaffold_from_smiles)
    rng = np.random.default_rng(seed)
    uniq = pd.Series(scafs).unique().tolist()
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_size))
    test_scafs = set(uniq[:n_test])
    test_idx = [i for i, s in enumerate(scafs) if s in test_scafs]
    train_idx = [i for i, s in enumerate(scafs) if s not in test_scafs]
    return np.array(train_idx), np.array(test_idx)

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)

    df = load_df()
    df, X, y, scaler = featurize(df)
    tr, te = scaffold_split(df, test_size=0.2, seed=17)

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xte, label=yte)

    params = dict(
        objective="reg:squarederror",
        eval_metric="mae",
        eta=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        nthread=8,
    )
    evallist = [(dtr, "train"), (dval, "valid")]
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=5000,
        evals=evallist,
        early_stopping_rounds=200,
        verbose_eval=100,
    )
    preds = booster.predict(dval)
    mae = mean_absolute_error(yte, preds)
    r2 = r2_score(yte, preds)
    print(f"LogD | XGB | MAE={mae:.3f} | R^2={r2:.3f} | rounds={booster.best_iteration}")

    booster.save_model("out/logd_xgb.json")
    pd.to_pickle(scaler, "out/logd_xgb_scaler.pkl")
    pd.DataFrame(
        {"molecule_id": df.iloc[te].molecule_id, "y_true": yte, "y_pred": preds}
    ).to_csv("out/logd_xgb_eval.csv", index=False)

    importances = booster.get_score(importance_type="gain")  # or "weight"
    # Map last N indices to descriptor names (rest are ECFP bits)
    desc_names = ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors",
                "NumRotBonds","FractionCSP3","RingCount"]
    def feat_name(i, nbits=2048):
        return f"ECFP4_{i}" if i < nbits else f"DESC_{desc_names[i-nbits]}"

    imp_df = pd.DataFrame(
        [(feat_name(int(k[1:])), v) for k,v in importances.items()],
        columns=["feature","importance"]
    ).sort_values("importance", ascending=False)
    imp_df.to_csv("out/logd_xgb_importances.csv", index=False)

