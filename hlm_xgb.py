# scripts/hlm_xgb.py
import os, math
import numpy as np, pandas as pd, xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---- features (same as LogD) ----
DESCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.FractionCSP3,
    Descriptors.RingCount,
]

from rdkit.Chem import rdFingerprintGenerator
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

def smiles_to_mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.SanitizeMol(m); return m

def ecfp4(mol):
    fp = _morgan.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_feats(mol):
    return np.array([f(mol) for f in DESCS], dtype=np.float32)

def scaffold(smi):
    try:
        m = smiles_to_mol(smi); sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else "NA"
    except: return "NA"

# Teaser column map
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

TARGET = "HLM_Clint"      # mL/min/kg
BAND_LOG = 0.3            # optional band for plotting in log10 units (~×2)

def load_df():
    # Use HF CSV path if you prefer; this keeps your current flow
    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv")
    df = df.rename(columns=COLMAP)
    # numeric + drop non-positive (log transform requires >0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[(df[TARGET].notna()) & (df[TARGET] > 0)][["molecule_id","smiles",TARGET]].reset_index(drop=True)
    return df

def featurize(df):
    mols = [smiles_to_mol(s) for s in tqdm(df.smiles, desc="mol")]
    keep = [i for i,m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    X_fp = np.vstack([ecfp4(m) for m in tqdm(mols, desc="ecfp4")])
    X_desc = np.vstack([rdkit_feats(m) for m in tqdm(mols, desc="desc")])
    scaler = StandardScaler().fit(X_desc)
    X = np.concatenate([X_fp, scaler.transform(X_desc)], axis=1)

    # target in log10 per challenge guidance (non-log endpoints get logged)
    y = np.log10(df[TARGET].values.astype(np.float32))
    return df, X, y, scaler

def scaffold_split(df, test_size=0.2, seed=17):
    scafs = df.smiles.map(scaffold)
    uniq = pd.Series(scafs).unique().tolist()
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    k = max(1, int(len(uniq)*test_size))
    te_scaf = set(uniq[:k])
    te = [i for i,s in enumerate(scafs) if s in te_scaf]
    tr = [i for i,s in enumerate(scafs) if s not in te_scaf]
    return np.array(tr), np.array(te)

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    df = load_df()
    df, X, y_log, scaler = featurize(df)
    tr, te = scaffold_split(df, 0.2, 17)

    dtr, dte = xgb.DMatrix(X[tr], label=y_log[tr]), xgb.DMatrix(X[te], label=y_log[te])

    params = dict(
        objective="reg:squarederror",
        eval_metric="mae",
        eta=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.6,
        reg_lambda=2.0, nthread=8
    )
    booster = xgb.train(params, dtr, num_boost_round=5000,
                        evals=[(dtr,"train"),(dte,"valid")],
                        early_stopping_rounds=200, verbose_eval=100)

    pred_log = booster.predict(dte)
    # metrics in log space
    mae_log = mean_absolute_error(y_log[te], pred_log)

    # also report MAE in original units (optional)
    y_true = (10**y_log[te])
    y_pred = (10**pred_log)
    mae_lin = mean_absolute_error(y_true, y_pred)

    # simple R^2 in log space
    ss_res = np.sum((y_log[te]-pred_log)**2)
    ss_tot = np.sum((y_log[te]-np.mean(y_log[te]))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else float("nan")

    print(f"HLM_Clint | XGB | MAE_log10={mae_log:.3f} | MAE_linear={mae_lin:.1f} mL/min/kg | R^2_log={r2:.3f} | rounds={booster.best_iteration}")

    # save
    booster.save_model("out/hlm_xgb.json")
    pd.to_pickle(scaler, "out/hlm_xgb_scaler.pkl")
    pd.DataFrame({"molecule_id": df.iloc[te].molecule_id,
                  "y_true_log10": y_log[te], "y_pred_log10": pred_log,
                  "y_true": y_true, "y_pred": y_pred}).to_csv("out/hlm_xgb_eval.csv", index=False)
