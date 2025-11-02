import os
import numpy as np, pandas as pd, xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

# --- Column map from the teaser CSV ---
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

TARGET = "KSOL"   # µM, we’ll train on log10(KSOL)

from rdkit.Chem import Descriptors, rdMolDescriptors
DESCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.FractionCSP3,
    Descriptors.RingCount, Descriptors.HeavyAtomCount, Descriptors.MolMR,
    rdMolDescriptors.CalcNumAromaticRings, rdMolDescriptors.CalcNumAliphaticRings,
    rdMolDescriptors.CalcNumAmideBonds, rdMolDescriptors.CalcNumHeteroatoms,
]

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

def smiles_to_mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.SanitizeMol(m)
    return m

def ecfp4(m):
    fp = _morgan.GetFingerprint(m)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_feats(m):
    return np.array([f(m) for f in DESCS], dtype=np.float32)

def scaffold(smi):
    try:
        m = smiles_to_mol(smi); sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else "NA"
    except: return "NA"

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

    # Load teaser CSV (you can swap to hf_hub_download if you prefer)
    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv").rename(columns=COLMAP)

    # Keep valid rows, positive KSOL (log10 requires >0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[(df[TARGET].notna()) & (df[TARGET] > 0)][["molecule_id","smiles",TARGET]].reset_index(drop=True)

    # after you build df with molecule_id, smiles, KSOL
    oof = pd.read_csv("out/logd_oof_preds.csv")  # columns: molecule_id, logd_pred
    df = df.merge(oof, on="molecule_id", how="left")
    # fill missing safely (use train-mean later, but simple mean works here)
    df["logd_pred"] = df["logd_pred"].fillna(df["logd_pred"].mean())


    # Featurize
    mols = [smiles_to_mol(s) for s in tqdm(df.smiles, desc="mol")]
    keep = [i for i,m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    X_fp   = np.vstack([ecfp4(m) for m in tqdm(mols, desc="ecfp4")])
    X_desc = np.vstack([rdkit_feats(m) for m in tqdm(mols, desc="desc")])
    scaler = StandardScaler().fit(X_desc)
    X_dense = scaler.transform(X_desc)
    logd_col = df["logd_pred"].to_numpy().reshape(-1, 1)
    X = np.concatenate([X_fp, X_dense, logd_col], axis=1)


    # Target in log10
    y_log = np.log10(df[TARGET].values.astype(np.float32))

    # Split (scaffold)
    tr, te = scaffold_split(df, 0.2, seed=17)

    dtr, dte = xgb.DMatrix(X[tr], label=y_log[tr]), xgb.DMatrix(X[te], label=y_log[te])

    params = dict(
    objective="reg:squarederror",
    eval_metric="mae",
    eta=0.02,              # slower lr
    max_depth=6,           # shallower trees generalize better
    min_child_weight=4,    # avoid tiny leaves
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=3.0,        # a bit more L2
    reg_alpha=0.0,
    nthread=8,
    )


    booster = xgb.train(
    params, dtr, num_boost_round=6000,
    evals=[(dtr,"train"), (dte,"valid")],
    early_stopping_rounds=400, verbose_eval=200
    )


    pred_log = booster.predict(dte)
    mae_log = mean_absolute_error(y_log[te], pred_log)
    r2 = r2_score(y_log[te], pred_log)

    # Also report linear-scale MAE (optional)
    y_true = 10**y_log[te]
    y_pred = 10**pred_log
    mae_lin = mean_absolute_error(y_true, y_pred)

    print(f"KSOL | XGB | MAE_log10={mae_log:.3f} | MAE_linear={mae_lin:.1f} µM | R^2_log={r2:.3f} | rounds={booster.best_iteration}")

    # Save artifacts + eval CSV (for plotting)
    booster.save_model("out/ksol_xgb.json")
    pd.to_pickle(scaler, "out/ksol_xgb_scaler.pkl")
    pd.DataFrame({
        "molecule_id": df.iloc[te].molecule_id,
        "y_true_log10": y_log[te], "y_pred_log10": pred_log,
        "y_true": y_true, "y_pred": y_pred
    }).to_csv("out/ksol_xgb_eval.csv", index=False)
