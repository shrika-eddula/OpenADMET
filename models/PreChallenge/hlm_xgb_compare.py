import os, numpy as np, pandas as pd, xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator

TARGET = "HLM_Clint"   
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
DESCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.FractionCSP3,
    Descriptors.RingCount,
]
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

def smiles_to_mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.SanitizeMol(m); return m

def ecfp4(m):
    fp = _morgan.GetFingerprint(m)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_feats(m): return np.array([f(m) for f in DESCS], dtype=np.float32)

def scaffold(smi):
    try:
        m = smiles_to_mol(smi); sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else "NA"
    except: return "NA"

# this is pretty arbitrary, but works for now
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
    # ---- Load + standardize ----
    raw = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv").rename(columns=COLMAP)
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")
    df = raw[(raw[TARGET].notna()) & (raw[TARGET] > 0)][["molecule_id","smiles",TARGET]].reset_index(drop=True)

    # Merge OOF LogD predictions 
    oof = pd.read_csv("out/logd_oof_preds.csv") 
    df = df.merge(oof, on="molecule_id", how="left")

    # ---- Featurize ----
    mols = [smiles_to_mol(s) for s in tqdm(df.smiles, desc="mol")]
    keep = [i for i,m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    X_fp   = np.vstack([ecfp4(m) for m in tqdm(mols, desc="ecfp4")])
    X_desc = np.vstack([rdkit_feats(m) for m in tqdm(mols, desc="desc")])
    scaler = StandardScaler().fit(X_desc)
    X_base = np.concatenate([X_fp, scaler.transform(X_desc)], axis=1)

    # y in log10
    y_log = np.log10(df[TARGET].values.astype(np.float32))

    # Same scaffold split for both A and B
    tr, te = scaffold_split(df, 0.2, 17)

    # ---- (A) WITHOUT predicted LogD ----
    dtrA, dteA = xgb.DMatrix(X_base[tr], label=y_log[tr]), xgb.DMatrix(X_base[te], label=y_log[te])
    params = dict(objective="reg:squarederror", eval_metric="mae",
                  eta=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0)
    bstA = xgb.train(params, dtrA, num_boost_round=5000,
                     evals=[(dtrA,"train"),(dteA,"valid")],
                     early_stopping_rounds=200, verbose_eval=False)
    predA = bstA.predict(dteA)
    mae_log_A = mean_absolute_error(y_log[te], predA)

    # back-transform for linear MAE (optional)
    y_true = 10**y_log[te]; y_predA = 10**predA
    mae_lin_A = mean_absolute_error(y_true, y_predA)

    # ---- (B) WITH predicted LogD as a feature ----
    logd_pred = df["logd_pred"].to_numpy()
    if np.isnan(logd_pred).any():
        fill = np.nanmean(logd_pred[tr]) if np.isfinite(np.nanmean(logd_pred[tr])) else 0.0
        logd_pred = np.where(np.isnan(logd_pred), fill, logd_pred)

    X_aug = np.concatenate([X_base, logd_pred.reshape(-1,1)], axis=1)
    dtrB, dteB = xgb.DMatrix(X_aug[tr], label=y_log[tr]), xgb.DMatrix(X_aug[te], label=y_log[te])
    bstB = xgb.train(params, dtrB, num_boost_round=5000,
                     evals=[(dtrB,"train"),(dteB,"valid")],
                     early_stopping_rounds=200, verbose_eval=False)
    predB = bstB.predict(dteB)
    mae_log_B = mean_absolute_error(y_log[te], predB)
    y_predB = 10**predB
    mae_lin_B = mean_absolute_error(y_true, y_predB)

    # ---- Report ----
    print("\n=== HLM_Clint (log10 scale) — A/B comparison ===")
    print(f"A: WITHOUT LogD_pred | MAE_log10={mae_log_A:.3f} | MAE_linear={mae_lin_A:.1f} mL/min/kg | rounds={bstA.best_iteration}")
    print(f"B: WITH    LogD_pred | MAE_log10={mae_log_B:.3f} | MAE_linear={mae_lin_B:.1f} mL/min/kg | rounds={bstB.best_iteration}")
    delta = mae_log_A - mae_log_B
    print(f"Δ (A−B) = {delta:+.3f} in log10 MAE  ({'improvement' if delta>0 else 'worse' if delta<0 else 'no change'})")

    # Save eval CSVs
    pd.DataFrame({"molecule_id": df.iloc[te].molecule_id,
                  "y_true_log10": y_log[te], "y_pred_log10": predA,
                  "y_true": y_true, "y_pred": y_predA}).to_csv("out/hlm_xgb_eval_no_logd.csv", index=False)

    pd.DataFrame({"molecule_id": df.iloc[te].molecule_id,
                  "y_true_log10": y_log[te], "y_pred_log10": predB,
                  "y_true": y_true, "y_pred": y_predB}).to_csv("out/hlm_xgb_eval_with_logd.csv", index=False)
