"""
ProToxNet | Fair Baseline Comparison (Table 2)
===============================================
All methods evaluated on identical FAERS held-out test pairs
(same random_state=42 split as step4_train.py).

Methods:
  1. Random                          AUC=0.500  (theoretical)
  2. AE frequency only (FAERS bias)  AUC=0.8681 (reporting-frequency baseline)
  3. LR: drug exposure only          AUC=0.5825 (unfair — no AE repr)
  4. LR: drug exposure + AE freq     AUC=0.8525 (fair pair-level baseline)
  5. Chem-only: Morgan FP + AE freq  AUC=0.9680 (pair-level, collapses under LDO)
     → Chem-only LDO cold-start:     AUC=0.5992 (drug-disjoint, same LDO split)
  6. ProToxNet (ours)                AUC=0.9581

The Morgan FP + AE freq result illustrates pair-level memorisation:
high pair-level AUC but collapses to near-random under LDO,
contrasting with ProToxNet's LDO AUC of 0.8544.

Outputs:
  fair_baselines.csv
  morgan_mat.npy     (cached, reused by eval_ldo.py for chem-only LDO)
"""

import pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from eval import get_data_dir
warnings.filterwarnings("ignore")

DRIVE  = get_data_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class ProToxBilinear(nn.Module):
    def __init__(self, drug_dim=68, n_aes=13200, ae_emb_dim=64, latent=128,
                 n_drugs=4310):
        super().__init__()
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, latent), nn.LayerNorm(latent))
        self.ae_emb  = nn.Embedding(n_aes, ae_emb_dim, max_norm=1.0)
        self.ae_enc  = nn.Sequential(nn.Linear(ae_emb_dim, latent),
                                     nn.LayerNorm(latent))
        self.W         = nn.Parameter(torch.ones(latent))
        self.drug_bias = nn.Embedding(n_drugs, 1)
        self.ae_bias   = nn.Embedding(n_aes, 1)
    def forward(self, df, ai, di):
        dp = self.drug_enc(df)
        ap = self.ae_enc(self.ae_emb(ai))
        return ((dp * self.W * ap).sum(-1)
                + self.drug_bias(di).squeeze(-1)
                + self.ae_bias(ai).squeeze(-1))


def build_morgan_matrix(drug_list):
    """Compute or load 2048-bit Morgan FP matrix for all drugs."""
    cache = DRIVE / "morgan_mat.npy"
    if cache.exists():
        mat = np.load(cache)
        print(f"Morgan FP matrix loaded: {mat.shape}")
        return mat

    from rdkit import Chem
    from rdkit.Chem import AllChem
    import psycopg2
    conn = psycopg2.connect(host="unmtid-dbs.net", port=5433,
                            user="drugman", password="dosage",
                            dbname="drugcentral")
    structs = pd.read_sql(
        "SELECT name, smiles FROM structures WHERE smiles IS NOT NULL", conn)
    conn.close()
    structs["name_lower"] = structs["name"].str.lower()
    name2smi = structs.set_index("name_lower")["smiles"].to_dict()

    mat = np.zeros((len(drug_list), 2048), dtype=np.float32)
    for i, drug in enumerate(tqdm(drug_list, desc="Morgan FP")):
        smi = name2smi.get(drug)
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    mat[i] = np.array(fp)
            except Exception:
                pass
    np.save(cache, mat)
    print(f"✅ Morgan FP matrix saved: {mat.shape}")
    return mat


def main():
    print("="*60)
    print("ProToxNet — Fair Baseline Comparison")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(DRIVE / "exposure_matrix.pkl", "rb") as f:
        pkg = pickle.load(f)
    exposure_z = pkg["exposure_z"]
    drug_list  = pkg["drug_list"]

    with open(DRIVE / "bilinear_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    ae_list = emb["ae_list"]
    ae2id   = {a: i for i, a in enumerate(ae_list)}
    drug2id = {d: i for i, d in enumerate(drug_list)}

    faers_df  = pd.read_csv(DRIVE / "drugcentral_faers.csv")
    faers_pos = faers_df[faers_df["llr"] > 0].copy()
    faers_pos["drug_lower"] = faers_pos["drug_name"].str.lower()
    faers_pos = faers_pos[faers_pos["drug_lower"].isin(drug2id)]
    fv = faers_pos[faers_pos["meddra_name"].isin(ae2id)].copy()
    fv["drug_id"] = fv["drug_lower"].map(drug2id)
    fv["ae_id"]   = fv["meddra_name"].map(ae2id)

    # ── Same test split as step4_train.py ─────────────────────────────────────
    idx = np.arange(len(fv))
    _, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    _, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)
    np.random.seed(42)
    test_pos_di = fv["drug_id"].values[idx_te]
    test_pos_ai = fv["ae_id"].values[idx_te]
    n_te   = len(idx_te)
    neg_di = np.random.randint(0, len(drug_list), n_te)
    neg_ai = np.random.randint(0, len(ae2id), n_te)
    all_di = np.concatenate([test_pos_di, neg_di])
    all_ai = np.concatenate([test_pos_ai, neg_ai])
    y_test = np.concatenate([np.ones(n_te), np.zeros(n_te)])
    print(f"Test pairs: {len(y_test):,} (pos:{n_te:,})")

    # ── Features ──────────────────────────────────────────────────────────────
    X_drug    = exposure_z[all_di]
    ae_counts = fv["ae_id"].value_counts().to_dict()
    X_ae_freq = np.array([ae_counts.get(ai, 0) for ai in all_ai],
                          dtype=np.float32).reshape(-1,1)
    morgan_mat = build_morgan_matrix(drug_list)
    X_morgan  = morgan_mat[all_di]

    half      = n_te // 2
    train_idx = np.concatenate([np.arange(half), np.arange(n_te, n_te+half)])

    results = {}

    # 1. Random
    results["Random"] = {"auc": 0.500, "ap": float(y_test.mean()),
                         "description": "Theoretical lower bound"}

    # 2. AE frequency only
    results["AE frequency (FAERS)"] = {
        "auc": roc_auc_score(y_test, X_ae_freq.flatten()),
        "ap":  average_precision_score(y_test, X_ae_freq.flatten()),
        "description": "Reporting frequency only (bias baseline)"}

    # 3. LR: drug exposure only (unfair — no AE representation)
    sc1 = StandardScaler(); X1 = sc1.fit_transform(X_drug)
    lr1 = LogisticRegression(max_iter=200, random_state=42)
    lr1.fit(X1[train_idx], y_test[train_idx])
    p1  = lr1.predict_proba(X1)[:,1]
    results["LR: drug exposure only"] = {
        "auc": roc_auc_score(y_test, p1),
        "ap":  average_precision_score(y_test, p1),
        "description": "68-dim exposure, no AE representation (unfair)"}

    # 4. LR: drug exposure + AE frequency (fair)
    sc2 = StandardScaler(); X2 = sc2.fit_transform(np.hstack([X_drug, X_ae_freq]))
    lr2 = LogisticRegression(max_iter=200, random_state=42)
    lr2.fit(X2[train_idx], y_test[train_idx])
    p2  = lr2.predict_proba(X2)[:,1]
    results["LR: drug exposure + AE freq"] = {
        "auc": roc_auc_score(y_test, p2),
        "ap":  average_precision_score(y_test, p2),
        "description": "68-dim exposure + AE frequency (fair pair baseline)"}

    # 5. Chem-only: Morgan FP + AE freq (pair-level; collapses to 0.5992 under LDO)
    sc3 = StandardScaler()
    X3  = sc3.fit_transform(np.hstack([X_morgan, X_ae_freq]))
    lr3 = LogisticRegression(max_iter=300, C=0.1, solver="saga",
                             n_jobs=-1, random_state=42)
    lr3.fit(X3[train_idx], y_test[train_idx])
    p3  = lr3.predict_proba(X3)[:,1]
    results["Chem-only: Morgan FP + AE freq\n(pair-level; LDO collapses to 0.5992)"] = {
        "auc": roc_auc_score(y_test, p3),
        "ap":  average_precision_score(y_test, p3),
        "description": "2048-dim Morgan FP + AE freq (pair-level memorisation)"}

    # 6. ProToxNet
    model = ProToxBilinear(drug_dim=exposure_z.shape[1],
                           n_aes=len(ae2id), n_drugs=len(drug_list)).to(DEVICE)
    model.load_state_dict(torch.load(DRIVE / "protoxnet_bilinear.pt",
                                     map_location=DEVICE))
    model.eval()
    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    all_probs  = []
    BATCH = 4096
    with torch.no_grad():
        for s in range(0, len(all_di), BATCH):
            di_b = torch.tensor(all_di[s:s+BATCH], device=DEVICE)
            ai_b = torch.tensor(all_ai[s:s+BATCH], device=DEVICE)
            all_probs.append(torch.sigmoid(
                model(drug_feats[di_b], ai_b, di_b)).cpu().numpy())
    pt_p = np.concatenate(all_probs)
    results["ProToxNet (ours)"] = {
        "auc": roc_auc_score(y_test, pt_p),
        "ap":  average_precision_score(y_test, pt_p),
        "description": "Bilinear, tissue engagement potential features"}

    # ── Print + save ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAIR BASELINE COMPARISON (FAERS held-out test)")
    print(f"{'='*70}")
    print(f"  {'Method':<55} {'AUC':>6} {'AP':>6}")
    print(f"  {'-'*70}")
    for name, res in results.items():
        marker = " ◀" if "ProToxNet" in name else ""
        print(f"  {name.split(chr(10))[0]:<55} {res['auc']:>6.4f} "
              f"{res['ap']:>6.4f}{marker}")
    print(f"{'='*70}")

    rows = [{"Method": k, "AUC": round(v["auc"],4),
             "AP": round(v["ap"],4), "Description": v["description"]}
            for k, v in results.items()]
    pd.DataFrame(rows).to_csv(DRIVE / "fair_baselines.csv", index=False)
    print("✅ fair_baselines.csv saved")


if __name__ == "__main__":
    main()
