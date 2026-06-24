"""
ProToxNet | Leave-AE-Out (LAO) Evaluation
==========================================
AE-disjoint split: 15% of MedDRA PT terms held out.
AE-specific bias terms b_a are zeroed at test time to simulate
unseen AE vocabulary.

Result: AUC 0.8472 (AP 0.8255, 1,980 held-out AEs, 92,688 test pairs)

Outputs:
  lao_results.csv  — AUC/AP + split summary
"""

import pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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


def main():
    print("="*60)
    print("ProToxNet — Leave-AE-Out Evaluation")
    print("="*60)

    # ── Load ──────────────────────────────────────────────────────────────────
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
    print(f"FAERS pairs: {len(fv):,}")

    N_DRUGS = len(drug_list)
    N_AES   = len(ae2id)

    # ── LAO split (AE-level) ──────────────────────────────────────────────────
    print("\n[A] AE-level split (85% train / 15% test)...")
    all_aes = fv["meddra_name"].unique()
    train_aes, test_aes = train_test_split(all_aes, test_size=0.15, random_state=42)
    train_ae_set = set(train_aes)
    test_ae_set  = set(test_aes)
    fv_train = fv[fv["meddra_name"].isin(train_ae_set)].copy()
    fv_test  = fv[fv["meddra_name"].isin(test_ae_set)].copy()
    print(f"  Train AEs: {len(train_ae_set):,} | Test AEs: {len(test_ae_set):,}")
    print(f"  Train pairs: {len(fv_train):,} | Test pairs: {len(fv_test):,}")

    # ── Build train/test arrays ────────────────────────────────────────────────
    np.random.seed(42)

    def make_arrays(df_pos, ae_pool, n_neg_drugs=None):
        pos_di = df_pos["drug_id"].values
        pos_ai = df_pos["ae_id"].values
        n = len(pos_di)
        if n_neg_drugs is None:
            neg_di = np.random.randint(0, N_DRUGS, n)
        else:
            neg_di = np.random.choice(list(n_neg_drugs), n)
        neg_ai = np.array([ae2id[a] for a in
                           np.random.choice(list(ae_pool), n)])
        all_di = np.concatenate([pos_di, neg_di])
        all_ai = np.concatenate([pos_ai, neg_ai])
        y      = np.concatenate([np.ones(n), np.zeros(n)])
        return all_di, all_ai, y

    tr_di, tr_ai, y_tr = make_arrays(fv_train, train_ae_set)
    te_di, te_ai, y_te = make_arrays(fv_test, test_ae_set)
    print(f"  Train array: {len(y_tr):,} | Test array: {len(y_te):,}")

    # ── Model + training ──────────────────────────────────────────────────────
    print("\n[B] Training...")
    torch.manual_seed(42); np.random.seed(42)
    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    model = ProToxBilinear(
        drug_dim=exposure_z.shape[1], n_aes=N_AES, n_drugs=N_DRUGS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=100, eta_min=1e-5)

    BATCH  = 4096
    EPOCHS = 100
    EPS    = 0.05

    for ep in tqdm(range(1, EPOCHS+1), desc="LAO Training", ncols=80):
        model.train()
        perm = np.random.permutation(len(tr_di))
        for s in range(0, len(perm), BATCH):
            b  = perm[s:s+BATCH]
            di = torch.tensor(tr_di[b], device=DEVICE)
            ai = torch.tensor(tr_ai[b], device=DEVICE)
            y  = torch.tensor(y_tr[b],  dtype=torch.float32, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(
                model(drug_feats[di], ai, di),
                y * (1-EPS) + (1-y)*EPS).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # ── Test: zero b_a for held-out AEs ──────────────────────────────────────
    print("\n[C] Test evaluation (AE bias zeroed for held-out AEs)...")
    model.eval()
    with torch.no_grad():
        for ae_name in test_ae_set:
            if ae_name in ae2id:
                model.ae_bias.weight[ae2id[ae_name]] = 0.0

    probs = []
    with torch.no_grad():
        for s in range(0, len(te_di), BATCH):
            di = torch.tensor(te_di[s:s+BATCH], device=DEVICE)
            ai = torch.tensor(te_ai[s:s+BATCH], device=DEVICE)
            probs.append(torch.sigmoid(model(drug_feats[di], ai, di)).cpu().numpy())
    probs = np.concatenate(probs)

    auc = roc_auc_score(y_te, probs)
    ap  = average_precision_score(y_te, probs)

    print(f"\n  ✅ LAO Results:")
    print(f"     Test AEs (held out): {len(test_ae_set):,}")
    print(f"     Test pairs:          {len(y_te):,}")
    print(f"     AUC: {auc:.4f}")
    print(f"     AP:  {ap:.4f}")

    pd.DataFrame([{
        "eval":          "Leave-AE-out",
        "test_auc":      round(auc, 4),
        "test_ap":       round(ap, 4),
        "n_test_aes":    len(test_ae_set),
        "n_test_pairs":  len(y_te),
    }]).to_csv(DRIVE / "lao_results.csv", index=False)
    print("  ✅ lao_results.csv saved")


if __name__ == "__main__":
    main()
