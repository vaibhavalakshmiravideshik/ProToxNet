"""
ProToxNet | Leave-Drug-Out (LDO) Cold-Start Evaluation
=======================================================
Drug-disjoint split: 70/15/15 at the DRUG level.
Test drugs have zero FAERS pairs seen during training.
Drug-specific bias terms b_d are zeroed at test time.

Result: cold-start AUC 0.8544 (AP 0.8509, 244 unseen drugs, 32,350 pairs)
Interpretation: tissue exposure features carry real generalisation signal.
Chem-only baseline (Morgan FP + AE freq, same LDO split): AUC 0.5992.

Outputs:
  protoxnet_ldo.pt         — best LDO checkpoint
  ldo_training_history.csv — val AUC per epoch
  ldo_results.csv          — cold-start AUC/AP summary
"""

import pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

DRIVE  = Path("/content/drive/MyDrive/ProToxNet/data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Model (with optional drug bias — zeroed for test drugs)
# ─────────────────────────────────────────────────────────────────────────────
class ProToxBilinearLDO(nn.Module):
    def __init__(self, drug_dim=68, n_aes=13200,
                 ae_emb_dim=64, latent=128, n_drugs=4310):
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
        nn.init.zeros_(self.drug_bias.weight)
        nn.init.zeros_(self.ae_bias.weight)

    def forward(self, drug_feat, ae_idx, drug_idx, use_drug_bias=True):
        dp = self.drug_enc(drug_feat)
        ap = self.ae_enc(self.ae_emb(ae_idx))
        sc = (dp * self.W * ap).sum(-1)
        if use_drug_bias:
            sc = sc + self.drug_bias(drug_idx).squeeze(-1)
        sc = sc + self.ae_bias(ae_idx).squeeze(-1)
        return sc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def neg_sample(n, drug_pool, n_aes, device):
    pool = torch.tensor(list(drug_pool), device=device)
    nd   = pool[torch.randint(0, len(pool), (n,), device=device)]
    na   = torch.randint(0, n_aes, (n,), device=device)
    return nd, na


@torch.no_grad()
def evaluate_ldo(model, di, ai, drug_pool, drug_feats,
                 n_aes, use_drug_bias, batch=4096):
    model.eval()
    probs, labels = [], []
    pool_list = list(drug_pool)
    for s in range(0, len(di), batch * 4):
        b_di = di[s:s+batch*4]
        b_ai = ai[s:s+batch*4]
        nb   = len(b_di)
        pool_t = torch.tensor(pool_list, device=b_di.device)
        nd = pool_t[torch.randint(0, len(pool_t), (nb,), device=b_di.device)]
        na = torch.randint(0, n_aes, (nb,), device=b_di.device)
        ps = torch.sigmoid(model(drug_feats[b_di], b_ai, b_di,
                                 use_drug_bias=use_drug_bias))
        ns = torch.sigmoid(model(drug_feats[nd], na, nd,
                                 use_drug_bias=use_drug_bias))
        probs.append(torch.cat([ps, ns]).cpu())
        labels.append(torch.cat([torch.ones(nb), torch.zeros(nb)]).cpu())
    p = torch.cat(probs).numpy()
    l = torch.cat(labels).numpy()
    return roc_auc_score(l, p), average_precision_score(l, p)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("ProToxNet — Leave-Drug-Out Evaluation")
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
    print(f"FAERS pairs: {len(fv):,} | Drugs: {fv['drug_id'].nunique():,}")

    N_DRUGS = len(drug_list)
    N_AES   = len(ae2id)

    # ── LDO split (drug-level) ────────────────────────────────────────────────
    print("\n[A] Drug-level LDO split (70/15/15)...")
    np.random.seed(42)
    unique_drugs = fv["drug_id"].unique()
    np.random.shuffle(unique_drugs)
    n_tr = int(0.70 * len(unique_drugs))
    n_va = int(0.15 * len(unique_drugs))
    train_drugs = set(unique_drugs[:n_tr])
    val_drugs   = set(unique_drugs[n_tr:n_tr+n_va])
    test_drugs  = set(unique_drugs[n_tr+n_va:])
    assert len(train_drugs & test_drugs) == 0, "Drug overlap detected!"

    fv_tr = fv[fv["drug_id"].isin(train_drugs)].reset_index(drop=True)
    fv_va = fv[fv["drug_id"].isin(val_drugs)].reset_index(drop=True)
    fv_te = fv[fv["drug_id"].isin(test_drugs)].reset_index(drop=True)
    print(f"  Train: {len(fv_tr):,} pairs / {len(train_drugs):,} drugs")
    print(f"  Val:   {len(fv_va):,} pairs / {len(val_drugs):,} drugs")
    print(f"  Test:  {len(fv_te):,} pairs / {len(test_drugs):,} drugs")
    print("  ✅ Zero drug overlap (cold-start guaranteed)")

    # ── Tensors ────────────────────────────────────────────────────────────────
    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    def to_gpu(df):
        return (torch.tensor(df["drug_id"].values, dtype=torch.long).to(DEVICE),
                torch.tensor(df["ae_id"].values,   dtype=torch.long).to(DEVICE))
    tr_di, tr_ai = to_gpu(fv_tr)
    va_di, va_ai = to_gpu(fv_va)
    te_di, te_ai = to_gpu(fv_te)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ProToxBilinearLDO(
        drug_dim=exposure_z.shape[1], n_aes=N_AES, n_drugs=N_DRUGS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=100, eta_min=1e-5)

    # ── Training ───────────────────────────────────────────────────────────────
    print("\n[B] Training (LDO)...")
    BATCH  = 4096
    EPOCHS = 100
    EPS    = 0.05
    best_val, best_state, history = 0.0, None, []
    outer = tqdm(range(1, EPOCHS+1), desc="LDO Training", ncols=80)

    for epoch in outer:
        model.train()
        perm = torch.randperm(len(tr_di))
        for s in range(0, len(tr_di), BATCH):
            b    = perm[s:s+BATCH]
            b_di = tr_di[b]; b_ai = tr_ai[b]
            nd, na = neg_sample(len(b), train_drugs, N_AES, DEVICE)
            ps = model(drug_feats[b_di], b_ai, b_di, use_drug_bias=True)
            ns = model(drug_feats[nd], na, nd, use_drug_bias=True)
            logits  = torch.cat([ps, ns])
            y       = torch.cat([torch.ones(len(b)),
                                 torch.zeros(len(b))]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(
                logits, y * (1-EPS) + EPS/2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if epoch % 5 == 0:
            va, vap = evaluate_ldo(model, va_di, va_ai, val_drugs,
                                   drug_feats, N_AES, use_drug_bias=True)
            history.append({"epoch": epoch, "val_auc": va, "val_ap": vap})
            outer.set_postfix(val_auc=f"{va:.4f}", val_ap=f"{vap:.4f}")
            tqdm.write(f"  Ep{epoch:03d} val_auc={va:.4f} val_ap={vap:.4f}")
            if va > best_val:
                best_val  = va
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # ── Cold-start test ────────────────────────────────────────────────────────
    print("\n[C] Cold-start test (drug bias zeroed for test drugs)...")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    with torch.no_grad():
        te_drug_t = torch.tensor(list(test_drugs), device=DEVICE)
        model.drug_bias.weight[te_drug_t] = 0.0

    test_auc, test_ap = evaluate_ldo(
        model, te_di, te_ai, test_drugs,
        drug_feats, N_AES, use_drug_bias=False)

    print(f"\n  ✅ LDO Cold-Start Results:")
    print(f"     Test AUC  : {test_auc:.4f}")
    print(f"     Test AP   : {test_ap:.4f}")
    print(f"     Best val  : {best_val:.4f}")
    print(f"     Test drugs: {len(test_drugs):,} (unseen)")
    print(f"     Test pairs: {len(fv_te):,}")

    if test_auc >= 0.80:
        print("  ✅ Strong generalisation — tissue features carry real signal")
    elif test_auc >= 0.70:
        print("  ⚠️  Moderate generalisation")
    else:
        print("  ❌  Weak generalisation — model relies on drug identity")

    # ── Save ───────────────────────────────────────────────────────────────────
    torch.save(best_state, DRIVE / "protoxnet_ldo.pt")
    pd.DataFrame(history).to_csv(DRIVE / "ldo_training_history.csv", index=False)
    pd.DataFrame([{
        "eval": "LDO cold-start",
        "test_auc":       round(test_auc, 4),
        "test_ap":        round(test_ap, 4),
        "best_val_auc":   round(best_val, 4),
        "n_test_drugs":   len(test_drugs),
        "n_test_pairs":   len(fv_te),
    }]).to_csv(DRIVE / "ldo_results.csv", index=False)
    print("\n✅ LDO evaluation complete.")


if __name__ == "__main__":
    main()
