"""
ProToxNet | Ablation Study (Table 3)
=====================================
Tests the contribution of two architectural choices:

  1. Bias terms (b_d, b_a)
     Removed → val AUC drops from 0.9586 to 0.8874 (Δ=-0.0729)
     Interpretation: bias terms absorb popularity signal, freeing
     the bilinear component to learn tissue-specific signal.

  2. Label smoothing (ε=0.05)
     Removed → val AUC drops to 0.9416 (Δ=-0.0171)
     Interpretation: smoothing reduces overconfidence on noisy FAERS labels.

Both ablations use identical training hyperparameters to step4_train.py.

Outputs:
  ablation.csv   — {Condition, Val_AUC, Test_AUC, Test_AP, Delta}
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

# Reference results from step4_train.py
REFERENCE_AUC = 0.9585
REFERENCE_AP  = 0.9579
REFERENCE_VAL = 0.9586


# ─────────────────────────────────────────────────────────────────────────────
# Flexible model with optional bias terms
# ─────────────────────────────────────────────────────────────────────────────
class ProToxBilinear(nn.Module):
    def __init__(self, drug_dim=68, n_aes=13200, ae_emb_dim=64, latent=128,
                 n_drugs=4310, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, latent), nn.LayerNorm(latent))
        self.ae_emb  = nn.Embedding(n_aes, ae_emb_dim, max_norm=1.0)
        self.ae_enc  = nn.Sequential(nn.Linear(ae_emb_dim, latent),
                                     nn.LayerNorm(latent))
        self.W = nn.Parameter(torch.ones(latent))
        if use_bias:
            self.drug_bias = nn.Embedding(n_drugs, 1)
            self.ae_bias   = nn.Embedding(n_aes, 1)
            nn.init.zeros_(self.drug_bias.weight)
            nn.init.zeros_(self.ae_bias.weight)

    def forward(self, drug_feat, ae_idx, drug_idx):
        dp = self.drug_enc(drug_feat)
        ap = self.ae_enc(self.ae_emb(ae_idx))
        sc = (dp * self.W * ap).sum(-1)
        if self.use_bias:
            sc = (sc
                  + self.drug_bias(drug_idx).squeeze(-1)
                  + self.ae_bias(ae_idx).squeeze(-1))
        return sc


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate one ablation condition
# ─────────────────────────────────────────────────────────────────────────────
def train_variant(exposure_z, fv, idx_tr, idx_va, idx_te,
                  use_bias: bool, label_smooth: float, name: str,
                  drug_feats, pos_di, pos_ai,
                  n_drugs, n_aes, epochs=100, batch=4096):

    model = ProToxBilinear(drug_dim=exposure_z.shape[1], n_aes=n_aes,
                           n_drugs=n_drugs, use_bias=use_bias).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5)

    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_va_t = torch.tensor(idx_va, dtype=torch.long)
    idx_te_t = torch.tensor(idx_te, dtype=torch.long)

    def neg_sample(n):
        nd = torch.randint(0, n_drugs, (n,), device=DEVICE)
        na = torch.randint(0, n_aes,   (n,), device=DEVICE)
        return nd, na

    @torch.no_grad()
    def evaluate(idx_t):
        model.eval()
        probs, labels = [], []
        for s in range(0, len(idx_t), batch*4):
            b   = idx_t[s:s+batch*4].to(DEVICE)
            nd, na = neg_sample(len(b))
            ps = torch.sigmoid(model(drug_feats[pos_di[b]], pos_ai[b], pos_di[b]))
            ns = torch.sigmoid(model(drug_feats[nd], na, nd))
            probs.append(torch.cat([ps, ns]).cpu())
            labels.append(torch.cat([torch.ones(len(b)),
                                     torch.zeros(len(b))]).cpu())
        p = torch.cat(probs).numpy()
        l = torch.cat(labels).numpy()
        return roc_auc_score(l, p), average_precision_score(l, p)

    best_val, best_state = 0.0, None
    outer = tqdm(range(1, epochs+1), desc=name, ncols=70)
    for epoch in outer:
        model.train()
        perm = torch.randperm(len(idx_tr_t))
        for s in range(0, len(idx_tr_t), batch):
            b  = idx_tr_t[perm[s:s+batch]].to(DEVICE)
            nd, na = neg_sample(len(b))
            ps = model(drug_feats[pos_di[b]], pos_ai[b], pos_di[b])
            ns = model(drug_feats[nd], na, nd)
            logits = torch.cat([ps, ns])
            y      = torch.cat([torch.ones(len(b)),
                                torch.zeros(len(b))]).to(DEVICE)
            eps    = label_smooth
            y_s    = y * (1 - eps) + eps / 2
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits, y_s).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if epoch % 10 == 0:
            va, _ = evaluate(idx_va_t)
            outer.set_postfix(val_auc=f"{va:.4f}")
            if va > best_val:
                best_val  = va
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_auc, test_ap = evaluate(idx_te_t)
    tqdm.write(f"  {name}: val={best_val:.4f} "
               f"test_auc={test_auc:.4f} test_ap={test_ap:.4f}")
    return best_val, test_auc, test_ap


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("ProToxNet — Ablation Study")
    print("="*60)

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

    idx = np.arange(len(fv))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    idx_va, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    N_DRUGS = len(drug_list)
    N_AES   = len(ae2id)

    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    pos_di = torch.tensor(fv["drug_id"].values, dtype=torch.long).to(DEVICE)
    pos_ai = torch.tensor(fv["ae_id"].values,   dtype=torch.long).to(DEVICE)

    ablation_results = {}

    print("\n[1/2] Ablation: no bias terms (use_bias=False, ε=0.05)...")
    v1, a1, ap1 = train_variant(
        exposure_z, fv, idx_tr, idx_va, idx_te,
        use_bias=False, label_smooth=0.05,
        name="No bias terms",
        drug_feats=drug_feats, pos_di=pos_di, pos_ai=pos_ai,
        n_drugs=N_DRUGS, n_aes=N_AES)
    ablation_results["ProToxNet without bias terms"] = (v1, a1, ap1)

    print("\n[2/2] Ablation: no label smoothing (use_bias=True, ε=0.0)...")
    v2, a2, ap2 = train_variant(
        exposure_z, fv, idx_tr, idx_va, idx_te,
        use_bias=True, label_smooth=0.0,
        name="No label smoothing",
        drug_feats=drug_feats, pos_di=pos_di, pos_ai=pos_ai,
        n_drugs=N_DRUGS, n_aes=N_AES)
    ablation_results["ProToxNet without label smoothing"] = (v2, a2, ap2)

    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Condition':<45} {'Val AUC':>8} {'Test AUC':>9} {'Test AP':>8} {'Δ AUC':>7}")
    print(f"  {'-'*80}")
    print(f"  {'Full ProToxNet (reference)':<45} "
          f"{REFERENCE_VAL:>8.4f} {REFERENCE_AUC:>9.4f} "
          f"{REFERENCE_AP:>8.4f} {'±0.0000':>7}")
    rows = [{"Condition": "Full ProToxNet (reference)",
             "Val_AUC": REFERENCE_VAL, "Test_AUC": REFERENCE_AUC,
             "Test_AP": REFERENCE_AP, "Delta": 0.0}]
    for name, (val, auc, ap) in ablation_results.items():
        delta = auc - REFERENCE_AUC
        print(f"  {name:<45} {val:>8.4f} {auc:>9.4f} "
              f"{ap:>8.4f} {delta:>+7.4f}")
        rows.append({"Condition": name, "Val_AUC": round(val,4),
                     "Test_AUC": round(auc,4), "Test_AP": round(ap,4),
                     "Delta": round(delta,4)})

    pd.DataFrame(rows).to_csv(DRIVE / "ablation.csv", index=False)
    print("✅ ablation.csv saved")


if __name__ == "__main__":
    main()
