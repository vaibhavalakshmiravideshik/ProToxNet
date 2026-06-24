"""
ProToxNet | Step 4: Bilinear Model Training
============================================
Model: Diagonal bilinear (DistMult-style) drug-AE interaction model.

    score(drug, ae) = (drug_proj ⊙ W ⊙ ae_proj).sum() + b_drug + b_ae

Drug path:  68-dim tissue exposure z-scores → Linear(256) → LN → ReLU
            → Dropout(0.2) → Linear(128) → LN
AE path:    Embedding(64) → Linear(128) → LN
Bias terms: per-drug b_d, per-AE b_a (initialised to 0)

Training:   AdamW, lr=5e-4, weight_decay=1e-4
            CosineAnnealingLR (T_max=100, eta_min=1e-5)
            Label smoothing ε=0.05
            1:1 random negative sampling per batch

Inputs:
  exposure_matrix.pkl      — drug × tissue z-scores (4310 × 68)
  drugcentral_faers.csv    — FAERS LLR signals (ground truth)

Outputs:
  protoxnet_bilinear.pt    — best checkpoint (by val AUC)
  bilinear_embeddings.pkl  — {drug_emb, ae_emb, drug_list, ae_list, ...}
  bilinear_history.csv     — epoch / loss / val_auc / val_ap
"""

import gc, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from pipeline import get_data_dir
warnings.filterwarnings("ignore")

DRIVE  = get_data_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ProToxBilinear(nn.Module):
    """
    Diagonal bilinear (DistMult-style) drug–AE scoring model.

    Parameters
    ----------
    drug_dim : int   Input dimensionality (tissue exposure z-scores, 68)
    n_aes    : int   Number of AE terms in vocabulary
    ae_emb_dim: int  Learned AE embedding dimension (64)
    latent   : int   Shared latent dimension for bilinear product (128)
    """
    def __init__(self, drug_dim: int = 68, n_aes: int = 13200,
                 ae_emb_dim: int = 64, latent: int = 128):
        super().__init__()
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent), nn.LayerNorm(latent),
        )
        self.ae_emb = nn.Embedding(n_aes, ae_emb_dim, max_norm=1.0)
        self.ae_enc = nn.Sequential(
            nn.Linear(ae_emb_dim, latent), nn.LayerNorm(latent),
        )
        self.W          = nn.Parameter(torch.ones(latent))
        self.drug_bias  = nn.Embedding(4310, 1)
        self.ae_bias    = nn.Embedding(n_aes, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.drug_enc[0].weight)
        nn.init.xavier_uniform_(self.drug_enc[4].weight)
        nn.init.normal_(self.ae_emb.weight, std=0.1)
        nn.init.zeros_(self.drug_bias.weight)
        nn.init.zeros_(self.ae_bias.weight)

    def encode_drugs(self, x):
        return self.drug_enc(x)

    def encode_aes(self, idx):
        return self.ae_enc(self.ae_emb(idx))

    def score(self, dp, ap, di, ai):
        return ((dp * self.W * ap).sum(-1)
                + self.drug_bias(di).squeeze(-1)
                + self.ae_bias(ai).squeeze(-1))

    def forward(self, drug_feat, ae_idx, drug_idx):
        dp = self.encode_drugs(drug_feat)
        ap = self.encode_aes(ae_idx)
        return self.score(dp, ap, drug_idx, ae_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def neg_sample(n, n_drugs, n_aes, device):
    nd = torch.randint(0, n_drugs, (n,), device=device)
    na = torch.randint(0, n_aes,   (n,), device=device)
    return nd, na


@torch.no_grad()
def evaluate(model, idx_t, pos_di, pos_ai, drug_feats,
             n_drugs, n_aes, batch=4096):
    model.eval()
    probs, labels = [], []
    for s in range(0, len(idx_t), batch * 4):
        b   = idx_t[s:s + batch * 4].to(DEVICE)
        nd, na = neg_sample(len(b), n_drugs, n_aes, DEVICE)
        ps = torch.sigmoid(model(drug_feats[pos_di[b]], pos_ai[b], pos_di[b]))
        ns = torch.sigmoid(model(drug_feats[nd], na, nd))
        probs.append(torch.cat([ps, ns]).cpu())
        labels.append(torch.cat([torch.ones(len(b)),
                                 torch.zeros(len(b))]).cpu())
    p = torch.cat(probs).numpy()
    l = torch.cat(labels).numpy()
    return roc_auc_score(l, p), average_precision_score(l, p)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("ProToxNet — Step 4: Bilinear Model Training")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[A] Loading data...")
    with open(DRIVE / "exposure_matrix.pkl", "rb") as f:
        pkg = pickle.load(f)
    exposure_z  = pkg["exposure_z"]        # (4310, 68)
    drug_list   = pkg["drug_list"]
    tissue_list = pkg["tissue_list"]
    print(f"  Drugs: {len(drug_list):,} | Tissues: {len(tissue_list)}")

    faers_df = pd.read_csv(DRIVE / "drugcentral_faers.csv")

    # ── ID maps ────────────────────────────────────────────────────────────────
    print("\n[B] Building ID maps...")
    drug2id  = {d: i for i, d in enumerate(drug_list)}
    faers_pos = faers_df[faers_df["llr"] > 0].copy()
    faers_pos["drug_lower"] = faers_pos["drug_name"].str.lower()
    faers_pos = faers_pos[faers_pos["drug_lower"].isin(drug2id)]
    ae_list  = sorted(faers_pos["meddra_name"].dropna().unique())
    ae2id    = {a: i for i, a in enumerate(ae_list)}
    fv       = faers_pos[faers_pos["meddra_name"].isin(ae2id)].copy()
    fv["drug_id"] = fv["drug_lower"].map(drug2id)
    fv["ae_id"]   = fv["meddra_name"].map(ae2id)
    print(f"  AEs: {len(ae2id):,} | Pairs: {len(fv):,}")

    # ── Tensors ────────────────────────────────────────────────────────────────
    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    pos_di = torch.tensor(fv["drug_id"].values, dtype=torch.long).to(DEVICE)
    pos_ai = torch.tensor(fv["ae_id"].values,   dtype=torch.long).to(DEVICE)

    # ── Split ──────────────────────────────────────────────────────────────────
    print("\n[C] Splitting (70/15/15)...")
    idx = np.arange(len(fv))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    idx_va, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)
    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_va_t = torch.tensor(idx_va, dtype=torch.long)
    idx_te_t = torch.tensor(idx_te, dtype=torch.long)
    print(f"  Train: {len(idx_tr):,} | Val: {len(idx_va):,} | Test: {len(idx_te):,}")

    N_DRUGS = len(drug_list)
    N_AES   = len(ae2id)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[D] Initialising model...")
    model = ProToxBilinear(drug_dim=exposure_z.shape[1],
                           n_aes=N_AES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} | Drug dim: {exposure_z.shape[1]} "
          f"→ 128 | AE emb: 64 → 128")

    # ── Training ───────────────────────────────────────────────────────────────
    print("\n[E] Training (100 epochs)...")
    BATCH  = 4096
    EPOCHS = 100
    EPS    = 0.05    # label smoothing
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-5)

    best_val, best_state, history = 0.0, None, []
    outer = tqdm(range(1, EPOCHS + 1), desc="Epochs", ncols=80)

    for epoch in outer:
        model.train()
        perm = torch.randperm(len(idx_tr_t))
        e_loss, n_b = 0.0, 0
        for s in range(0, len(idx_tr_t), BATCH):
            b  = idx_tr_t[perm[s:s+BATCH]].to(DEVICE)
            nd, na = neg_sample(len(b), N_DRUGS, N_AES, DEVICE)
            ps = model(drug_feats[pos_di[b]], pos_ai[b], pos_di[b])
            ns = model(drug_feats[nd], na, nd)
            logits  = torch.cat([ps, ns])
            y       = torch.cat([torch.ones(len(b)),
                                 torch.zeros(len(b))]).to(DEVICE)
            y_smooth = y * (1 - EPS) + EPS / 2
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits, y_smooth).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_val = F.binary_cross_entropy_with_logits(
                logits.detach(), y_smooth).item()
            e_loss += loss_val
            n_b += 1
        sched.step()
        e_loss /= max(n_b, 1)

        if epoch % 5 == 0:
            va, vap = evaluate(model, idx_va_t, pos_di, pos_ai,
                               drug_feats, N_DRUGS, N_AES)
            history.append({"epoch": epoch, "loss": e_loss,
                            "val_auc": va, "val_ap": vap,
                            "lr": opt.param_groups[0]["lr"]})
            outer.set_postfix(loss=f"{e_loss:.4f}",
                              val_auc=f"{va:.4f}", val_ap=f"{vap:.4f}")
            tqdm.write(f"  Ep{epoch:03d} loss={e_loss:.4f} "
                       f"val_auc={va:.4f} val_ap={vap:.4f}")
            if va > best_val:
                best_val  = va
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # ── Test ───────────────────────────────────────────────────────────────────
    print("\n[F] Test evaluation...")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_auc, test_ap = evaluate(model, idx_te_t, pos_di, pos_ai,
                                 drug_feats, N_DRUGS, N_AES)
    print(f"  Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

    # ── Extract embeddings ─────────────────────────────────────────────────────
    print("\n[G] Extracting embeddings...")
    model.eval()
    with torch.no_grad():
        drug_emb_np = model.encode_drugs(drug_feats).cpu().numpy()
        ae_idx_all  = torch.arange(N_AES, device=DEVICE)
        ae_emb_np   = model.encode_aes(ae_idx_all).cpu().numpy()
    print(f"  Drug embeddings: {drug_emb_np.shape} | AE embeddings: {ae_emb_np.shape}")

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[H] Saving...")
    torch.save(best_state, DRIVE / "protoxnet_bilinear.pt")
    print("  ✅ protoxnet_bilinear.pt")

    with open(DRIVE / "bilinear_embeddings.pkl", "wb") as f:
        pickle.dump({"drug_emb": drug_emb_np, "ae_emb": ae_emb_np,
                     "drug_list": drug_list, "ae_list": ae_list,
                     "tissue_list": tissue_list}, f)
    print("  ✅ bilinear_embeddings.pkl")

    pd.DataFrame(history).to_csv(DRIVE / "bilinear_history.csv", index=False)
    print("  ✅ bilinear_history.csv")

    print(f"\n{'='*60}")
    print("Step 4 complete.")
    print(f"  Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")
    print(f"  Best val AUC: {best_val:.4f} | Params: {n_params:,}")
    print("Next: step5_ctade.py")


if __name__ == "__main__":
    main()
