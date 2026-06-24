"""
ProToxNet | Bootstrap 95% Confidence Intervals
===============================================
1000-iteration percentile bootstrap on:
  FAERS test set   — AUC 0.9576 (95% CI: 0.9565–0.9588)
                     AP  0.9569 (95% CI: 0.9553–0.9584)
  CT-ADE external  — AUC 0.9157 (95% CI: 0.9131–0.9182)
                     AP  0.0100 (95% CI: 0.0095–0.0105)

Note: CT-ADE AP is low (0.01) because prevalence is ~0.07%
(14.27M pairs, 9,660 positives). AUC is the appropriate metric.

Outputs:
  bootstrap_ci.csv   — {dataset, metric, mean, ci_lo, ci_hi}
"""

import pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from eval import get_data_dir
warnings.filterwarnings("ignore")

DRIVE  = get_data_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 1000
SEED   = 42


class ProToxBilinear(nn.Module):
    def __init__(self, drug_dim=68, n_aes=13200, ae_emb_dim=64,
                 latent=128, n_drugs=4310):
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


def bootstrap_ci(y_true, y_score, metric_fn,
                 n=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    n_samp = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samp, size=n_samp)
        yt  = y_true[idx]; ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, ys))
    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return float(np.mean(scores)), float(lo), float(hi)


def get_faers_predictions(exposure_z, drug_list, ae2id, drug2id, faers_df):
    """Rebuild FAERS test-set predictions from the saved checkpoint."""
    faers_pos = faers_df[faers_df["llr"] > 0].copy()
    faers_pos["drug_lower"] = faers_pos["drug_name"].str.lower()
    faers_pos = faers_pos[faers_pos["drug_lower"].isin(drug2id)]
    fv = faers_pos[faers_pos["meddra_name"].isin(ae2id)].copy()
    fv["drug_id"] = fv["drug_lower"].map(drug2id)
    fv["ae_id"]   = fv["meddra_name"].map(ae2id)

    idx = np.arange(len(fv))
    _, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    _, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    np.random.seed(42)
    te_pos_di = torch.tensor(fv["drug_id"].values[idx_te], dtype=torch.long).to(DEVICE)
    te_pos_ai = torch.tensor(fv["ae_id"].values[idx_te],   dtype=torch.long).to(DEVICE)
    n_te = len(idx_te)
    neg_di = torch.randint(0, len(drug_list), (n_te,), device=DEVICE)
    neg_ai = torch.randint(0, len(ae2id),     (n_te,), device=DEVICE)

    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    model = ProToxBilinear(drug_dim=exposure_z.shape[1],
                           n_aes=len(ae2id), n_drugs=len(drug_list)).to(DEVICE)
    model.load_state_dict(torch.load(DRIVE / "protoxnet_bilinear.pt",
                                     map_location=DEVICE))
    model.eval()

    all_probs, all_labels = [], []
    BATCH = 4096
    with torch.no_grad():
        for s in range(0, n_te, BATCH):
            sl = slice(s, s+BATCH)
            ps = torch.sigmoid(model(drug_feats[te_pos_di[sl]],
                                     te_pos_ai[sl], te_pos_di[sl]))
            ns = torch.sigmoid(model(drug_feats[neg_di[sl]],
                                     neg_ai[sl], neg_di[sl]))
            all_probs.append(torch.cat([ps, ns]).cpu().numpy())
            all_labels.append(np.concatenate([np.ones(len(ps)),
                                              np.zeros(len(ns))]))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    print("="*60)
    print("ProToxNet — Bootstrap 95% CIs")
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
    faers_df = pd.read_csv(DRIVE / "drugcentral_faers.csv")

    # ── FAERS ─────────────────────────────────────────────────────────────────
    print(f"\n[A] FAERS test-set predictions ({N_BOOT} bootstrap iterations)...")
    faers_probs, faers_labels = get_faers_predictions(
        exposure_z, drug_list, ae2id, drug2id, faers_df)
    print(f"  Pairs: {len(faers_labels):,} | "
          f"Pos: {faers_labels.sum():.0f}")

    f_auc_mu, f_auc_lo, f_auc_hi = bootstrap_ci(
        faers_labels, faers_probs, roc_auc_score)
    f_ap_mu,  f_ap_lo,  f_ap_hi  = bootstrap_ci(
        faers_labels, faers_probs, average_precision_score)

    print(f"  FAERS AUC: {f_auc_mu:.4f} (95% CI: {f_auc_lo:.4f}–{f_auc_hi:.4f})")
    print(f"  FAERS AP:  {f_ap_mu:.4f}  (95% CI: {f_ap_lo:.4f}–{f_ap_hi:.4f})")

    # ── CT-ADE ────────────────────────────────────────────────────────────────
    print(f"\n[B] CT-ADE predictions ({N_BOOT} bootstrap iterations)...")
    ctade = pd.read_csv(DRIVE / "ctade_predictions_remapped.csv")
    ctade_probs  = ctade["pred"].values
    ctade_labels = ctade["label"].values
    print(f"  Pairs: {len(ctade_labels):,} | "
          f"Pos: {ctade_labels.sum():.0f} | "
          f"Prevalence: {100*ctade_labels.mean():.3f}%")

    c_auc_mu, c_auc_lo, c_auc_hi = bootstrap_ci(
        ctade_labels, ctade_probs, roc_auc_score)
    c_ap_mu,  c_ap_lo,  c_ap_hi  = bootstrap_ci(
        ctade_labels, ctade_probs, average_precision_score)

    print(f"  CT-ADE AUC: {c_auc_mu:.4f} (95% CI: {c_auc_lo:.4f}–{c_auc_hi:.4f})")
    print(f"  CT-ADE AP:  {c_ap_mu:.4f}  (95% CI: {c_ap_lo:.4f}–{c_ap_hi:.4f})")
    print(f"  (AP is low because prevalence ≈ 0.07%; AUC is the primary metric)")

    # ── LaTeX-ready output ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COPY INTO TABLE 1 (LaTeX)")
    print(f"{'='*60}")
    print(f"FAERS AUC: {f_auc_mu:.4f} "
          f"(95\\% CI: {f_auc_lo:.4f}--{f_auc_hi:.4f})")
    print(f"FAERS AP:  {f_ap_mu:.4f} "
          f"(95\\% CI: {f_ap_lo:.4f}--{f_ap_hi:.4f})")
    print(f"CT-ADE AUC: {c_auc_mu:.4f} "
          f"(95\\% CI: {c_auc_lo:.4f}--{c_auc_hi:.4f})")
    print(f"CT-ADE AP:  {c_ap_mu:.4f} "
          f"(95\\% CI: {c_ap_lo:.4f}--{c_ap_hi:.4f})")

    pd.DataFrame([
        {"dataset":"FAERS",  "metric":"AUC", "mean":f_auc_mu,
         "ci_lo":f_auc_lo, "ci_hi":f_auc_hi},
        {"dataset":"FAERS",  "metric":"AP",  "mean":f_ap_mu,
         "ci_lo":f_ap_lo,  "ci_hi":f_ap_hi},
        {"dataset":"CT-ADE", "metric":"AUC", "mean":c_auc_mu,
         "ci_lo":c_auc_lo, "ci_hi":c_auc_hi},
        {"dataset":"CT-ADE", "metric":"AP",  "mean":c_ap_mu,
         "ci_lo":c_ap_lo,  "ci_hi":c_ap_hi},
    ]).to_csv(DRIVE / "bootstrap_ci.csv", index=False)
    print("✅ bootstrap_ci.csv saved")


if __name__ == "__main__":
    main()
