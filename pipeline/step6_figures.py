"""
ProToxNet | Step 6: Figures and Baseline Table
===============================================
Generates all paper figures and Table 2.

Figures:
  fig2_training_curves.{png,pdf}   — loss + val AUC/AP
  fig3_heatmap.{png,pdf}           — drug-tissue exposure heatmap
  fig4_hepatotoxicity.{png,pdf}    — liver enrichment boxplot
  fig5_ctade_roc.{png,pdf}         — CT-ADE ROC + PR curves
  fig6_kinase.{png,pdf}            — kinase inhibitor AE rank plot
  fig7_baselines.{png,pdf}         — baseline comparison bar chart

Table 2 (baseline_comparison.csv):
  Random | Drug exposure (mean) | AE frequency (FAERS)
  Logistic Regression | ProToxNet (ours)
"""

import pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc as sklearn_auc,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pipeline import get_data_dir
warnings.filterwarnings("ignore")

DRIVE  = get_data_dir()
FIGDIR = DRIVE
FIGDIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nature-style settings
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
NC = ["#E64B35","#4DBBD5","#00A087","#3C5488",
      "#F39B7F","#8491B4","#91D1C2","#DC0000"]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ProToxBilinear(nn.Module):
    def __init__(self, drug_dim=68, n_aes=13200, ae_emb_dim=64, latent=128):
        super().__init__()
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, latent), nn.LayerNorm(latent))
        self.ae_emb  = nn.Embedding(n_aes, ae_emb_dim, max_norm=1.0)
        self.ae_enc  = nn.Sequential(nn.Linear(ae_emb_dim, latent),
                                     nn.LayerNorm(latent))
        self.W         = nn.Parameter(torch.ones(latent))
        self.drug_bias = nn.Embedding(4310, 1)
        self.ae_bias   = nn.Embedding(n_aes, 1)
    def forward(self, df, ai, di):
        dp = self.drug_enc(df)
        ap = self.ae_enc(self.ae_emb(ai))
        return ((dp * self.W * ap).sum(-1)
                + self.drug_bias(di).squeeze(-1)
                + self.ae_bias(ai).squeeze(-1))


def load_data():
    with open(DRIVE / "exposure_matrix.pkl", "rb") as f:
        pkg = pickle.load(f)
    with open(DRIVE / "bilinear_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    faers_df  = pd.read_csv(DRIVE / "drugcentral_faers.csv")
    ctade_df  = pd.read_csv(DRIVE / "ctade_predictions_remapped.csv")
    history   = pd.read_csv(DRIVE / "bilinear_history.csv")
    case_df   = pd.read_csv(DRIVE / "kinase_case_study.csv")
    return pkg, emb, faers_df, ctade_df, history, case_df


def load_model(drug_dim, n_aes):
    model = ProToxBilinear(drug_dim=drug_dim, n_aes=n_aes).to(DEVICE)
    model.load_state_dict(torch.load(DRIVE / "protoxnet_bilinear.pt",
                                     map_location=DEVICE))
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig2_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    ax.plot(history["epoch"], history["loss"], color=NC[0], lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("a", fontweight="bold", loc="left")

    ax = axes[1]
    ax.plot(history["epoch"], history["val_auc"],
            color=NC[1], lw=1.5, label="Val AUC")
    ax.plot(history["epoch"], history["val_ap"],
            color=NC[2], lw=1.5, ls="--", label="Val AP")
    ax.axhline(0.9576, color=NC[0], lw=0.8, ls=":", alpha=0.7,
               label="Test AUC (0.9576)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Metric")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("b", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig2_training_curves.{ext}")
    plt.close()
    print("✅ fig2_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Tissue exposure heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig3_heatmap(exposure_z, drug_list, tissue_list):
    exposure_df = pd.DataFrame(exposure_z, index=drug_list, columns=tissue_list)
    top_drugs   = exposure_df.var(axis=1).nlargest(40).index.tolist()
    top_tissues = exposure_df.loc[top_drugs].std(axis=0).nlargest(20).index.tolist()
    data        = exposure_df.loc[top_drugs, top_tissues]
    vmax        = min(data.values.std() * 2, 2.5)
    clean_t     = [t.replace("_"," ").replace("Mixed Cell","").strip()[:22]
                   for t in top_tissues]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(data, ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=clean_t, yticklabels=top_drugs,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Tissue exposure z-score", "shrink": 0.5})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5)
    ax.set_xlabel("Tissue", fontsize=9); ax.set_ylabel("Drug", fontsize=9)
    ax.set_title("Drug-tissue exposure profiles (top 40 drugs by tissue variance)",
                 fontsize=9, pad=8)
    plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig3_heatmap.{ext}")
    plt.close()
    print("✅ fig3_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Hepatotoxicity enrichment
# ─────────────────────────────────────────────────────────────────────────────
def fig4_hepatotoxicity(exposure_z, drug_list, tissue_list):
    hepatotoxic = ["acetaminophen","isoniazid","methotrexate","amiodarone",
                   "valproic acid","diclofenac","ketoconazole","troglitazone",
                   "nitrofurantoin","rifampicin"]
    exposure_df = pd.DataFrame(exposure_z, index=drug_list, columns=tissue_list)
    liver_cols  = [t for t in tissue_list if "liver" in t.lower()]
    liver_col   = liver_cols[0] if liver_cols else tissue_list[0]
    hep_present = [d for d in hepatotoxic if d in exposure_df.index]
    non_hep     = [d for d in drug_list if d not in set(hepatotoxic)]
    hep_scores  = exposure_df.loc[hep_present, liver_col].values
    non_scores  = exposure_df.loc[non_hep, liver_col].values
    _, pval     = stats.mannwhitneyu(hep_scores, non_scores, alternative="greater")

    fig, ax = plt.subplots(figsize=(3.5, 4))
    bp = ax.boxplot([non_scores, hep_scores], patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(lw=1), capprops=dict(lw=1),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
    bp["boxes"][0].set_facecolor(NC[1]); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(NC[0]); bp["boxes"][1].set_alpha(0.7)
    ax.scatter([2]*len(hep_scores),
               hep_scores + np.random.uniform(-0.05, 0.05, len(hep_scores)),
               color=NC[0], s=30, zorder=5, alpha=0.9)
    y_max = max(hep_scores.max(), non_scores.max()) + 0.3
    ax.plot([1, 2], [y_max, y_max], "k-", lw=0.8)
    ax.text(1.5, y_max + 0.05,
            f"p = {pval:.3f}{'*' if pval < 0.05 else ''}",
            ha="center", fontsize=9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Non-hepatotoxic\n(n=4,300)",
                         f"Hepatotoxic\n(n={len(hep_present)})"])
    ax.set_ylabel(f"Liver tissue exposure z-score\n({liver_col})")
    ax.set_title("Hepatotoxic drug enrichment\nin liver tissue exposure",
                 fontsize=9, pad=8)
    plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig4_hepatotoxicity.{ext}")
    plt.close()
    print(f"✅ fig4_hepatotoxicity (p={pval:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: CT-ADE ROC + PR curves
# ─────────────────────────────────────────────────────────────────────────────
def fig5_ctade_roc(ctade_df):
    y_true = ctade_df["label"].values
    y_pred = ctade_df["pred"].values
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc_val = sklearn_auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    ap_val = average_precision_score(y_true, y_pred)
    prevalence = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    ax = axes[0]
    ax.plot(fpr, tpr, color=NC[0], lw=1.5,
            label=f"ProToxNet (AUC = {roc_auc_val:.4f})")
    ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("a  ROC — CT-ADE external validation",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(rec, prec, color=NC[1], lw=1.5,
            label=f"ProToxNet (AP = {ap_val:.4f})")
    ax.axhline(prevalence, color="k", lw=0.8, ls="--", alpha=0.5,
               label=f"Random (prev = {prevalence:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("b  Precision-recall — CT-ADE",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig5_ctade_roc.{ext}")
    plt.close()
    print(f"✅ fig5_ctade_roc (AUC={roc_auc_val:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Kinase inhibitor AE ranks
# ─────────────────────────────────────────────────────────────────────────────
def fig6_kinase(case_df):
    if case_df.empty:
        print("⚠️  kinase_case_study.csv empty — skipping fig6")
        return
    drugs_present = case_df["drug"].unique()[:6]
    colors_drug   = {d: NC[i] for i, d in enumerate(drugs_present)}

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in case_df.iterrows():
        if row["drug"] not in drugs_present:
            continue
        pct = 100 * (1 - row["rank"] / 13200)
        ax.scatter(row["drug"], pct, color=colors_drug[row["drug"]],
                   s=60, zorder=5, alpha=0.85)
        ax.annotate(str(row["known_ae"])[:20], (row["drug"], pct),
                    textcoords="offset points", xytext=(5,0),
                    fontsize=6, alpha=0.7)
    ax.axhline(99.33, color="k", lw=0.8, ls="--", alpha=0.5,
               label="Top 0.67% (median rank 88)")
    ax.set_ylabel("Percentile rank of known AE\n(higher = better predicted)")
    ax.set_xlabel("Kinase inhibitor"); ax.set_ylim(90, 101)
    ax.set_title("Known adverse event recovery\nfor approved kinase inhibitors",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig6_kinase.{ext}")
    plt.close()
    print("✅ fig6_kinase")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Baseline comparison bar chart  +  Table 2
# ─────────────────────────────────────────────────────────────────────────────
def fig7_baselines(exposure_z, drug_list, faers_df, model, ae2id, drug2id):
    drug_feats = torch.tensor(exposure_z, dtype=torch.float32).to(DEVICE)
    faers_pos  = faers_df[faers_df["llr"] > 0].copy()
    faers_pos["drug_lower"] = faers_pos["drug_name"].str.lower()
    faers_pos  = faers_pos[faers_pos["drug_lower"].isin(drug2id)]
    fv = faers_pos[faers_pos["meddra_name"].isin(ae2id)].copy()
    fv["drug_id"] = fv["drug_lower"].map(drug2id)
    fv["ae_id"]   = fv["meddra_name"].map(ae2id)

    idx = np.arange(len(fv))
    _, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    _, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)
    np.random.seed(42)
    test_pos_di = fv["drug_id"].values[idx_te]
    test_pos_ai = fv["ae_id"].values[idx_te]
    n_te  = len(idx_te)
    neg_di = np.random.randint(0, len(drug_list), n_te)
    neg_ai = np.random.randint(0, len(ae2id), n_te)
    all_di = np.concatenate([test_pos_di, neg_di])
    all_ai = np.concatenate([test_pos_ai, neg_ai])
    y_test = np.concatenate([np.ones(n_te), np.zeros(n_te)])

    X_drug  = exposure_z[all_di]
    ae_counts = fv["ae_id"].value_counts().to_dict()
    X_ae_freq = np.array([ae_counts.get(ai, 0) for ai in all_ai],
                          dtype=np.float32).reshape(-1,1)
    half      = n_te // 2
    train_idx = np.concatenate([np.arange(half), np.arange(n_te, n_te+half)])

    baselines = {}
    baselines["Random"] = {"auc": 0.500, "ap": float(y_test.mean())}

    drug_mean = X_drug.mean(axis=1)
    baselines["Drug exposure (mean)"] = {
        "auc": roc_auc_score(y_test, drug_mean),
        "ap":  average_precision_score(y_test, drug_mean)}

    baselines["AE frequency (FAERS)"] = {
        "auc": roc_auc_score(y_test, X_ae_freq.flatten()),
        "ap":  average_precision_score(y_test, X_ae_freq.flatten())}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_drug)
    lr  = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_s[train_idx], y_test[train_idx])
    lr_p = lr.predict_proba(X_s)[:,1]
    baselines["Logistic Regression"] = {
        "auc": roc_auc_score(y_test, lr_p),
        "ap":  average_precision_score(y_test, lr_p)}

    # Morgan FP + AE freq bar (added to expose pair-level memorisation)
    morgan_mat = np.load(DRIVE / "morgan_mat.npy") if (DRIVE/"morgan_mat.npy").exists() else None
    if morgan_mat is not None:
        X_m = np.hstack([morgan_mat[all_di], X_ae_freq])
        sc2 = StandardScaler(); X_ms = sc2.fit_transform(X_m)
        lr2 = LogisticRegression(max_iter=300, C=0.1, solver="saga",
                                 n_jobs=-1, random_state=42)
        lr2.fit(X_ms[train_idx], y_test[train_idx])
        mp = lr2.predict_proba(X_ms)[:,1]
        baselines["Morgan FP + AE freq\n(pair-level memorisation,\ncollapses to 0.599 under LDO)"] = {
            "auc": roc_auc_score(y_test, mp),
            "ap":  average_precision_score(y_test, mp)}

    # ProToxNet
    all_probs = []
    BATCH = 4096
    with torch.no_grad():
        for s in range(0, len(all_di), BATCH):
            di_b = torch.tensor(all_di[s:s+BATCH], device=DEVICE)
            ai_b = torch.tensor(all_ai[s:s+BATCH], device=DEVICE)
            sc   = model(drug_feats[di_b], ai_b, di_b)
            all_probs.append(torch.sigmoid(sc).cpu().numpy())
    pt_p = np.concatenate(all_probs)
    baselines["ProToxNet (ours)"] = {
        "auc": roc_auc_score(y_test, pt_p),
        "ap":  average_precision_score(y_test, pt_p)}

    # Print
    print("\nBaseline comparison (FAERS held-out test):")
    print(f"  {'Method':<50} {'AUC':>6} {'AP':>6}")
    print(f"  {'-'*64}")
    for name, res in baselines.items():
        marker = " ◀" if "ProToxNet" in name else ""
        print(f"  {name:<50} {res['auc']:>6.4f} {res['ap']:>6.4f}{marker}")

    # Save CSV
    rows = [{"Method": k, "AUC": round(v["auc"],4), "AP": round(v["ap"],4)}
            for k, v in baselines.items()]
    pd.DataFrame(rows).to_csv(DRIVE / "baseline_comparison.csv", index=False)

    # Bar chart
    methods = list(baselines.keys())
    aucs    = [baselines[m]["auc"] for m in methods]
    colors  = [NC[0] if "ProToxNet" in m
               else "#FFA500" if "Morgan" in m
               else "#CCCCCC" for m in methods]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(methods)), aucs, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("AUC (FAERS test set)")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title("ProToxNet vs baselines — FAERS held-out test set", fontsize=9)
    plt.tight_layout()
    for ext in ["pdf","png"]:
        plt.savefig(FIGDIR / f"fig7_baselines.{ext}")
    plt.close()
    print("✅ fig7_baselines + baseline_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("ProToxNet — Step 6: Figures")
    print("="*60)

    pkg, emb, faers_df, ctade_df, history, case_df = load_data()
    exposure_z  = pkg["exposure_z"]
    drug_list   = pkg["drug_list"]
    tissue_list = pkg["tissue_list"]
    ae_list     = emb["ae_list"]
    ae2id       = {a: i for i, a in enumerate(ae_list)}
    drug2id     = {d: i for i, d in enumerate(drug_list)}

    model = load_model(exposure_z.shape[1], len(ae2id))

    fig2_training(history)
    fig3_heatmap(exposure_z, drug_list, tissue_list)
    fig4_hepatotoxicity(exposure_z, drug_list, tissue_list)
    fig5_ctade_roc(ctade_df)
    fig6_kinase(case_df)
    fig7_baselines(exposure_z, drug_list, faers_df, model, ae2id, drug2id)

    print(f"\n✅ All figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
