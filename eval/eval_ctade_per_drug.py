"""
ProToxNet | CT-ADE Per-Trial-Arm AUC
======================================
Computes per-trial-arm AUC from ctade_predictions_remapped.csv.
The predictions file has block structure:
  rows [i*18877 : (i+1)*18877] = predictions for drug i
  (756 drugs × 18,877 AE columns = 14,271,012 total rows)

Arms with no positive labels (n=259) are excluded.

Results:
  Arms evaluated : 497
  Mean AUC       : 0.9679
  Median AUC     : 0.9844
  Std            : 0.0514
  > AUC 0.8      : 489 / 497 (98.4%)
  Min / Max       : 0.5233 / 0.9996

Outputs:
  fig_ctade_per_drug_auc.png   — histogram (saved to data/)
  ctade_per_drug_auc.csv       — per-arm AUC values
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from eval import get_data_dir
warnings.filterwarnings("ignore")

DRIVE = get_data_dir()


def main():
    print("="*60)
    print("ProToxNet — CT-ADE Per-Trial-Arm AUC")
    print("="*60)

    preds = pd.read_csv(DRIVE / "ctade_predictions_remapped.csv")
    print(f"Predictions loaded: {len(preds):,} rows")

    N_AES     = 18877          # AE columns per drug block
    pred_arr  = preds["pred"].values
    label_arr = preds["label"].values
    n_drugs   = len(pred_arr) // N_AES
    print(f"Drug blocks: {n_drugs} × {N_AES} AEs = {n_drugs*N_AES:,} pairs")

    per_drug_aucs = []
    skipped       = 0

    for i in range(n_drugs):
        s      = i * N_AES
        e      = s + N_AES
        y_pred = pred_arr[s:e]
        y_true = label_arr[s:e]
        n_pos  = y_true.sum()
        if n_pos < 1 or n_pos == N_AES:
            skipped += 1
            continue
        try:
            per_drug_aucs.append(roc_auc_score(y_true, y_pred))
        except Exception:
            skipped += 1

    per_drug_aucs = np.array(per_drug_aucs)
    n_eval = len(per_drug_aucs)

    print(f"\nTrial arms evaluated : {n_eval}  (skipped: {skipped})")
    print(f"Mean AUC             : {per_drug_aucs.mean():.4f}")
    print(f"Median AUC           : {np.median(per_drug_aucs):.4f}")
    print(f"Std                  : {per_drug_aucs.std():.4f}")
    print(f"Min                  : {per_drug_aucs.min():.4f}")
    print(f"Max                  : {per_drug_aucs.max():.4f}")
    print(f"> 0.8  : {(per_drug_aucs > 0.8).sum()} "
          f"({100*(per_drug_aucs > 0.8).mean():.1f}%)")
    print(f"> 0.7  : {(per_drug_aucs > 0.7).sum()} "
          f"({100*(per_drug_aucs > 0.7).mean():.1f}%)")
    print(f"< 0.5  : {(per_drug_aucs < 0.5).sum()} "
          f"({100*(per_drug_aucs < 0.5).mean():.1f}%)")

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_drug_aucs, bins=30, color="#2c7bb6",
            edgecolor="white", alpha=0.85)
    ax.axvline(per_drug_aucs.mean(), color="#d7191c", lw=1.8, ls="--",
               label=f"Mean = {per_drug_aucs.mean():.3f}")
    ax.axvline(np.median(per_drug_aucs), color="#fdae61", lw=1.8, ls=":",
               label=f"Median = {np.median(per_drug_aucs):.3f}")
    ax.axvline(0.5, color="black", lw=1.0, alpha=0.4, label="Random (0.5)")
    ax.set_xlabel("Per-trial-arm AUC (CT-ADE)", fontsize=12)
    ax.set_ylabel("Number of trial arms", fontsize=12)
    ax.set_title(f"CT-ADE per-trial-arm AUC "
                 f"(n={n_eval} arms with ≥1 positive)", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig_path = DRIVE / "fig_ctade_per_drug_auc.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ {fig_path.name} saved")

    # ── Save per-arm AUCs ─────────────────────────────────────────────────────
    pd.DataFrame({"arm_index": np.where(
                      np.array([label_arr[i*N_AES:(i+1)*N_AES].sum()
                                for i in range(n_drugs)]) >= 1)[0],
                  "auc": per_drug_aucs}).to_csv(
        DRIVE / "ctade_per_drug_auc.csv", index=False)
    print("✅ ctade_per_drug_auc.csv saved")

    # ── LaTeX sentence ────────────────────────────────────────────────────────
    print(f"\nLaTeX sentence for Section 2.4:")
    print(f"Per-trial-arm AUC across the {n_eval} evaluated arms "
          f"({skipped} arms with no positive labels were excluded) "
          f"yielded mean AUC {per_drug_aucs.mean():.3f} and median AUC "
          f"{np.median(per_drug_aucs):.3f}, with "
          f"{(per_drug_aucs > 0.8).sum()} of {n_eval} arms "
          f"({100*(per_drug_aucs > 0.8).mean():.1f}\\%) exceeding AUC 0.8, "
          f"confirming that the aggregate AUC 0.9157 reflects consistent "
          f"ranking performance across individual trial arms rather than "
          f"averaging over heterogeneous quality.")


if __name__ == "__main__":
    main()
