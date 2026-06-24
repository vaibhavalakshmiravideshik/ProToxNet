"""
ProToxNet | Step 3: Expression-Weighted Tissue Exposure Matrix
==============================================================
Formula:
    E(drug, tissue) = Σ_p  score(drug, p) × expr(p, tissue)

where the sum is over all 1,507 core proteins.
High E → drug binds proteins highly expressed in that tissue.

Inputs:
  conplex_scores_raw.csv   — drug × protein binding scores
  hpa_tissue_expression.csv — GTEx TPM (log1p-normalised)
  core_proteins.csv

Outputs:
  exposure_matrix.pkl      — {exposure, exposure_z, drug_list,
                               tissue_list, protein_list}
  exposure_matrix.csv      — (4310 × 68) human-readable
  exposure_topk.csv        — top-10 tissues per drug
  drug_tissue_zscore.csv   — z-scored matrix
"""

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from pipeline import get_data_dir
warnings.filterwarnings("ignore")

DRIVE = get_data_dir()


def main():
    print("="*60)
    print("ProToxNet — Step 3: Tissue Exposure Matrix")
    print("="*60)

    # ── Load ConPLex scores ────────────────────────────────────────────────────
    print("\n[A] Loading ConPLex scores...")
    scores_df = pd.read_csv(DRIVE / "conplex_scores_raw.csv")
    score_col = ("calibrated_prob" if "calibrated_prob" in scores_df.columns
                 else "conplex_score")
    print(f"  Pairs: {len(scores_df):,} | Score col: '{score_col}'")

    score_matrix = scores_df.pivot(index="drug", columns="protein",
                                   values=score_col)
    drug_list    = score_matrix.index.tolist()
    protein_list = score_matrix.columns.tolist()
    score_np     = np.nan_to_num(score_matrix.values.astype(np.float32), nan=0.0)
    print(f"  Score matrix: {score_np.shape}")

    # ── Load expression ────────────────────────────────────────────────────────
    print("\n[B] Loading tissue expression...")
    expr_df = pd.read_csv(DRIVE / "hpa_tissue_expression.csv")
    print(f"  Rows: {len(expr_df):,} | Columns: {list(expr_df.columns)}")

    uniprot_col = next((c for c in expr_df.columns
                        if "uniprot" in c.lower()), expr_df.columns[0])
    tissue_col  = next((c for c in expr_df.columns
                        if "tissue" in c.lower()), expr_df.columns[1])
    expr_col    = next((c for c in expr_df.columns
                        if any(x in c.lower() for x in ["tpm","expr","score"])),
                       expr_df.columns[2])
    print(f"  Columns → uniprot='{uniprot_col}' tissue='{tissue_col}' "
          f"expr='{expr_col}'")

    expr_df = expr_df[expr_df[uniprot_col].isin(protein_list)]
    tissues = sorted(expr_df[tissue_col].unique())
    print(f"  Tissues: {len(tissues)} | Proteins covered: "
          f"{expr_df[uniprot_col].nunique():,}")

    # ── Build protein × tissue matrix ─────────────────────────────────────────
    print("\n[C] Building expression matrix (proteins × tissues)...")
    expr_pivot = (expr_df
                  .groupby([uniprot_col, tissue_col])[expr_col]
                  .mean()
                  .unstack(fill_value=0.0))
    expr_pivot = expr_pivot.reindex(protein_list, fill_value=0.0)
    expr_np    = expr_pivot.values.astype(np.float32)
    tissue_list = expr_pivot.columns.tolist()
    print(f"  Expression matrix: {expr_np.shape}")

    # ── Compute exposure E = S · X ─────────────────────────────────────────────
    print("\n[D] Computing exposure matrix E = score · expr...")
    exposure_np = score_np @ expr_np          # (n_drugs, n_tissues)
    print(f"  Exposure matrix: {exposure_np.shape}")
    print(f"  Range: {exposure_np.min():.2f} – {exposure_np.max():.2f} | "
          f"Mean: {exposure_np.mean():.2f}")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    print("\n[E] Sanity checks...")
    exposure_df  = pd.DataFrame(exposure_np, index=drug_list, columns=tissue_list)
    liver_cols   = [t for t in tissue_list if "liver" in t.lower()]
    if liver_cols:
        top10 = exposure_df[liver_cols[0]].nlargest(10)
        print(f"  Top-10 liver exposure drugs:\n{top10.to_string()}")

    known_ki = ["imatinib","erlotinib","sorafenib","sunitinib","lapatinib"]
    present  = [d for d in known_ki if d in exposure_df.index]
    for drug in present[:3]:
        top3 = dict(exposure_df.loc[drug].nlargest(3).round(1))
        print(f"  {drug}: {top3}")

    # ── Z-score per tissue ─────────────────────────────────────────────────────
    print("\n[F] Z-score normalisation (per tissue column)...")
    exposure_z = np.nan_to_num(
        stats.zscore(exposure_np, axis=0), nan=0.0)
    print(f"  Z-score range: {exposure_z.min():.2f} – {exposure_z.max():.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[G] Saving...")
    with open(DRIVE / "exposure_matrix.pkl", "wb") as f:
        pickle.dump({
            "exposure":   exposure_np,
            "exposure_z": exposure_z,
            "drug_list":  drug_list,
            "tissue_list": tissue_list,
            "protein_list": protein_list,
        }, f)
    print("  ✅ exposure_matrix.pkl")

    exposure_df.to_csv(DRIVE / "exposure_matrix.csv")
    print(f"  ✅ exposure_matrix.csv {exposure_df.shape}")

    topk_rows = []
    for drug in drug_list:
        for rank, (tissue, val) in enumerate(
                exposure_df.loc[drug].nlargest(10).items(), 1):
            topk_rows.append({"drug": drug, "rank": rank,
                              "tissue": tissue, "exposure": val})
    pd.DataFrame(topk_rows).to_csv(DRIVE / "exposure_topk.csv", index=False)
    print(f"  ✅ exposure_topk.csv ({len(topk_rows):,} rows)")

    pd.DataFrame(exposure_z, index=drug_list,
                 columns=tissue_list).to_csv(DRIVE / "drug_tissue_zscore.csv")
    print("  ✅ drug_tissue_zscore.csv")

    print(f"\n{'='*60}")
    print("Step 3 complete.")
    print(f"  Drugs: {len(drug_list):,} | Tissues: {len(tissue_list)}")
    print("Next: step4_train.py")


if __name__ == "__main__":
    main()
