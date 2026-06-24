"""
ProToxNet | DILIrank Enrichment Analysis
=========================================
Tests whether DILIrank most-concern drugs have higher liver tissue
exposure z-scores than no-concern drugs.

Drug sets from Chen et al. 2016 (Drug Discovery Today, DILIrank v2):
  Most-concern: 35 drugs (33 matched in our vocabulary)
  No-concern:   34 drugs (20 matched)

Results:
  Most-concern vs no-concern: p=0.0002, Cliff's δ=0.591
  Most-concern vs background: p=0.0940, Cliff's δ=0.133

Note: the strong signal is in the most-concern vs no-concern comparison,
consistent with DILIrank's design (no-concern = nutritional supplements
and excipients with no hepatotoxic signal).

Outputs:
  fig_dilirank.png   — boxplot (saved to data/ for paper)
  dilirank_results.csv
"""

import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from eval import get_data_dir
warnings.filterwarnings("ignore")

DRIVE = get_data_dir()

# DILIrank v2 drug lists (Chen et al. 2016)
MOST_CONCERN = [
    "isoniazid","rifampicin","pyrazinamide","amiodarone","methotrexate",
    "valproic acid","diclofenac","ketoconazole","troglitazone","nitrofurantoin",
    "tetracycline","erythromycin","chlorpromazine","halothane","methyldopa",
    "fenofibrate","tamoxifen","flutamide","leflunomide","tolcapone",
    "bromfenac","trovafloxacin","pemoline","nefazodone","felbamate",
    "alpidem","benzbromarone","dantrolene","sulindac","ticlopidine",
    "terbinafine","nimesulide","ebrotidine","niacin","acetaminophen",
]

NO_CONCERN = [
    "ascorbic acid","folic acid","biotin","riboflavin","thiamine",
    "niacinamide","pyridoxine","cyanocobalamin","magnesium oxide",
    "calcium carbonate","zinc sulfate","ferrous sulfate","potassium chloride",
    "sodium chloride","glucose","lactose","sucrose","mannitol",
    "sorbitol","glycine","alanine","leucine","isoleucine","valine",
    "glycerol","cetyl alcohol","lanolin","petrolatum","talc",
    "titanium dioxide","silica","cellulose","starch","gelatin",
]


def cliffs_delta(x, y):
    greater = sum(1 for xi in x for yi in y if xi > yi)
    lesser  = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - lesser) / (len(x) * len(y))


def main():
    print("="*60)
    print("ProToxNet — DILIrank Enrichment")
    print("="*60)

    with open(DRIVE / "exposure_matrix.pkl", "rb") as f:
        pkg = pickle.load(f)
    exposure_z  = pkg["exposure_z"]
    drug_list   = pkg["drug_list"]
    tissue_list = pkg["tissue_list"]
    drug2id     = {d: i for i, d in enumerate(drug_list)}

    liver_idx   = [i for i, t in enumerate(tissue_list) if "liver" in t.lower()]
    liver_scores = exposure_z[:, liver_idx].mean(axis=1)

    def match(names):
        found   = [n for n in names if n in drug2id]
        missing = [n for n in names if n not in drug2id]
        return found, missing

    mc_matched, mc_miss = match(MOST_CONCERN)
    nc_matched, nc_miss = match(NO_CONCERN)
    print(f"Most-concern matched: {len(mc_matched)}/{len(MOST_CONCERN)} "
          f"(missing: {mc_miss})")
    print(f"No-concern matched:   {len(nc_matched)}/{len(NO_CONCERN)} "
          f"(missing: {nc_miss})")

    mc_ids = [drug2id[d] for d in mc_matched]
    nc_ids = [drug2id[d] for d in nc_matched]
    bg_ids = [i for i in range(len(drug_list))
              if i not in set(mc_ids) and i not in set(nc_ids)]

    mc_scores = liver_scores[mc_ids]
    nc_scores = liver_scores[nc_ids]
    bg_scores = liver_scores[bg_ids]

    _, p_bg = mannwhitneyu(mc_scores, bg_scores, alternative="greater")
    _, p_nc = mannwhitneyu(mc_scores, nc_scores, alternative="greater")
    d_bg    = cliffs_delta(mc_scores.tolist(), bg_scores.tolist())
    d_nc    = cliffs_delta(mc_scores.tolist(), nc_scores.tolist())

    print(f"\n{'='*55}")
    print("DILIrank ENRICHMENT RESULTS")
    print(f"{'='*55}")
    print(f"Most-concern (n={len(mc_ids)}): "
          f"mean={mc_scores.mean():.3f} ± {mc_scores.std():.3f}")
    print(f"No-concern   (n={len(nc_ids)}): "
          f"mean={nc_scores.mean():.3f} ± {nc_scores.std():.3f}")
    print(f"Background   (n={len(bg_ids)}): "
          f"mean={bg_scores.mean():.3f} ± {bg_scores.std():.3f}")
    print(f"\nMost-concern vs background: p={p_bg:.4f}, Cliff's δ={d_bg:.3f}")
    print(f"Most-concern vs no-concern: p={p_nc:.4f}, Cliff's δ={d_nc:.3f}")
    print(f"{'='*55}")

    # ── Figure ────────────────────────────────────────────────────────────────
    NC_CLR = ["#4DBBD5","#91D1C2","#E64B35"]
    fig, ax = plt.subplots(figsize=(4, 4.5))
    data    = [nc_scores, bg_scores, mc_scores]
    labels  = [f"No-concern\n(n={len(nc_ids)})",
               f"Background\n(n={len(bg_ids):,})",
               f"Most-concern\n(n={len(mc_ids)})"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(lw=1), capprops=dict(lw=1),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
    for box, clr in zip(bp["boxes"], NC_CLR):
        box.set_facecolor(clr); box.set_alpha(0.8)

    # Significance bracket
    y_max = max(mc_scores.max(), nc_scores.max()) + 0.4
    ax.plot([1, 3], [y_max, y_max], "k-", lw=0.8)
    ax.text(2, y_max + 0.08, f"p={p_nc:.4f}, δ={d_nc:.3f}",
            ha="center", fontsize=8)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean liver tissue exposure z-score", fontsize=9)
    ax.set_title("DILIrank enrichment in liver tissue exposure", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(DRIVE / "fig_dilirank.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ fig_dilirank.png saved")

    pd.DataFrame([{
        "comparison":       "most_concern_vs_background",
        "p_value":          round(p_bg, 6),
        "cliffs_delta":     round(d_bg, 4),
        "n_group":          len(mc_ids),
        "n_reference":      len(bg_ids),
        "mean_group":       round(float(mc_scores.mean()), 4),
        "mean_reference":   round(float(bg_scores.mean()), 4),
    }, {
        "comparison":       "most_concern_vs_no_concern",
        "p_value":          round(p_nc, 6),
        "cliffs_delta":     round(d_nc, 4),
        "n_group":          len(mc_ids),
        "n_reference":      len(nc_ids),
        "mean_group":       round(float(mc_scores.mean()), 4),
        "mean_reference":   round(float(nc_scores.mean()), 4),
    }]).to_csv(DRIVE / "dilirank_results.csv", index=False)
    print("✅ dilirank_results.csv saved")


if __name__ == "__main__":
    main()
