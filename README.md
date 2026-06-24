# ProToxNet

**Expression-Aware Proteome-Wide Binding for Tissue-Stratified Adverse Event Prediction**

> Code will be released upon acceptance.  
> Paper under review at *npj Digital Medicine*.

## Overview

ProToxNet predicts drug–adverse event (AE) associations by combining:
- **ConPLex** proteome-wide drug–protein binding scores (ESM-1b + Morgan FP)
- **GTEx** tissue expression profiles (68 tissues)
- A **bilinear interaction model** (DistMult-style) trained on DrugCentral FAERS signals

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_data.py` | Download DrugCentral, FAERS, STRING PPI, CT-ADE |
| 2 | `step2_conplex.py` | ESM-1b protein embeddings + ConPLex scoring |
| 3 | `step3_exposure.py` | Drug × tissue exposure matrix (S · X) |
| 4 | `step4_train.py` | Train bilinear ProToxNet model |
| 5 | `step5_ctade.py` | CT-ADE external validation + AE vocabulary remapping |
| 6 | `step6_figures.py` | All paper figures + Table 2 baselines |

## Evaluation

| Script | Experiment |
|--------|------------|
| `eval_ldo.py` | Leave-drug-out cold-start (AUC 0.8544) |
| `eval_lao.py` | Leave-AE-out (AUC 0.8472) |
| `eval_dilirank.py` | DILIrank enrichment (p=0.0002, Cliff's δ=0.591) |
| `eval_baselines.py` | Fair baseline comparison (Table 2) |
| `eval_ablation.py` | Bias terms + label smoothing ablation (Table 3) |
| `eval_dti_sensitivity.py` | DTI sensitivity ablation |
| `eval_bootstrap_ci.py` | Bootstrap 95% CIs (FAERS + CT-ADE) |
| `eval_ctade_per_drug.py` | Per-trial-arm AUC histogram |

## Key Results

| Metric | Value |
|--------|-------|
| FAERS pair-level AUC | 0.9576 (95% CI: 0.9565–0.9588) |
| FAERS pair-level AP | 0.9569 |
| FAERS LDO AUC | 0.8544 |
| FAERS LAO AUC | 0.8472 |
| CT-ADE external AUC | 0.9157 (95% CI: 0.9131–0.9182) |
| CT-ADE per-trial-arm AUC (mean/median) | 0.968 / 0.984 (n=497 arms) |
| DILIrank most-concern vs no-concern | p=0.0002, Cliff's δ=0.591 |
| Kinase inhibitor known-AE median rank | 88 / 13,200 (top 0.67%) |

## Requirements

```
pip install -r requirements.txt
```

Requires Google Drive mount at `/content/drive/MyDrive/ProToxNet/data/` (Colab)  
or set `DRIVE` path in each script.

## Data

All data is publicly available:
- **DrugCentral**: https://unmtid-dbs.net (PostgreSQL) or https://drugcentral.org/download
- **GTEx v10 / HPA**: https://www.proteinatlas.org/download
- **STRING v12**: https://string-db.org/cgi/download
- **CT-ADE**: https://figshare.com/articles/dataset/CT-ADE/28142453
- **ConPLex checkpoint**: https://cb.csail.mit.edu/cb/conplex/data/models/

## Citation

```bibtex
@article{ravideshik2025protoxnet,
  title={Expression-Aware Proteome-Wide Binding for Tissue-Stratified Adverse Event Prediction},
  author={Ravideshik, Vaibhava Lakshmi},
  journal={npj Digital Medicine},
  year={2025},
  note={Under review}
}
```
