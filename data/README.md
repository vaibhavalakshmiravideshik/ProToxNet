# ProToxNet data directory

All data files live here at runtime and are intentionally excluded from git.

## Directory layout at runtime

```text
data/
├── drugcentral_drug_target.csv     DrugCentral MoA targets (step1)
├── drugcentral_faers.csv           FAERS LLR/ROR signals (step1)
├── drugcentral_faers_female.csv    FAERS female-stratified (step1)
├── drugcentral_faers_male.csv      FAERS male-stratified (step1)
├── drugcentral_faers_ger.csv       FAERS geriatric-stratified (step1)
├── hpa_tissue_expression.csv       GTEx/HPA TPM (assumed present, step1)
├── string_ppi.csv                  STRING v12 PPI edges (step1)
├── core_proteins.csv               DC ∩ GTEx ∩ STRING proteins (step1)
├── bindingdb_kinase.csv            BindingDB/ChEMBL kinase affinities (step1)
├── ctade_drug_ae.csv               CT-ADE benchmark test split (step1)
├── id_maps.pkl                     Unified integer ID maps (step1)
│
├── checkpoints/                    Cached ConPLex checkpoints
├── protein_sequences.pkl           UniProt FASTA sequences (step2)
├── protein_esm_embeddings.pkl      ESM-1b mean-pool embeddings (step2)
├── conplex_scores_raw.csv          Drug × protein scores 6.5M pairs (step2)
├── conplex_scores_matrix.pkl       Pivot matrix drugs × proteins (step2)
├── conplex_calibration.pkl         Platt scaling params (step2)
├── conplex_positives.csv           Known pairs with scores (step2)
├── conplex_scoring_checkpoint.pkl  Scoring resume checkpoint (step2)
│
├── exposure_matrix.pkl             Drug × tissue exposure matrix (step3)
├── exposure_matrix.csv             Human-readable (4310 × 68) (step3)
├── exposure_topk.csv               Top-10 tissues per drug (step3)
├── drug_tissue_zscore.csv          Z-scored exposure matrix (step3)
│
├── protoxnet_bilinear.pt           Best model checkpoint (step4)
├── bilinear_embeddings.pkl         Drug + AE embeddings (step4)
├── bilinear_history.csv            Training curves (step4)
│
├── ctade_ae_mapping.csv            CT-ADE AE → FAERS ae2id map (step5)
├── ctade_predictions_remapped.csv  CT-ADE predictions 14.27M pairs (step5)
│
├── fig_ctade_per_drug_auc.png      Per-trial-arm AUC histogram (eval)
├── fig_dilirank.png                DILIrank enrichment boxplot (eval)
├── fig_ldo_per_drug_auc.png        LDO per-drug AUC histogram (eval)
├── morgan_mat.npy                  Morgan FP matrix 4310 × 2048 (eval)
│
├── bootstrap_ci.csv                Bootstrap 95% CIs (eval)
├── ldo_results.csv                 LDO cold-start AUC/AP (eval)
├── lao_results.csv                 LAO AUC/AP (eval)
├── dilirank_results.csv            DILIrank enrichment stats (eval)
├── fair_baselines.csv              Baseline comparison Table 2 (eval)
├── ablation.csv                    Ablation Table 3 (eval)
├── dti_sensitivity.csv             DTI sensitivity ablation (eval)
├── baseline_comparison.csv         Step6 baseline bar chart data (eval)
├── kinase_case_study.csv           Kinase inhibitor AE ranks (step5)
└── results_summary.csv             Summary of key metrics
```

## Data sources

| File | Source | URL |
|------|--------|-----|
| drugcentral_*.csv | DrugCentral PostgreSQL | drugcentral.org / unmtid-dbs.net |
| hpa_tissue_expression.csv | Human Protein Atlas / GTEx v10 | proteinatlas.org |
| 9606.protein.*.gz | STRING v12 | stringdb-downloads.org |
| ctade_drug_ae.csv | CT-ADE | figshare.com/articles/dataset/28142453 |
| ConPLex checkpoint | ConPLex | cb.csail.mit.edu/cb/conplex/data/models/ |

## Approximate storage

| Stage | Approx size |
|-------|-------------|
| Step 1 raw data | ~400 MB |
| Step 2 embeddings + scores | ~2 GB |
| Step 3 exposure matrices | ~50 MB |
| Step 4 model artifacts | ~20 MB |
| Step 5 CT-ADE predictions | ~250 MB |
| Total | ~3 GB |
