# ProToxNet

ProToxNet is a tissue-aware adverse event prediction pipeline that combines proteome-wide drug–protein binding, tissue-specific expression, and bilinear drug–AE modeling.

## Repository structure

```text
ProToxNet/
├── README.md
├── requirements.txt
├── .gitignore
├── run.py
├── pipeline/
│   ├── __init__.py
│   ├── step1_data.py
│   ├── step2_conplex.py
│   ├── step3_exposure.py
│   ├── step4_train.py
│   ├── step5_ctade.py
│   └── step6_figures.py
├── eval/
│   ├── __init__.py
│   ├── eval_ldo.py
│   ├── eval_lao.py
│   ├── eval_dilirank.py
│   ├── eval_baselines.py
│   ├── eval_ablation.py
│   ├── eval_dti_sensitivity.py
│   ├── eval_bootstrap_ci.py
│   └── eval_ctade_per_drug.py
└── data/
    ├── .gitkeep
    └── README.md
```

## Usage

Run the full pipeline:

```bash
python run.py
```

Run selected pipeline steps:

```bash
python run.py --steps 1 2 3
python run.py --steps 4 5 6
```

Run evaluations only:

```bash
python run.py --eval ldo lao
python run.py --eval all
```

Override the default runtime data directory:

```bash
python run.py --drive /path/to/data --steps 1 2 3
```

By default, outputs are written to `data/` inside the repository.

## Installation

```bash
pip install -r requirements.txt
```

## Notes

- Large datasets, checkpoints, matrices, and generated figures are not committed.
- Runtime data layout and expected files are documented in `data/README.md`.
- The main orchestration entry point is `run.py`.
