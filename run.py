"""
ProToxNet | Full Pipeline Runner
=================================
Runs the complete ProToxNet pipeline end-to-end.
Individual steps can be skipped by commenting them out.

Usage:
    python run.py                        # full pipeline
    python run.py --steps 1 2 3          # specific steps only
    python run.py --eval all             # all eval scripts
    python run.py --eval ldo lao         # specific evals

Drive path:
    Set DRIVE_PATH below to your Google Drive data directory,
    or pass --drive /path/to/data.
"""

import argparse
import sys
from pathlib import Path

# ── Allow imports from pipeline/ and eval/ ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_pipeline(steps):
    if 1 in steps:
        print("\n" + "="*60)
        print("STEP 1: Data Acquisition")
        print("="*60)
        from pipeline.step1_data import (
            fetch_drugcentral_targets, fetch_faers, build_string_ppi,
            build_core_proteins, fetch_kinase_affinities,
            fetch_ctade, build_id_maps
        )
        fetch_drugcentral_targets()
        fetch_faers()
        build_string_ppi()
        build_core_proteins()
        fetch_kinase_affinities()
        fetch_ctade()
        build_id_maps()

    if 2 in steps:
        print("\n" + "="*60)
        print("STEP 2: ConPLex Proteome-Wide Scoring")
        print("="*60)
        from pipeline.step2_conplex import main as step2
        step2()

    if 3 in steps:
        print("\n" + "="*60)
        print("STEP 3: Tissue Exposure Matrix")
        print("="*60)
        from pipeline.step3_exposure import main as step3
        step3()

    if 4 in steps:
        print("\n" + "="*60)
        print("STEP 4: Bilinear Model Training")
        print("="*60)
        from pipeline.step4_train import main as step4
        step4()

    if 5 in steps:
        print("\n" + "="*60)
        print("STEP 5: CT-ADE External Validation")
        print("="*60)
        from pipeline.step5_ctade import main as step5
        step5()

    if 6 in steps:
        print("\n" + "="*60)
        print("STEP 6: Figures")
        print("="*60)
        from pipeline.step6_figures import main as step6
        step6()


def run_evals(evals):
    eval_map = {
        "ldo":          ("eval.eval_ldo",          "main"),
        "lao":          ("eval.eval_lao",           "main"),
        "dilirank":     ("eval.eval_dilirank",      "main"),
        "baselines":    ("eval.eval_baselines",     "main"),
        "ablation":     ("eval.eval_ablation",      "main"),
        "dti":          ("eval.eval_dti_sensitivity","main"),
        "bootstrap":    ("eval.eval_bootstrap_ci",  "main"),
        "ctade_per_drug":("eval.eval_ctade_per_drug","main"),
    }

    for name in evals:
        if name not in eval_map:
            print(f"Unknown eval: {name}. "
                  f"Options: {list(eval_map.keys())}")
            continue
        module_path, fn_name = eval_map[name]
        print(f"\n{'='*60}")
        print(f"EVAL: {name.upper()}")
        print("="*60)
        import importlib
        mod = importlib.import_module(module_path)
        getattr(mod, fn_name)()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ProToxNet pipeline runner")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Pipeline steps to run (default: all 1-6)")
    parser.add_argument(
        "--eval", nargs="+", type=str, default=[],
        help=("Eval scripts to run. Options: ldo, lao, dilirank, "
              "baselines, ablation, dti, bootstrap, ctade_per_drug, all"))
    parser.add_argument(
        "--drive", type=str, default=None,
        help="Override DRIVE path (default: set in each script)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override DRIVE path if provided
    if args.drive:
        import os
        os.environ["PROTOXNET_DRIVE"] = args.drive

    # Resolve "all" for evals
    all_evals = ["ldo","lao","dilirank","baselines",
                 "ablation","dti","bootstrap","ctade_per_drug"]
    evals = all_evals if "all" in args.eval else args.eval

    if args.steps:
        run_pipeline(args.steps)

    if evals:
        run_evals(evals)

    print("\n✅ Done.")
