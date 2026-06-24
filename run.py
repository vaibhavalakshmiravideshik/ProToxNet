"""CLI entry point for the ProToxNet pipeline and evaluations."""

import argparse
import importlib
import os
import sys
from pathlib import Path

# ── Allow imports from pipeline/ and eval/ ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


PIPELINE_STEPS = {
    1: ("Data acquisition", "pipeline.step1_data", None),
    2: ("ConPLex proteome-wide scoring", "pipeline.step2_conplex", "main"),
    3: ("Tissue exposure matrix", "pipeline.step3_exposure", "main"),
    4: ("Bilinear model training", "pipeline.step4_train", "main"),
    5: ("CT-ADE external validation", "pipeline.step5_ctade", "main"),
    6: ("Figure generation", "pipeline.step6_figures", "main"),
}

EVAL_STEPS = {
    "ldo": ("eval.eval_ldo", "main"),
    "lao": ("eval.eval_lao", "main"),
    "dilirank": ("eval.eval_dilirank", "main"),
    "baselines": ("eval.eval_baselines", "main"),
    "ablation": ("eval.eval_ablation", "main"),
    "dti": ("eval.eval_dti_sensitivity", "main"),
    "bootstrap": ("eval.eval_bootstrap_ci", "main"),
    "ctade_per_drug": ("eval.eval_ctade_per_drug", "main"),
}


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
        module = importlib.import_module(PIPELINE_STEPS[2][1])
        getattr(module, PIPELINE_STEPS[2][2])()

    if 3 in steps:
        print("\n" + "="*60)
        print("STEP 3: Tissue Exposure Matrix")
        print("="*60)
        module = importlib.import_module(PIPELINE_STEPS[3][1])
        getattr(module, PIPELINE_STEPS[3][2])()

    if 4 in steps:
        print("\n" + "="*60)
        print("STEP 4: Bilinear Model Training")
        print("="*60)
        module = importlib.import_module(PIPELINE_STEPS[4][1])
        getattr(module, PIPELINE_STEPS[4][2])()

    if 5 in steps:
        print("\n" + "="*60)
        print("STEP 5: CT-ADE External Validation")
        print("="*60)
        module = importlib.import_module(PIPELINE_STEPS[5][1])
        getattr(module, PIPELINE_STEPS[5][2])()

    if 6 in steps:
        print("\n" + "="*60)
        print("STEP 6: Figures")
        print("="*60)
        module = importlib.import_module(PIPELINE_STEPS[6][1])
        getattr(module, PIPELINE_STEPS[6][2])()


def run_evals(evals):
    for name in evals:
        if name not in EVAL_STEPS:
            print(f"Unknown eval: {name}. "
                  f"Options: {list(EVAL_STEPS.keys())}")
            continue
        module_path, fn_name = EVAL_STEPS[name]
        print(f"\n{'='*60}")
        print(f"EVAL: {name.upper()}")
        print("="*60)
        mod = importlib.import_module(module_path)
        getattr(mod, fn_name)()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ProToxNet pipeline runner")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=None,
        help="Pipeline steps to run (default: all 1-6 when no --eval is provided)")
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

    if args.drive:
        os.environ["PROTOXNET_DRIVE"] = args.drive

    all_steps = sorted(PIPELINE_STEPS)
    all_evals = list(EVAL_STEPS)
    steps = args.steps if args.steps is not None else (all_steps if not args.eval else [])
    evals = all_evals if "all" in args.eval else args.eval

    if steps:
        run_pipeline(steps)

    if evals:
        run_evals(evals)

    print("\n✅ Done.")
