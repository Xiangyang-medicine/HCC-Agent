#!/usr/bin/env python3
"""Run formal 5x5x5 nested CV training for Phase 3A.

Usage:
    python run_formal_training.py --pilot           # Pilot mode: repeat=1 only (5 folds)
    python run_formal_training.py --full            # Full locked 25-fold training
    python run_formal_training.py --sensitivity SA2 # Sensitivity analysis 2 (pediatric excluded)
    python run_formal_training.py --sensitivity SA3 # Sensitivity analysis 3 (missing stage/grade excluded)
"""

import sys
import argparse
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(PACKAGE_DIR))

# ACM root is 4 levels up from scripts/ folder
project_root = SCRIPT_DIR.parent.parent.parent.resolve()

print(f"Project root: {project_root}")

from prognostic_engine.training import NestedCVTrainer


def main():
    """Run formal nested CV training."""
    parser = argparse.ArgumentParser(description="Phase 3A Nested CV Training")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true",
                      help="Run pilot mode (repeat=1 only, 5 folds)")
    mode.add_argument("--full", action="store_true",
                      help="Run the locked full protocol (5 repeats, 25 folds)")
    mode.add_argument("--sensitivity", choices=["SA2", "SA3"],
                      help="Run sensitivity analysis SA2 or SA3")

    args = parser.parse_args()

    data_path = project_root / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"

    # Determine output directory and splits path based on mode
    if args.sensitivity:
        # Sensitivity analysis - protected output directory
        sa_name = args.sensitivity
        output_dir = project_root / "experiments" / "phase3a" / "sensitivity" / sa_name
        splits_path = project_root / "experiments" / "phase3a" / "sensitivity" / sa_name / "splits" / "outer_splits.csv"
        n_repeats = 5
        print(f"\n{'='*70}")
        print(f"SENSITIVITY ANALYSIS {sa_name}: {n_repeats} repeats × 5 folds")
        print(f"{'='*70}")
        print(f"NOTE: SA output goes to sensitivity/{sa_name}/ (protected directory)")
    elif args.pilot:
        output_dir = project_root / "experiments" / "phase3a" / "pilot"
        splits_path = project_root / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
        n_repeats = 1
        print(f"\n{'='*70}")
        print("PILOT MODE: Running repeat=1 only (5 folds)")
        print(f"{'='*70}")
    else:
        output_dir = project_root / "experiments" / "phase3a" / "formal"
        splits_path = project_root / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
        n_repeats = 5
        print(f"\n{'='*70}")
        print(f"FORMAL TRAINING: {n_repeats} repeats × 5 folds")
        print(f"{'='*70}")

    print(f"Data: {data_path}")
    print(f"Splits: {splits_path}")
    print(f"Output: {output_dir}")

    # Determine sa_name for SA-aware validation
    sa_name = args.sensitivity if args.sensitivity else None
    trainer = NestedCVTrainer(data_path, splits_path, output_dir, sa_name=sa_name)
    trainer.run(n_repeats=n_repeats)


if __name__ == "__main__":
    main()
