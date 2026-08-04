#!/usr/bin/env python3
"""Generate sensitivity analysis splits from original outer_splits.csv.

Usage:
    python generate_sa_splits.py --sa2   # SA2: exclude pediatric patients (age < 18)
    python generate_sa_splits.py --sa3   # SA3: exclude missing stage/grade patients

This script:
1. Loads the original outer_splits.csv
2. Filters out patients based on SA exclusion criteria
3. Maintains original repeat/fold assignments
4. Saves to sensitivity/{SA}/splits/outer_splits.csv
5. Computes SHA-256 hash for integrity tracking
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(PACKAGE_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_data():
    """Load the original dataset and splits."""
    data_path = PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    splits_path = PROJECT_ROOT / "experiments" / "phase3a" / "splits" / "outer_splits.csv"

    df = pd.read_parquet(data_path)
    splits = pd.read_csv(splits_path)

    return df, splits


def generate_sa2_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Generate SA2 splits: exclude pediatric patients (age < 18)."""
    # Identify pediatric patients
    pediatric_threshold = 18  # years
    pediatric_mask = df['age_at_diagnosis'] < pediatric_threshold
    pediatric_case_ids = set(df[pediatric_mask]['case_id'].values)

    print(f"SA2: Excluding {len(pediatric_case_ids)} pediatric patients (age < 18)")
    for cid in pediatric_case_ids:
        row = df[df['case_id'] == cid].iloc[0]
        print(f"  - {row['submitter_id']} (age={row['age_at_diagnosis']})")

    # Filter splits
    sa_splits = splits[~splits['case_id'].isin(pediatric_case_ids)].copy()

    # Verify no orphaned repeats/folds
    for repeat in range(1, 6):
        for fold in range(1, 6):
            test_cases = sa_splits[
                (sa_splits['repeat'] == repeat) &
                (sa_splits['fold'] == fold) &
                (sa_splits['fold_type'] == 'test')
            ]
            if len(test_cases) == 0:
                print(f"  WARNING: Repeat {repeat}, Fold {fold} has no test cases!")

    return sa_splits


def generate_sa3_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Generate SA3 splits: exclude patients missing stage or grade."""
    # Identify patients with missing stage or grade
    missing_mask = df['ajcc_stage'].isna() | df['tumor_grade'].isna()
    missing_case_ids = set(df[missing_mask]['case_id'].values)

    print(f"SA3: Excluding {len(missing_case_ids)} patients with missing stage/grade")
    for cid in missing_case_ids:
        row = df[df['case_id'] == cid].iloc[0]
        stage = row['ajcc_stage'] if pd.notna(row['ajcc_stage']) else 'NaN'
        grade = row['tumor_grade'] if pd.notna(row['tumor_grade']) else 'NaN'
        print(f"  - {row['submitter_id']}: stage={stage}, grade={grade}")

    # Filter splits
    sa_splits = splits[~splits['case_id'].isin(missing_case_ids)].copy()

    # Verify no orphaned repeats/folds
    for repeat in range(1, 6):
        for fold in range(1, 6):
            test_cases = sa_splits[
                (sa_splits['repeat'] == repeat) &
                (sa_splits['fold'] == fold) &
                (sa_splits['fold_type'] == 'test')
            ]
            if len(test_cases) == 0:
                print(f"  WARNING: Repeat {repeat}, Fold {fold} has no test cases!")

    return sa_splits


def save_splits(sa_name: str, sa_splits: pd.DataFrame, original_splits: pd.DataFrame):
    """Save SA splits and compute SHA-256 hashes."""
    # Create output directory
    output_dir = PROJECT_ROOT / "experiments" / "phase3a" / "sensitivity" / sa_name / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    output_path = output_dir / "outer_splits.csv"
    sa_splits.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Compute SHA-256 of original and new splits
    original_path = PROJECT_ROOT / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
    original_hash = compute_sha256(original_path)
    new_hash = compute_sha256(output_path)

    # Verify original hash matches expected
    expected_original = "7b21074e208a563bc99b6a0a8c458f076a5ab333612e65ce2659cd9d6571228f"
    if original_hash != expected_original:
        print(f"  WARNING: Original splits SHA-256 mismatch!")
        print(f"    Expected: {expected_original}")
        print(f"    Actual:   {original_hash}")

    # Save metadata
    metadata = {
        "sa_name": sa_name,
        "original_sha256": original_hash,
        "sa_splits_sha256": new_hash,
        "original_n_patients": len(original_splits['case_id'].unique()),
        "sa_n_patients": len(sa_splits['case_id'].unique()),
        "excluded_patients": len(original_splits['case_id'].unique()) - len(sa_splits['case_id'].unique()),
        "n_test_per_fold": sa_splits[sa_splits['fold_type'] == 'test'].groupby(['repeat', 'fold']).size().mean()
    }

    metadata_path = output_dir / "metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate sensitivity analysis splits")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sa2", action="store_true", help="Generate SA2 splits (exclude pediatric)")
    mode.add_argument("--sa3", action="store_true", help="Generate SA3 splits (exclude missing stage/grade)")

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE SENSITIVITY ANALYSIS SPLITS")
    print("=" * 70)

    # Load data
    df, original_splits = load_data()
    print(f"Original dataset: {len(df)} patients")
    print(f"Original splits: {len(original_splits)} rows")
    print()

    # Generate SA splits
    if args.sa2:
        sa_name = "SA2"
        print(f"\n{'='*70}")
        print(f"GENERATING {sa_name} SPLITS")
        print(f"{'='*70}")
        sa_splits = generate_sa2_splits(df, original_splits)
    elif args.sa3:
        sa_name = "SA3"
        print(f"\n{'='*70}")
        print(f"GENERATING {sa_name} SPLITS")
        print(f"{'='*70}")
        sa_splits = generate_sa3_splits(df, original_splits)

    # Save and verify
    metadata = save_splits(sa_name, sa_splits, original_splits)

    # Summary
    print()
    print("=" * 70)
    print(f"{sa_name} SUMMARY")
    print("=" * 70)
    print(f"Original patients: {metadata['original_n_patients']}")
    print(f"{sa_name} patients:   {metadata['sa_n_patients']}")
    print(f"Excluded:         {metadata['excluded_patients']}")
    print(f"Mean test/fold:   {metadata['n_test_per_fold']:.1f}")
    print()
    print(f"Predictions expected: {metadata['sa_n_patients']} × 5 = {metadata['sa_n_patients'] * 5} per model")
    print(f"Total predictions:   {metadata['sa_n_patients']} × 5 × 5 = {metadata['sa_n_patients'] * 25} (5 models)")


if __name__ == "__main__":
    main()
