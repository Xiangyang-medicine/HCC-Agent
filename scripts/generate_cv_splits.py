#!/usr/bin/env python3
"""Generate Phase 3A nested CV splits with proper stratification."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Constants
MODELING_DIR = Path("data/modeling")
SPLITS_DIR = Path("experiments/phase3a/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Seeds
OUTER_SEED = 42
INNER_SEED = 123
DEEPSURV_SEED = 456

# CV parameters
N_OUTER_FOLDS = 5
N_OUTER_REPEATS = 5
N_INNER_FOLDS = 5

def create_outer_splits(df, n_folds=5, n_repeats=5, seed=42):
    """Create stratified outer CV splits."""
    splits = []
    event = df['event'].values

    for repeat in range(n_repeats):
        rng = np.random.RandomState(seed + repeat)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 10000))

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(df)), event)):
            for idx in test_idx:
                splits.append({
                    'case_id': df.iloc[idx]['case_id'],
                    'submitter_id': df.iloc[idx]['submitter_id'],
                    'repeat': repeat + 1,
                    'fold': fold + 1,
                    'fold_type': 'test' if idx in test_idx else 'train',
                    'event': df.iloc[idx]['event']
                })

    return pd.DataFrame(splits)

def create_outer_splits_v2(df, n_folds=5, n_repeats=5, seed=42):
    """
    Create proper outer CV splits where each patient appears exactly once in test per repeat.
    Returns: DataFrame with columns: case_id, submitter_id, repeat, fold, fold_type, event
    """
    splits = []
    event = df['event'].values
    case_ids = df['case_id'].values
    submitter_ids = df['submitter_id'].values

    for repeat in range(n_repeats):
        # Create stratified k-fold
        rng = np.random.RandomState(seed + repeat)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 10000))

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(df)), event)):
            for idx in test_idx:
                splits.append({
                    'case_id': case_ids[idx],
                    'submitter_id': submitter_ids[idx],
                    'repeat': repeat + 1,
                    'fold': fold + 1,
                    'fold_type': 'test',
                    'event': int(event[idx])
                })

    return pd.DataFrame(splits)

def verify_splits(splits_df, n_patients, n_repeats, n_folds):
    """Verify split integrity."""
    errors = []

    # Check each repeat has exactly n_patients in test
    for repeat in range(1, n_repeats + 1):
        repeat_data = splits_df[splits_df['repeat'] == repeat]
        test_data = repeat_data[repeat_data['fold_type'] == 'test']
        if len(test_data) != n_patients:
            errors.append(f"Repeat {repeat}: {len(test_data)} test samples (expected {n_patients})")

        # Check each patient appears exactly once per repeat
        unique_patients = test_data['case_id'].nunique()
        if unique_patients != n_patients:
            errors.append(f"Repeat {repeat}: {unique_patients} unique patients in test (expected {n_patients})")

        # Check each patient appears only once in test per repeat
        for patient in test_data['case_id'].values:
            patient_count = (test_data['case_id'] == patient).sum()
            if patient_count != 1:
                errors.append(f"Repeat {repeat}: Patient {patient} appears {patient_count} times in test")

    # Check each patient appears in train for 4 folds per repeat
    for repeat in range(1, n_repeats + 1):
        repeat_data = splits_df[splits_df['repeat'] == repeat]
        for patient in splits_df['case_id'].unique():
            test_patients_in_repeat = repeat_data[
                (repeat_data['fold_type'] == 'test') &
                (repeat_data['case_id'] == patient)
            ]
            # Patient should be in exactly one test fold
            if len(test_patients_in_repeat) != 1:
                errors.append(f"Repeat {repeat}: Patient {patient} in {len(test_patients_in_repeat)} test folds")

    return errors

def main():
    print("=" * 70)
    print("Generating Phase 3A Nested CV Splits")
    print("=" * 70)

    # Load modeling dataset
    df = pd.read_parquet(MODELING_DIR / "tcga_lihc_modeling_dataset.parquet")
    print(f"\nDataset: {len(df)} patients, {df['event'].sum()} events")

    # Create outer splits
    print(f"\nCreating outer splits: {N_OUTER_REPEATS} repeats × {N_OUTER_FOLDS} folds")
    outer_splits = create_outer_splits_v2(df, n_folds=N_OUTER_FOLDS, n_repeats=N_OUTER_REPEATS, seed=OUTER_SEED)

    print(f"Outer splits generated: {len(outer_splits)} rows")
    print(f"  (Should be: {len(df)} patients × {N_OUTER_REPEATS} repeats = {len(df) * N_OUTER_REPEATS})")

    # Verify splits
    print("\nVerifying split integrity...")
    errors = verify_splits(outer_splits, len(df), N_OUTER_REPEATS, N_OUTER_FOLDS)
    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("  All checks passed!")

    # Save outer splits
    outer_splits_path = SPLITS_DIR / "outer_splits.csv"
    outer_splits.to_csv(outer_splits_path, index=False)
    print(f"\nSaved: {outer_splits_path}")

    # Create inner CV config
    inner_config = {
        "n_inner_folds": N_INNER_FOLDS,
        "seed": INNER_SEED,
        "stratify": "event",
        "purpose": "Hyperparameter tuning within outer training fold"
    }

    # Create full config
    config = {
        "outer_cv": {
            "n_folds": N_OUTER_FOLDS,
            "n_repeats": N_OUTER_REPEATS,
            "seed": OUTER_SEED,
            "stratify": "event"
        },
        "inner_cv": inner_config,
        "seeds": {
            "outer": OUTER_SEED,
            "inner": INNER_SEED,
            "deepsurv": DEEPSURV_SEED,
            "numpy": OUTER_SEED,
            "torch": DEEPSURV_SEED
        },
        "total_outer_test_sets": N_OUTER_FOLDS * N_OUTER_REPEATS,
        "created_at": datetime.now().isoformat()
    }

    config_path = SPLITS_DIR / "inner_split_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Split Summary")
    print("=" * 70)
    print(f"\nOuter CV:")
    print(f"  Folds: {N_OUTER_FOLDS}")
    print(f"  Repeats: {N_OUTER_REPEATS}")
    print(f"  Total test sets: {N_OUTER_FOLDS * N_OUTER_REPEATS}")

    for repeat in range(1, N_OUTER_REPEATS + 1):
        repeat_data = outer_splits[outer_splits['repeat'] == repeat]
        test_data = repeat_data[repeat_data['fold_type'] == 'test']
        event_rate = test_data['event'].mean()
        print(f"\n  Repeat {repeat}:")
        for fold in range(1, N_OUTER_FOLDS + 1):
            fold_data = test_data[test_data['fold'] == fold]
            n = len(fold_data)
            n_event = fold_data['event'].sum()
            print(f"    Fold {fold}: N={n}, Events={n_event} ({n_event/n*100:.1f}%)")

    print(f"\nInner CV: {N_INNER_FOLDS} folds (for hyperparameter tuning)")
    print(f"Seeds: outer={OUTER_SEED}, inner={INNER_SEED}, deepsurv={DEEPSURV_SEED}")

if __name__ == "__main__":
    main()
