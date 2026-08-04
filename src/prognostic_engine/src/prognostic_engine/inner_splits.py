"""Inner CV split generation for nested CV hyperparameter tuning.

Per Phase 3A reset: Split management uses EXPLICIT case_id lists, not row indices.

This ensures:
1. Case IDs are preserved through preprocessing pipeline
2. Inner fold data extraction by case_id (not position)
3. Full reproducibility via case_id round-trip
4. No implicit reliance on DataFrame row order
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

from prognostic_engine.config import N_INNER_FOLDS, INNER_SEED, OUTPUT_DIR


def generate_inner_splits(outer_train_case_ids, repeat, outer_fold, n_folds=N_INNER_FOLDS, seed=INNER_SEED):
    """
    Generate inner CV splits using EXPLICIT case_id lists.

    Per Phase 3A reset: Returns case_id lists, not row indices.

    Parameters
    ----------
    outer_train_case_ids : array-like
        Case IDs in the outer training set (list or array of case_id values)
    repeat : int
        Outer repeat number
    outer_fold : int
        Outer fold number
    n_folds : int
        Number of inner CV folds
    seed : int
        Random seed

    Returns
    -------
    inner_splits : dict
        Dictionary with case_id lists per fold:
        {
            'repeat': repeat,
            'outer_fold': outer_fold,
            'case_ids': list of all case_ids,
            'folds': [
                {
                    'fold': fold_idx,
                    'train_case_ids': [...],
                    'val_case_ids': [...]
                },
                ...
            ]
        }
    """
    # Convert to list for stable indexing
    outer_train_case_ids = list(outer_train_case_ids)
    n = len(outer_train_case_ids)

    # Create index-based KFold for reproducible splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    inner_splits = {
        'repeat': repeat,
        'outer_fold': outer_fold,
        'case_ids': outer_train_case_ids,  # Full list for reference
        'folds': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n)), start=1):
        # Convert indices to case_id lists
        train_case_ids = [outer_train_case_ids[i] for i in train_idx]
        val_case_ids = [outer_train_case_ids[i] for i in val_idx]

        inner_splits['folds'].append({
            'fold': fold_idx,
            'train_case_ids': train_case_ids,
            'val_case_ids': val_case_ids,
            # Keep indices for backward compatibility (may be removed later)
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })

    return inner_splits


def save_inner_splits(inner_splits, output_dir=None):
    """
    Save inner splits to CSV with case_id assignments.

    Per Phase 3A reset: CSV uses case_id column, enabling round-trip verification.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "inner_splits"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repeat = inner_splits['repeat']
    outer_fold = inner_splits['outer_fold']

    # Save as CSV with case_id column (not row indices)
    assignments = []
    for fold_info in inner_splits['folds']:
        fold = fold_info['fold']

        # Train case_ids
        for case_id in fold_info['train_case_ids']:
            assignments.append({
                'case_id': case_id,
                'repeat': repeat,
                'outer_fold': outer_fold,
                'inner_fold': fold,
                'fold_type': 'train'
            })

        # Val case_ids
        for case_id in fold_info['val_case_ids']:
            assignments.append({
                'case_id': case_id,
                'repeat': repeat,
                'outer_fold': outer_fold,
                'inner_fold': fold,
                'fold_type': 'val'
            })

    df = pd.DataFrame(assignments)
    csv_path = output_dir / f"inner_assignments_repeat_{repeat}_fold_{outer_fold}.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def load_inner_splits(repeat, outer_fold, output_dir=None):
    """
    Load inner splits from CSV using case_id column.

    Per Phase 3A reset: Returns case_id lists, not row indices.
    This enables proper round-trip verification and DataFrame-based extraction.

    Parameters
    ----------
    repeat : int
        Outer repeat number
    outer_fold : int
        Outer fold number
    output_dir : Path, optional
        Directory containing CSV files

    Returns
    -------
    inner_splits : dict
        Inner splits with case_id lists per fold
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "inner_splits"
    output_dir = Path(output_dir)

    csv_path = output_dir / f"inner_assignments_repeat_{repeat}_fold_{outer_fold}.csv"
    df = pd.read_csv(csv_path)

    # Verify case_id column exists (per Phase 3A reset)
    if 'case_id' not in df.columns:
        raise ValueError(
            f"CSV file {csv_path} missing 'case_id' column. "
            "Ensure splits were saved with case_id-based assignment."
        )

    # Get unique case_ids in order of first appearance
    case_ids_ordered = df['case_id'].unique().tolist()

    inner_splits = {
        'repeat': repeat,
        'outer_fold': outer_fold,
        'case_ids': case_ids_ordered,
        'folds': []
    }

    for fold in range(1, N_INNER_FOLDS + 1):
        fold_df = df[df['inner_fold'] == fold]

        # Extract case_id lists (not indices!)
        train_case_ids = fold_df[fold_df['fold_type'] == 'train']['case_id'].tolist()
        val_case_ids = fold_df[fold_df['fold_type'] == 'val']['case_id'].tolist()

        inner_splits['folds'].append({
            'fold': fold,
            'train_case_ids': train_case_ids,
            'val_case_ids': val_case_ids
        })

    return inner_splits


def verify_case_id_roundtrip(original_case_ids, inner_splits):
    """
    Verify that case_ids can be recovered from inner_splits.

    Per Phase 3A reset: This ensures no case_id loss during split generation.

    Parameters
    ----------
    original_case_ids : list
        Original case_ids passed to generate_inner_splits()
    inner_splits : dict
        Generated inner splits

    Returns
    -------
    bool
        True if all case_ids are preserved
    """
    # Each case_id should appear in exactly one train and one val per inner fold
    # (for n_folds=5, each case_id appears in 4 train folds and 1 val fold)

    original_set = set(original_case_ids)
    recovered_set = set(inner_splits['case_ids'])

    if original_set != recovered_set:
        missing = original_set - recovered_set
        extra = recovered_set - original_set
        raise ValueError(
            f"Case ID mismatch:\n"
            f"  Missing from inner_splits: {missing}\n"
            f"  Extra in inner_splits: {extra}"
        )

    # Verify each case_id appears in correct number of folds
    n_folds = N_INNER_FOLDS
    for case_id in original_case_ids:
        train_count = 0
        val_count = 0
        for fold_info in inner_splits['folds']:
            if case_id in fold_info['train_case_ids']:
                train_count += 1
            if case_id in fold_info['val_case_ids']:
                val_count += 1

        # Each case should be in (n_folds-1) train folds and 1 val fold
        if train_count != n_folds - 1 or val_count != 1:
            raise ValueError(
                f"Case {case_id} has incorrect fold assignments: "
                f"train_count={train_count}, val_count={val_count} "
                f"(expected train={n_folds-1}, val=1)"
            )

    return True


def extract_inner_fold_data(df, train_case_ids, val_case_ids):
    """
    Extract inner fold DataFrames by case_id.

    Per Phase 3A reset: This is the canonical way to get inner fold data
    from the full training DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with 'case_id' column
    train_case_ids : list
        Case IDs for inner training set
    val_case_ids : list
        Case IDs for inner validation set

    Returns
    -------
    tuple
        (inner_train_df, inner_val_df)
    """
    inner_train_df = df[df['case_id'].isin(train_case_ids)].copy()
    inner_val_df = df[df['case_id'].isin(val_case_ids)].copy()

    # Verify expected counts (optional but helpful for debugging)
    n_train_expected = len(train_case_ids)
    n_val_expected = len(val_case_ids)

    if len(inner_train_df) != n_train_expected:
        # Check for duplicate case_ids in df
        duplicate_cases = df[df.duplicated('case_id', keep=False)]['case_id'].unique()
        if len(duplicate_cases) > 0:
            raise ValueError(
                f"DataFrame has duplicate case_ids: {duplicate_cases[:5]}..."
            )
        raise ValueError(
            f"Inner train count mismatch: expected {n_train_expected}, got {len(inner_train_df)}"
        )

    if len(inner_val_df) != n_val_expected:
        raise ValueError(
            f"Inner val count mismatch: expected {n_val_expected}, got {len(inner_val_df)}"
        )

    return inner_train_df, inner_val_df
