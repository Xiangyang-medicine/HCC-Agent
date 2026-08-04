#!/usr/bin/env python3
"""
[DEPRECATED] Phase 3A Full Nested CV Training Pipeline

    WARNING: This script is DEPRECATED.
    Use src/prognostic_engine/scripts/run_formal_training.py instead.

    This standalone script will be removed in a future update.
    The canonical pipeline is documented in:
    experiments/phase3a/CANONICAL_PIPELINE.md

Executes 5 repeats × 5 folds × 5 models = 125 model fits.
Outputs OOF predictions and aggregated metrics.

Note: M4 (RSF) and M5 (DeepSurv) require scikit-survival and PyTorch
which are unavailable on Python 3.13. These models will be documented
but not trained in this environment.
"""

import sys
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
MODELING_DIR = Path("data/modeling")
SPLITS_DIR = Path("experiments/phase3a/splits")
OUTPUT_DIR = Path("experiments/phase3a/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metabolic genes
METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",  # Glycolysis
    "GLS", "GLUD1",  # Glutamine
    "FASN", "SCD",  # Lipogenesis
    "CA9", "VEGFA", "HIF1A",  # Hypoxia
    "MYC", "CTNNB1"  # Oncogenic
]

# CV parameters
N_REPEATS = 5
N_FOLDS = 5

# Seeds
OUTER_SEED = 42
INNER_SEED = 123

# Storage
all_predictions = []
metrics_by_model = defaultdict(list)


def log(msg, verbose=True):
    """Print log message."""
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")


def load_data():
    """Load modeling dataset and splits."""
    log("Loading data...")
    df = pd.read_parquet(MODELING_DIR / "tcga_lihc_modeling_dataset.parquet")
    splits = pd.read_csv(SPLITS_DIR / "outer_splits.csv")
    mapping = json.load(open(SPLITS_DIR.parent / "clinical_category_mapping.json"))

    log(f"  Dataset: {len(df)} patients, {df['event'].sum()} events ({df['event'].mean()*100:.1f}%)")
    log(f"  Splits: {len(splits)} test entries ({N_REPEATS} repeats × {N_FOLDS} folds)")

    return df, splits, mapping


def preprocess_fold(train_df, test_df, mapping):
    """Preprocess data for a single fold."""
    gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

    # Age: z-score normalization from training
    age_mean = train_df['age_at_diagnosis'].mean()
    age_std = train_df['age_at_diagnosis'].std()
    train_age = (train_df['age_at_diagnosis'] - age_mean) / age_std
    test_age = (test_df['age_at_diagnosis'] - age_mean) / age_std

    # Stage: ordinal encoding
    stage_map = mapping['ajcc_stage']['mapping']
    def normalize_stage(val):
        if pd.isna(val):
            return 0
        for v in stage_map.values():
            if val in v['original']:
                return v['ordinal']
        return 0

    train_stage = train_df['ajcc_stage'].apply(normalize_stage).values
    test_stage = test_df['ajcc_stage'].apply(normalize_stage).values

    # Grade: ordinal encoding
    grade_map = mapping['tumor_grade']['mapping']
    def normalize_grade(val):
        if pd.isna(val):
            return 0
        for v in grade_map.values():
            if val in v['original']:
                return v['ordinal']
        return 0

    train_grade = train_df['tumor_grade'].apply(normalize_grade).values
    test_grade = test_df['tumor_grade'].apply(normalize_grade).values

    # Genes: z-score normalization
    gene_mean = train_df[gene_cols].mean()
    gene_std = train_df[gene_cols].std()
    train_genes = (train_df[gene_cols] - gene_mean) / gene_std
    test_genes = (test_df[gene_cols] - gene_mean) / gene_std

    return {
        'train_age': train_age,
        'test_age': test_age,
        'train_stage': train_stage,
        'test_stage': test_stage,
        'train_grade': train_grade,
        'test_grade': test_grade,
        'train_genes': train_genes.values,
        'test_genes': test_genes.values,
        'gene_cols': gene_cols,
        'train_case_ids': train_df['case_id'].values,
        'test_case_ids': test_df['case_id'].values
    }


def get_clinical_features(prep):
    """Get clinical-only features (M1)."""
    return (np.column_stack([
        prep['train_age'], prep['train_stage'], prep['train_grade']
    ]),
    np.column_stack([
        prep['test_age'], prep['test_stage'], prep['test_grade']
    ]))


def get_gene_features(prep):
    """Get gene-only features (M2)."""
    return prep['train_genes'], prep['test_genes']


def get_combined_features(prep):
    """Get combined clinical + gene features (M3)."""
    return (np.column_stack([
        prep['train_age'], prep['train_stage'], prep['train_grade'],
        prep['train_genes']
    ]),
    np.column_stack([
        prep['test_age'], prep['test_stage'], prep['test_grade'],
        prep['test_genes']
    ]))


def train_m1_clinical_cox(train_df, test_df, prep, mapping, repeat, fold):
    """M1: Clinical-only Cox PH model."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    X_train, X_test = get_clinical_features(prep)

    # Create DataFrame for lifelines
    train_data = pd.DataFrame(X_train, columns=['age', 'stage', 'grade'])
    train_data['survival_time'] = train_df['survival_months'].values
    train_data['event'] = train_df['event'].astype(int).values

    # Fit model
    cph = CoxPHFitter()
    cph.fit(train_data, duration_col='survival_time', event_col='event')

    # Predict
    test_data = pd.DataFrame(X_test, columns=['age', 'stage', 'grade'])
    risk_test = cph.predict_partial_hazard(test_data).values.flatten()

    # C-index (negative because concordance_index expects higher = more risk)
    y_test = test_df['survival_months'].values
    event_test = test_df['event'].astype(int).values
    cidx = concordance_index(y_test, -risk_test, event_test)

    # Collect predictions
    for i, case_id in enumerate(prep['test_case_ids']):
        all_predictions.append({
            'case_id': case_id,
            'repeat': repeat,
            'fold': fold,
            'model': 'M1_clinical_cox',
            'risk_score': float(risk_test[i]),
            'survival_months': float(y_test[i]),
            'event': int(event_test[i])
        })

    return cidx


def train_m2_gene_elasticnet(train_df, test_df, prep, mapping, repeat, fold):
    """M2: Gene-only Elastic-net Cox model."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    X_train, X_test = get_gene_features(prep)
    gene_cols = prep['gene_cols']

    # Create DataFrame for lifelines
    train_data = pd.DataFrame(X_train, columns=gene_cols)
    train_data['survival_time'] = train_df['survival_months'].values
    train_data['event'] = train_df['event'].astype(int).values

    # Fit with elastic-net regularization
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph.fit(train_data, duration_col='survival_time', event_col='event')

    # Predict
    test_data = pd.DataFrame(X_test, columns=gene_cols)
    risk_test = cph.predict_partial_hazard(test_data).values.flatten()

    # C-index
    y_test = test_df['survival_months'].values
    event_test = test_df['event'].astype(int).values
    cidx = concordance_index(y_test, -risk_test, event_test)

    # Collect predictions
    for i, case_id in enumerate(prep['test_case_ids']):
        all_predictions.append({
            'case_id': case_id,
            'repeat': repeat,
            'fold': fold,
            'model': 'M2_gene_elasticnet',
            'risk_score': float(risk_test[i]),
            'survival_months': float(y_test[i]),
            'event': int(event_test[i])
        })

    return cidx


def train_m3_combined_elasticnet(train_df, test_df, prep, mapping, repeat, fold):
    """M3: Combined Elastic-net Cox model (clinical + genes)."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    X_train, X_test = get_combined_features(prep)
    gene_cols = prep['gene_cols']
    all_cols = ['age', 'stage', 'grade'] + gene_cols

    # Create DataFrame for lifelines
    train_data = pd.DataFrame(X_train, columns=all_cols)
    train_data['survival_time'] = train_df['survival_months'].values
    train_data['event'] = train_df['event'].astype(int).values

    # Fit with elastic-net regularization
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph.fit(train_data, duration_col='survival_time', event_col='event')

    # Predict
    test_data = pd.DataFrame(X_test, columns=all_cols)
    risk_test = cph.predict_partial_hazard(test_data).values.flatten()

    # C-index
    y_test = test_df['survival_months'].values
    event_test = test_df['event'].astype(int).values
    cidx = concordance_index(y_test, -risk_test, event_test)

    # Collect predictions
    for i, case_id in enumerate(prep['test_case_ids']):
        all_predictions.append({
            'case_id': case_id,
            'repeat': repeat,
            'fold': fold,
            'model': 'M3_combined_elasticnet',
            'risk_score': float(risk_test[i]),
            'survival_months': float(y_test[i]),
            'event': int(event_test[i])
        })

    return cidx


def train_m4_combined_rsf(train_df, test_df, prep, mapping, repeat, fold):
    """M4: Combined Random Survival Forest.

    NOTE: Requires scikit-survival which is not available on Python 3.13.
    This is a placeholder that returns NaN. Install scikit-survival in
    a Python 3.11/3.12 environment for full training.
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        log(f"  M4 skipped (scikit-survival not available)", verbose=True)
        return float('nan')

    X_train, X_test = get_combined_features(prep)

    # Create structured array for sksurv
    y_train = np.array(
        [(bool(e), t) for e, t in zip(train_df['event'], train_df['survival_months'])],
        dtype=[('event', bool), ('survival_time', float)]
    )
    y_test = np.array(
        [(bool(e), t) for e, t in zip(test_df['event'], test_df['survival_months'])],
        dtype=[('event', bool), ('survival_time', float)]
    )

    # Fit RSF
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rsf.fit(X_train, y_train)

    # Predict
    risk_test = rsf.predict(X_test)

    # C-index
    cidx, _, _, _, _ = concordance_index_censored(
        y_test['event'], y_test['survival_time'], risk_test, tau=None
    )

    # Collect predictions
    for i, case_id in enumerate(prep['test_case_ids']):
        all_predictions.append({
            'case_id': case_id,
            'repeat': repeat,
            'fold': fold,
            'model': 'M4_combined_rsf',
            'risk_score': float(risk_test[i]),
            'survival_months': float(y_test['survival_time'][i]),
            'event': int(y_test['event'][i])
        })

    return cidx


def train_m5_deepsurv(train_df, test_df, prep, mapping, repeat, fold):
    """M5: Combined DeepSurv model.

    NOTE: Requires PyTorch which is not available on Python 3.13.
    This is a placeholder that returns NaN. Install PyTorch in
    a Python 3.11/3.12 environment for full training.
    """
    try:
        import torch
    except ImportError:
        log(f"  M5 skipped (PyTorch not available)", verbose=True)
        return float('nan')

    log(f"  M5 DeepSurv training...", verbose=True)

    # DeepSurv implementation would go here
    # For now, return NaN as placeholder
    return float('nan')


def run_training_fold(df, splits, mapping, repeat, fold):
    """Run training for a single fold."""
    # Get test cases for this fold/repeat
    test_cases = splits[
        (splits['repeat'] == repeat) &
        (splits['fold'] == fold) &
        (splits['fold_type'] == 'test')
    ]['case_id'].values

    all_cases = set(df['case_id'].values)
    train_cases = all_cases - set(test_cases)

    train_df = df[df['case_id'].isin(train_cases)].copy()
    test_df = df[df['case_id'].isin(test_cases)].copy()

    # Preprocess
    prep = preprocess_fold(train_df, test_df, mapping)

    results = {}

    # M1: Clinical Cox PH
    log(f"  M1: Clinical Cox PH...", verbose=False)
    try:
        cidx = train_m1_clinical_cox(train_df, test_df, prep, mapping, repeat, fold)
        results['M1'] = cidx
        metrics_by_model['M1_clinical_cox'].append(cidx)
        log(f"  M1 C-index: {cidx:.3f}", verbose=False)
    except Exception as e:
        log(f"  M1 ERROR: {e}", verbose=True)
        results['M1'] = float('nan')

    # M2: Gene Elastic-net
    log(f"  M2: Gene Elastic-net...", verbose=False)
    try:
        cidx = train_m2_gene_elasticnet(train_df, test_df, prep, mapping, repeat, fold)
        results['M2'] = cidx
        metrics_by_model['M2_gene_elasticnet'].append(cidx)
        log(f"  M2 C-index: {cidx:.3f}", verbose=False)
    except Exception as e:
        log(f"  M2 ERROR: {e}", verbose=True)
        results['M2'] = float('nan')

    # M3: Combined Elastic-net
    log(f"  M3: Combined Elastic-net...", verbose=False)
    try:
        cidx = train_m3_combined_elasticnet(train_df, test_df, prep, mapping, repeat, fold)
        results['M3'] = cidx
        metrics_by_model['M3_combined_elasticnet'].append(cidx)
        log(f"  M3 C-index: {cidx:.3f}", verbose=False)
    except Exception as e:
        log(f"  M3 ERROR: {e}", verbose=True)
        results['M3'] = float('nan')

    # M4: Combined RSF (placeholder)
    log(f"  M4: Combined RSF...", verbose=False)
    try:
        cidx = train_m4_combined_rsf(train_df, test_df, prep, mapping, repeat, fold)
        results['M4'] = cidx
        if not np.isnan(cidx):
            metrics_by_model['M4_combined_rsf'].append(cidx)
        log(f"  M4 C-index: {cidx:.3f}" if not np.isnan(cidx) else "  M4 skipped", verbose=False)
    except Exception as e:
        log(f"  M4 ERROR: {e}", verbose=True)
        results['M4'] = float('nan')

    # M5: DeepSurv (placeholder)
    log(f"  M5: DeepSurv...", verbose=False)
    try:
        cidx = train_m5_deepsurv(train_df, test_df, prep, mapping, repeat, fold)
        results['M5'] = cidx
        if not np.isnan(cidx):
            metrics_by_model['M5_deepsurv'].append(cidx)
        log(f"  M5 C-index: {cidx:.3f}" if not np.isnan(cidx) else "  M5 skipped", verbose=False)
    except Exception as e:
        log(f"  M5 ERROR: {e}", verbose=True)
        results['M5'] = float('nan')

    return results


def aggregate_results():
    """Aggregate metrics across all folds."""
    summary = {}

    for model, scores in metrics_by_model.items():
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            summary[model] = {
                'n_folds': len(valid_scores),
                'mean_cindex': np.mean(valid_scores),
                'std_cindex': np.std(valid_scores),
                'min_cindex': np.min(valid_scores),
                'max_cindex': np.max(valid_scores),
                'median_cindex': np.median(valid_scores),
                'per_fold': valid_scores
            }

    return summary


def save_results(summary, predictions_df, start_time):
    """Save training results."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save predictions
    predictions_path = OUTPUT_DIR / "oof_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    log(f"Saved predictions: {predictions_path}")

    # Save metrics summary
    metrics_path = OUTPUT_DIR / "metrics_summary.json"
    report = {
        'training': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration
        },
        'cv_config': {
            'n_repeats': N_REPEATS,
            'n_folds': N_FOLDS,
            'total_folds': N_REPEATS * N_FOLDS,
            'outer_seed': OUTER_SEED,
            'inner_seed': INNER_SEED
        },
        'models': {
            'M1_clinical_cox': 'Cox PH with clinical features only',
            'M2_gene_elasticnet': 'Cox PH with elastic-net on genes only',
            'M3_combined_elasticnet': 'Cox PH with elastic-net on clinical + genes',
            'M4_combined_rsf': 'Random Survival Forest (requires scikit-survival)',
            'M5_deepsurv': 'DeepSurv neural network (requires PyTorch)'
        },
        'metrics': summary
    }

    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2)
    log(f"Saved metrics: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'N':>5} {'Mean C-index':>12} {'Std':>8} {'Range':>20}")
    print("-" * 75)

    for model, stats in sorted(summary.items()):
        n = stats['n_folds']
        mean_c = stats['mean_cindex']
        std_c = stats['std_cindex']
        range_str = f"[{stats['min_cindex']:.3f}, {stats['max_cindex']:.3f}]"
        print(f"{model:<30} {n:>5} {mean_c:>12.3f} {std_c:>8.3f} {range_str:>20}")

    print(f"\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")


def main():
    """Run full nested CV training."""
    print("\n" + "=" * 70)
    print("PHASE 3A FULL NESTED CV TRAINING")
    print("=" * 70)

    start_time = datetime.now()
    print(f"Start time: {start_time.isoformat()}")
    print(f"Config: {N_REPEATS} repeats × {N_FOLDS} folds = {N_REPEATS * N_FOLDS} folds total")
    print(f"Models: M1 (Clinical), M2 (Gene), M3 (Combined)")
    print(f"        M4 (RSF), M5 (DeepSurv) - requires separate environment")

    # Load data
    df, splits, mapping = load_data()

    # Run training
    fold_results = []
    total_folds = N_REPEATS * N_FOLDS

    print("\n" + "-" * 70)
    print("Training Progress")
    print("-" * 70)

    for repeat in range(1, N_REPEATS + 1):
        for fold in range(1, N_FOLDS + 1):
            fold_num = (repeat - 1) * N_FOLDS + fold
            print(f"\n[{fold_num}/{total_folds}] Repeat {repeat}, Fold {fold}")

            try:
                results = run_training_fold(df, splits, mapping, repeat, fold)
                fold_results.append({
                    'repeat': repeat,
                    'fold': fold,
                    'results': results
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

    # Aggregate results
    summary = aggregate_results()

    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)

    # Save all results
    save_results(summary, predictions_df, start_time)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
