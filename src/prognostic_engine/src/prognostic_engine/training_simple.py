"""Simplified Phase 3A training - Cox models only (avoids RSF segfault)."""

import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

# sklearn 1.8 + scikit-survival 0.28 compatibility shim
import sys
class _BootstrapCompat:
    @staticmethod
    def _get_n_samples_bootstrap(n_samples, max_samples, sample_weight=None):
        from sklearn.ensemble._forest import _get_n_samples_bootstrap as orig
        return orig(n_samples, max_samples)
sys.modules['sklearn.ensemble._bootstrap'] = _BootstrapCompat()

from prognostic_engine.config import (
    METABOLIC_GENES, N_OUTER_REPEATS, N_OUTER_FOLDS, N_INNER_FOLDS,
    OUTER_SEED, INNER_SEED, EVALUATION_TIMES, OUTPUT_DIR
)
from prognostic_engine.preprocessing import (
    preprocess_fold_clinical, preprocess_fold_genes
)
from prognostic_engine.inner_splits import generate_inner_splits, save_inner_splits
from prognostic_engine.models import M1ClinicalCox, M2M3Coxnet
from prognostic_engine.metrics import (
    compute_all_metrics, harrell_c_index, uno_c_index
)
from prognostic_engine.bootstrap import patient_level_paired_bootstrap

warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', 'all coefficients are zero')


def main():
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    splits_path = project_root / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
    output_dir = project_root / "experiments" / "phase3a" / "formal"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 3A SIMPLIFIED NESTED CV (Cox Models Only)")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(data_path)
    splits = pd.read_csv(splits_path)
    print(f"\nDataset: {len(df)} patients, {df['event'].sum()} events ({df['event'].mean()*100:.1f}%)")
    print(f"Outer splits: {len(splits)} test entries")

    all_predictions = []
    metrics_by_model = defaultdict(list)

    start_time = datetime.now()
    total_folds = N_OUTER_REPEATS * N_OUTER_FOLDS

    print(f"\nProtocol: {N_OUTER_REPEATS} repeats × {N_OUTER_FOLDS} folds × {N_INNER_FOLDS} inner folds")
    print("Models: M1 (Clinical Cox), M2 (Gene Coxnet), M3 (Combined Coxnet)")
    print("-" * 70)

    for repeat in range(1, N_OUTER_REPEATS + 1):
        for fold in range(1, N_OUTER_FOLDS + 1):
            fold_num = (repeat - 1) * N_OUTER_FOLDS + fold
            print(f"\n[{fold_num}/{total_folds}] Repeat {repeat}, Fold {fold}")

            # Get test cases
            test_cases = splits[
                (splits['repeat'] == repeat) &
                (splits['fold'] == fold) &
                (splits['fold_type'] == 'test')
            ]['case_id'].values

            all_cases = set(df['case_id'].values)
            train_cases = all_cases - set(test_cases)

            train_df = df[df['case_id'].isin(train_cases)].copy()
            test_df = df[df['case_id'].isin(test_cases)].copy()

            print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

            # Preprocess
            clinical_prep = preprocess_fold_clinical(train_df, test_df)
            gene_prep = preprocess_fold_genes(train_df, test_df)

            X_clinical_train = clinical_prep['clinical_train']
            X_clinical_test = clinical_prep['clinical_test']
            X_gene_train = gene_prep['train_genes']
            X_gene_test = gene_prep['test_genes']
            X_combined_train = np.hstack([X_clinical_train, X_gene_train])
            X_combined_test = np.hstack([X_clinical_test, X_gene_test])
            combined_cols = clinical_prep['clinical_cols'] + gene_prep['gene_cols']

            y_train = train_df['survival_months'].values
            y_test = test_df['survival_months'].values
            e_train = train_df['event'].values
            e_test = test_df['event'].values
            test_case_ids = test_df['case_id'].values

            # Generate inner splits
            train_case_ids_list = list(train_cases)
            inner_splits = generate_inner_splits(train_case_ids_list, repeat, fold)
            save_inner_splits(inner_splits, output_dir / "inner_splits")

            # M1: Clinical Cox PH
            print(f"  M1: Clinical Cox PH...", end=" ")
            try:
                m1 = M1ClinicalCox()
                m1.fit(X_clinical_train, y_train, e_train, clinical_prep['clinical_cols'])
                risk_test = m1.predict_risk(X_clinical_test)
                survival_test = m1.predict_survival(X_clinical_test, EVALUATION_TIMES)
                # FIXED: Pass train data for IPCW estimation
                m1_metrics = compute_all_metrics(
                    y_train, e_train, y_test, e_test, risk_test, survival_test,
                    times=EVALUATION_TIMES
                )
                metrics_by_model['M1_clinical_cox'].append(m1_metrics)
                print(f"C={m1_metrics['harrell_c']:.3f}")

                for i, case_id in enumerate(test_case_ids):
                    all_predictions.append({
                        'case_id': case_id, 'repeat': repeat, 'fold': fold,
                        'model': 'M1_clinical_cox',
                        'risk_score': float(risk_test[i]),
                        'survival_probability_12m': float(survival_test[i, 0]),
                        'survival_probability_36m': float(survival_test[i, 1]),
                        'survival_probability_60m': float(survival_test[i, 2]),
                        'survival_months': float(y_test[i]),
                        'event': int(e_test[i])
                    })
            except Exception as e:
                print(f"ERROR: {e}")

            # M2: Gene Coxnet
            print(f"  M2: Gene Coxnet...", end=" ")
            try:
                m2 = M2M3Coxnet('M2')
                best_alpha, best_l1 = m2.tune(X_gene_train, y_train, e_train)
                m2.fit(X_gene_train, y_train, e_train, gene_prep['gene_cols'])
                risk_test = m2.predict_risk(X_gene_test)
                survival_test = m2.predict_survival(X_gene_test, EVALUATION_TIMES)
                # FIXED: Pass train data for IPCW estimation
                m2_metrics = compute_all_metrics(
                    y_train, e_train, y_test, e_test, risk_test, survival_test,
                    times=EVALUATION_TIMES
                )
                metrics_by_model['M2_gene_elasticnet'].append(m2_metrics)
                print(f"C={m2_metrics['harrell_c']:.3f}")

                for i, case_id in enumerate(test_case_ids):
                    all_predictions.append({
                        'case_id': case_id, 'repeat': repeat, 'fold': fold,
                        'model': 'M2_gene_elasticnet',
                        'risk_score': float(risk_test[i]),
                        'survival_probability_12m': float(survival_test[i, 0]),
                        'survival_probability_36m': float(survival_test[i, 1]),
                        'survival_probability_60m': float(survival_test[i, 2]),
                        'survival_months': float(y_test[i]),
                        'event': int(e_test[i])
                    })
            except Exception as e:
                print(f"ERROR: {e}")

            # M3: Combined Coxnet
            print(f"  M3: Combined Coxnet...", end=" ")
            try:
                m3 = M2M3Coxnet('M3')
                best_alpha, best_l1 = m3.tune(X_combined_train, y_train, e_train)
                m3.fit(X_combined_train, y_train, e_train, combined_cols)
                risk_test = m3.predict_risk(X_combined_test)
                survival_test = m3.predict_survival(X_combined_test, EVALUATION_TIMES)
                # FIXED: Pass train data for IPCW estimation
                m3_metrics = compute_all_metrics(
                    y_train, e_train, y_test, e_test, risk_test, survival_test,
                    times=EVALUATION_TIMES
                )
                metrics_by_model['M3_combined_elasticnet'].append(m3_metrics)
                print(f"C={m3_metrics['harrell_c']:.3f}")

                for i, case_id in enumerate(test_case_ids):
                    all_predictions.append({
                        'case_id': case_id, 'repeat': repeat, 'fold': fold,
                        'model': 'M3_combined_elasticnet',
                        'risk_score': float(risk_test[i]),
                        'survival_probability_12m': float(survival_test[i, 0]),
                        'survival_probability_36m': float(survival_test[i, 1]),
                        'survival_probability_60m': float(survival_test[i, 2]),
                        'survival_months': float(y_test[i]),
                        'event': int(e_test[i])
                    })
            except Exception as e:
                print(f"ERROR: {e}")

    # Save predictions
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    pred_df = pd.DataFrame(all_predictions)
    pred_path = output_dir / "oof_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved predictions: {pred_path}")

    # Aggregate metrics
    summary = {}
    for model, metrics_list in metrics_by_model.items():
        if not metrics_list:
            continue
        metric_keys = metrics_list[0].keys()
        summary[model] = {}
        for key in metric_keys:
            try:
                values = []
                for m in metrics_list:
                    if key in m:
                        val = m[key]
                        # Handle both scalar and array metrics
                        if isinstance(val, (list, np.ndarray)):
                            val_array = np.array(val)
                            val = np.nanmean(val_array)
                        if not np.isnan(val):
                            values.append(val)

                if values:
                    summary[model][key] = {
                        'n_folds': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'per_fold': [float(v) for v in values]
                    }
            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Failed to aggregate metric {key} for {model}: {e}")
                summary[model][key] = {'error': str(e), 'per_fold': []}

    # Bootstrap comparison
    print("\nRunning bootstrap comparison...")
    comparison = {}
    model_pairs = [
        ('M2_gene_elasticnet', 'M1_clinical_cox'),
        ('M3_combined_elasticnet', 'M1_clinical_cox'),
        ('M3_combined_elasticnet', 'M2_gene_elasticnet'),
    ]
    for model_a, model_b in model_pairs:
        try:
            comp = patient_level_paired_bootstrap(
                pred_df, n_iterations=1000, seed=456,
                comparison_pair=(model_a, model_b)
            )
            comparison[f'{model_a}_vs_{model_b}'] = {
                'mean_diff': comp['mean_diff'],
                'ci_lower': comp['ci_lower'],
                'ci_upper': comp['ci_upper'],
                'p_value': comp['p_value'],
                'significant': comp['p_value'] < 0.05
            }
        except Exception as e:
            comparison[f'{model_a}_vs_{model_b}'] = {'error': str(e)}

    # Save report
    report = {
        'protocol': 'SIMPLIFIED_NESTED_CV',
        'status': 'COMPLETED',
        'training': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_hours': duration / 3600
        },
        'cv_config': {
            'n_repeats': N_OUTER_REPEATS,
            'n_outer_folds': N_OUTER_FOLDS,
            'n_inner_folds': N_INNER_FOLDS,
            'models': ['M1_clinical_cox', 'M2_gene_elasticnet', 'M3_combined_elasticnet']
        },
        'metrics': summary,
        'bootstrap_comparison': comparison
    }

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved metrics: {metrics_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<30} {'N':>5} {'Harrell C':>12} {'Uno C':>10} {'AUC 36m':>10} {'IBS':>8}")
    print("-" * 80)
    for model in ['M1_clinical_cox', 'M2_gene_elasticnet', 'M3_combined_elasticnet']:
        if model not in summary:
            print(f"{model:<30} {'--':>5} {'--':>12} {'--':>10} {'--':>10} {'--':>8}")
            continue
        m = summary[model]
        n = m.get('harrell_c', {}).get('n_folds', '--')
        hc = m.get('harrell_c', {}).get('mean', '--')
        uc = m.get('uno_c', {}).get('mean', '--')
        auc = m.get('auc_36m', {}).get('mean', '--')
        ibs = m.get('ibs', {}).get('mean', '--')
        hc_str = f"{hc:.3f}" if isinstance(hc, (int, float)) and not np.isnan(hc) else "--"
        uc_str = f"{uc:.3f}" if isinstance(uc, (int, float)) and not np.isnan(uc) else "--"
        auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) and not np.isnan(auc) else "--"
        ibs_str = f"{ibs:.3f}" if isinstance(ibs, (int, float)) and not np.isnan(ibs) else "--"
        print(f"{model:<30} {n:>5} {hc_str:>12} {uc_str:>10} {auc_str:>10} {ibs_str:>8}")

    print(f"\nBootstrap Comparisons:")
    for pair, result in comparison.items():
        if 'error' in result:
            print(f"  {pair}: ERROR - {result['error']}")
        else:
            sig = "**" if result['significant'] else ""
            print(f"  {pair}: diff={result['mean_diff']:.3f}, 95%CI=[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p={result['p_value']:.4f} {sig}")

    print(f"\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Status: COMPLETED")


if __name__ == "__main__":
    main()
