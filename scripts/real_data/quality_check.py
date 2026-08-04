#!/usr/bin/env python3
"""
Data Quality Check for TCGA-LIHC Dataset

Performs comprehensive data quality checks and generates a report.
This script verifies data integrity before using it in the Prognostic Engine.

Usage:
    python scripts/real_data/quality_check.py --input data/real/tcga_lihc_real.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Expected metabolic genes
METABOLIC_GENES = [
    'HK2', 'PKM', 'LDHA', 'LDHB', 'GPI', 'PFKL',  # Glycolysis
    'GLS', 'GLUD1',  # Glutamine metabolism
    'FASN', 'SCD',  # Lipogenesis
    'CA9', 'VEGFA', 'HIF1A',  # Hypoxia
    'MYC', 'CTNNB1'  # Oncogenic
]

def check_missing_data(df: pd.DataFrame) -> dict:
    """Check missing data patterns."""
    results = {}

    # Overall missingness
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    results['missing_by_column'] = missing_pct.to_dict()

    # Columns with > 20% missing
    results['high_missing_cols'] = [col for col, pct in missing_pct.items() if pct > 20]

    # Rows with any missing
    rows_with_missing = df.isnull().any(axis=1).sum()
    results['rows_with_missing'] = rows_with_missing
    results['rows_with_missing_pct'] = round(rows_with_missing / len(df) * 100, 2)

    return results

def check_survival_data(df: pd.DataFrame) -> dict:
    """Check survival data validity."""
    results = {}

    # Check survival_months
    results['survival_months_min'] = df['survival_months'].min()
    results['survival_months_max'] = df['survival_months'].max()
    results['survival_months_median'] = df['survival_months'].median()

    # Check for impossible values
    results['negative_survival'] = (df['survival_months'] < 0).sum()
    results['zero_survival_with_no_event'] = ((df['survival_months'] == 0) & (df['event'] == 0)).sum()

    # Check event status
    results['event_rate'] = df['event'].mean()
    results['n_events'] = df['event'].sum()
    results['n_censored'] = (1 - df['event']).sum()

    # Event rate sanity check (should be reasonable for HCC)
    if results['event_rate'] < 0.1 or results['event_rate'] > 0.9:
        results['warnings'] = results.get('warnings', [])
        results['warnings'].append(f"Unusual event rate: {results['event_rate']:.2%}")

    return results

def check_clinical_variables(df: pd.DataFrame) -> dict:
    """Check clinical variable distributions."""
    results = {}

    # Age
    if 'age' in df.columns:
        results['age_mean'] = df['age'].mean()
        results['age_range'] = (df['age'].min(), df['age'].max())
        if df['age'].max() > 100 or df['age'].min() < 18:
            results['warnings'] = results.get('warnings', [])
            results['warnings'].append(f"Suspicious age range: {df['age'].min()}-{df['age'].max()}")

    # Gender
    if 'gender' in df.columns:
        results['gender_distribution'] = df['gender'].value_counts().to_dict()

    # Stage
    if 'stage' in df.columns:
        results['stage_distribution'] = df['stage'].value_counts().to_dict()

    # Grade
    if 'grade' in df.columns:
        results['grade_distribution'] = df['grade'].value_counts().to_dict()

    return results

def check_gene_expression(df: pd.DataFrame) -> dict:
    """Check gene expression data quality."""
    results = {}

    available_genes = [g for g in METABOLIC_GENES if g in df.columns]
    results['available_genes'] = available_genes
    results['missing_genes'] = [g for g in METABOLIC_GENES if g not in df.columns]
    results['genes_available_pct'] = round(len(available_genes) / len(METABOLIC_GENES) * 100, 1)

    # Check expression distributions
    for gene in available_genes:
        expr = df[gene].dropna()
        if len(expr) > 0:
            z_scores = np.abs((expr - expr.mean()) / expr.std())
            outliers = (z_scores > 5).sum()
            if outliers > 0:
                results.setdefault('gene_outliers', {})[gene] = outliers

    # Check for unrealistic values (negative expression, extremely high)
    for gene in available_genes:
        expr = df[gene].dropna()
        if (expr < 0).any():
            results.setdefault('negative_values', []).append(gene)
        if (expr > 1e6).any():
            results.setdefault('extremely_high_values', []).append(gene)

    return results

def check_data_integrity(df: pd.DataFrame) -> dict:
    """Check overall data integrity."""
    results = {}

    # Check for duplicate patient IDs
    duplicates = df['patient_id'].duplicated().sum()
    results['duplicate_patients'] = duplicates

    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    results['duplicate_columns'] = duplicate_cols

    # Check data types
    results['data_types'] = df.dtypes.astype(str).to_dict()

    return results

def generate_report(df: pd.DataFrame, output_path: Path):
    """Generate comprehensive quality report."""
    report = {
        'dataset_info': {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist()
        },
        'missing_data': check_missing_data(df),
        'survival_data': check_survival_data(df),
        'clinical_variables': check_clinical_variables(df),
        'gene_expression': check_gene_expression(df),
        'data_integrity': check_data_integrity(df)
    }

    # Determine overall quality
    issues = []

    if report['missing_data']['rows_with_missing_pct'] > 50:
        issues.append("High proportion of rows with missing data")

    if report['survival_data'].get('negative_survival', 0) > 0:
        issues.append("Negative survival values found")

    if report['data_integrity']['duplicate_patients'] > 0:
        issues.append("Duplicate patient IDs found")

    if len(report['gene_expression'].get('missing_genes', [])) > len(METABOLIC_GENES) / 2:
        issues.append("Most metabolic genes missing")

    report['quality_issues'] = issues
    report['quality_status'] = 'PASS' if len(issues) == 0 else 'FAIL'

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report

def main():
    parser = argparse.ArgumentParser(description="Quality check for TCGA-LIHC data")
    parser.add_argument('--input', type=str,
                        default='F:/ACM/data/real/tcga_lihc_real.parquet',
                        help='Input parquet file')
    parser.add_argument('--output', type=str,
                        default='F:/ACM/data/real/quality_report.json',
                        help='Output report path')
    args = parser.parse_args()

    print("=" * 60)
    print("TCGA-LIHC Data Quality Check")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("\nTo download fresh data, run:")
        print("  python scripts/real_data/download_tcga.py")
        return

    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Generate report
    output_path = Path(args.output)
    report = generate_report(df, output_path)

    # Print summary
    print("\n" + "-" * 60)
    print("QUALITY CHECK SUMMARY")
    print("-" * 60)
    print(f"Status: {report['quality_status']}")
    print(f"Rows: {report['dataset_info']['n_rows']}")
    print(f"Columns: {report['dataset_info']['n_columns']}")

    if 'available_genes' in report['gene_expression']:
        print(f"Metabolic genes available: {report['gene_expression']['genes_available_pct']}%")
        print(f"  Available: {report['gene_expression']['available_genes']}")
        print(f"  Missing: {report['gene_expression'].get('missing_genes', [])}")

    print(f"\nSurvival data:")
    print(f"  Event rate: {report['survival_data']['event_rate']:.1%}")
    print(f"  Range: {report['survival_data']['survival_months_min']:.1f} - "
          f"{report['survival_data']['survival_months_max']:.1f} months")

    if report['quality_issues']:
        print(f"\n⚠️  Issues found:")
        for issue in report['quality_issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✓  No critical issues found")

    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
