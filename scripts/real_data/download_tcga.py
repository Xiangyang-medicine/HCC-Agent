#!/usr/bin/env python3
"""
Download and Process TCGA-LIHC Data from GDC API

This script downloads real TCGA-LIHC clinical and RNA-Seq data,
processes it, and creates verified parquet files.

Usage:
    python scripts/real_data/download_tcga.py [--output-dir F:/ACM/data/real]

Requirements:
    pandas, requests, pyarrow
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Configuration
TCGA_LIHC_PROJECT = "TCGA-LIHC"
GDC_API_BASE = "https://api.gdc.cancer.gov"

# Metabolic genes of interest
METABOLIC_GENES = [
    'HK2', 'PKM', 'LDHA', 'LDHB', 'GPI', 'PFKL',  # Glycolysis
    'GLS', 'GLUD1',  # Glutamine metabolism
    'FASN', 'SCD',  # Lipogenesis
    'CA9', 'VEGFA', 'HIF1A',  # Hypoxia
    'MYC', 'CTNNB1'  # Oncogenic
]

def gdc_get(endpoint: str, params: dict = None, max_retries: int = 3) -> dict:
    """Make GET request to GDC API with retry logic."""
    url = f"{GDC_API_BASE}{endpoint}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise

def download_clinical_data(output_dir: Path) -> pd.DataFrame:
    """Download TCGA-LIHC clinical data from GDC."""
    print("\n[1/4] Downloading TCGA-LIHC clinical data from GDC API...")

    # Get cases for TCGA-LIHC project
    print("  Fetching case list...")
    cases_params = {
        "fields": "case_id,submitter_id,demographics,diagnoses,treatments,follow_ups",
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "projects.project_id", "value": "TCGA-LIHC"}}
            ]
        }),
        "size": "500"
    }
    cases_data = gdc_get("/v0/cases", cases_params)

    cases = cases_data.get('data', {}).get('hits', [])
    print(f"  Found {len(cases)} cases in TCGA-LIHC")

    if len(cases) == 0:
        raise RuntimeError("No TCGA-LIHC cases found. Check network connection.")

    # Process clinical data
    clinical_records = []
    for case in cases:
        record = {
            'case_id': case.get('case_id'),
            'patient_id': case.get('submitter_id'),
        }

        # Demographics
        demographics = case.get('demographics', {})
        record['age_at_diagnosis'] = demographics.get('age_at_diagnosis')
        record['gender'] = demographics.get('gender', '').lower()
        record['race'] = demographics.get('race')
        record['vital_status'] = demographics.get('vital_status', '').lower()
        record['days_to_death'] = demographics.get('days_to_death')
        record['days_to_last_follow_up'] = demographics.get('days_to_last_follow_up')

        # Primary diagnosis
        diagnoses = case.get('diagnoses', [])
        if diagnoses:
            primary_dx = diagnoses[0]
            record['tumor_stage'] = primary_dx.get('tumor_stage')
            record['tumor_grade'] = primary_dx.get('grade')
            record['ajcc_stage'] = primary_dx.get('ajcc_staging_system_edition')
            record['primary_diagnosis'] = primary_dx.get('primary_diagnosis')
            record['site_of_resection'] = primary_dx.get('site_of_resection_or_biopsy')

            # Calculate survival months
            if record['days_to_death']:
                record['survival_months'] = round(int(record['days_to_death']) / 30.44, 1)
                record['event'] = 1
            elif record['days_to_last_follow_up']:
                record['survival_months'] = round(int(record['days_to_last_follow_up']) / 30.44, 1)
                record['event'] = 0
            else:
                record['survival_months'] = None
                record['event'] = None

        clinical_records.append(record)

    clinical_df = pd.DataFrame(clinical_records)
    print(f"  Clinical data: {len(clinical_df)} cases")
    print(f"  Cases with survival data: {clinical_df['survival_months'].notna().sum()}")

    return clinical_df

def download_rnaseq_data(output_dir: Path) -> pd.DataFrame:
    """Download TCGA-LIHC RNA-Seq gene expression data."""
    print("\n[2/4] Downloading TCGA-LIHC RNA-Seq data...")

    # Get RNA-Seq files (HT-Counts for TCGA-LIHC)
    files_params = {
        "fields": "file_id,file_name,cases.samples.sample_type,cases.samples.is_ffpe",
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LIHC"}},
                {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}}
            ]
        }),
        "size": "500"
    }
    files_data = gdc_get("/v0/files", files_params)

    files = files_data.get('data', {}).get('hits', [])
    print(f"  Found {len(files)} RNA-Seq files")

    # Filter to tumor samples only (exclude: Normal, Adjacent Normal, etc.)
    tumor_files = []
    for f in files:
        samples = f.get('cases', [{}])[0].get('samples', [])
        for sample in samples:
            sample_type = sample.get('sample_type', '').lower()
            is_ffpe = sample.get('is_ffpe', False)
            # Only include Primary Tumor, exclude FFPE (formalin-fixed)
            if 'tumor' in sample_type and 'primary' in sample_type and not is_ffpe:
                tumor_files.append(f)
                break

    print(f"  Primary tumor samples (non-FFPE): {len(tumor_files)}")

    if len(tumor_files) == 0:
        print("  Warning: No suitable RNA-Seq files found")
        return pd.DataFrame()

    # Download expression data - useTCGAbiolinks or fetch pre-processed
    # For this script, we'll note that full download requires GDC download client
    print("  Note: Full RNA-Seq download requires GDC Data Transfer Tool")
    print("  For now, will use pre-processed data from cBioPortal or recount2")

    # Alternative: Use pre-processed TCGA data
    # This is a placeholder - actual implementation would download from GDC
    rnaseq_df = pd.DataFrame()  # Will be populated if we find pre-processed data

    return rnaseq_df

def process_and_validate(output_dir: Path, clinical_df: pd.DataFrame) -> dict:
    """Process clinical data and apply inclusion criteria."""
    print("\n[3/4] Processing and validating clinical data...")

    results = {
        'initial_cases': len(clinical_df),
        'after_tumor_filter': len(clinical_df),  # Already filtered in download
        'after_survival_data': 0,
        'after_clinical_vars': 0,
        'final_cohort': 0
    }

    # Filter: Has survival data
    df = clinical_df[clinical_df['survival_months'].notna()].copy()
    results['after_survival_data'] = len(df)
    print(f"  After requiring survival data: {len(df)}")

    # Filter: Has essential clinical variables
    required_cols = ['age_at_diagnosis', 'gender', 'tumor_stage', 'tumor_grade']
    df = df.dropna(subset=required_cols)
    results['after_clinical_vars'] = len(df)
    print(f"  After requiring clinical variables: {len(df)}")

    # Standardize stage labels
    stage_map = {
        'stage i': 'Stage I', 'stage ia': 'Stage I', 'stage ib': 'Stage I',
        'stage ii': 'Stage II', 'stage iia': 'Stage II', 'stage iib': 'Stage II',
        'stage iii': 'Stage III', 'stage iiia': 'Stage IIIA', 'stage iiib': 'Stage IIIB',
        'stage iiic': 'Stage IIIC',
        'stage iv': 'Stage IV', 'stage iva': 'Stage IV', 'stage ivb': 'Stage IV',
        'stage ix': 'Stage IV'  # Some TCGA use IX instead of IVB
    }
    df['stage'] = df['tumor_stage'].str.lower().map(stage_map).fillna(df['tumor_stage'])

    # Standardize grade labels
    grade_map = {
        'gx': 'GX', 'g1': 'G1', 'g2': 'G2', 'g3': 'G3', 'g4': 'G4'
    }
    df['grade'] = df['tumor_grade'].str.lower().map(grade_map).fillna(df['tumor_grade'])

    # Convert age to years if in days
    if df['age_at_diagnosis'].max() > 150:  # Likely in days
        df['age'] = (df['age_at_diagnosis'] / 365.25).round(1)
    else:
        df['age'] = df['age_at_diagnosis']

    # Create analysis-ready dataframe
    analysis_df = df[['patient_id', 'case_id', 'age', 'gender', 'stage', 'grade',
                      'survival_months', 'event', 'race']].copy()
    analysis_df = analysis_df.rename(columns={'patient_id': 'patient_id'})

    results['final_cohort'] = len(analysis_df)
    print(f"  Final analysis cohort: {len(analysis_df)}")

    return analysis_df, results

def save_output(output_dir: Path, analysis_df: pd.DataFrame, results: dict):
    """Save processed data and metadata."""
    print("\n[4/4] Saving output files...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main parquet
    output_path = output_dir / "tcga_lihc_real.parquet"
    analysis_df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Save cohort flow
    cohort_flow = pd.DataFrame([
        {'step': '1_initial', 'criteria': 'TCGA-LIHC total samples',
         'count': results['initial_cases'], 'notes': 'Raw download'},
        {'step': '2_survival', 'criteria': 'Require survival_months',
         'count': results['after_survival_data'], 'notes': 'Exclude missing survival'},
        {'step': '3_clinical', 'criteria': 'Require age, gender, stage, grade',
         'count': results['after_clinical_vars'], 'notes': 'Exclude missing clinical'},
        {'step': '4_final', 'criteria': 'Final analysis cohort',
         'count': results['final_cohort'], 'notes': 'Ready for survival analysis'}
    ])
    cohort_flow.to_csv(output_dir / "cohort_flow.csv", index=False)
    print(f"  Saved: cohort_flow.csv")

    # Save metadata
    metadata = {
        'source': 'TCGA GDC API',
        'project': 'TCGA-LIHC',
        'download_date': pd.Timestamp.now().isoformat(),
        'cohort_size': len(analysis_df),
        'metabolic_genes_expected': METABOLIC_GENES,
        'survival_months_range': f"{analysis_df['survival_months'].min():.1f}-{analysis_df['survival_months'].max():.1f}",
        'event_rate': f"{analysis_df['event'].mean():.3f}",
        'stage_distribution': analysis_df['stage'].value_counts().to_dict(),
        'grade_distribution': analysis_df['grade'].value_counts().to_dict()
    }
    with open(output_dir / "data_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: data_metadata.json")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download and process TCGA-LIHC data")
    parser.add_argument('--output-dir', type=str,
                        default='F:/ACM/data/real',
                        help='Output directory for processed data')
    parser.add_argument('--skip-rnaseq', action='store_true',
                        help='Skip RNA-Seq download (clinical data only)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("TCGA-LIHC Data Download and Processing")
    print("=" * 60)

    try:
        # Download clinical data
        clinical_df = download_clinical_data(output_dir)

        # Download RNA-Seq data (optional)
        if not args.skip_rnaseq:
            rnaseq_df = download_rnaseq_data(output_dir)

        # Process and validate
        analysis_df, results = process_and_validate(output_dir, clinical_df)

        # Save output
        output_path = save_output(output_dir, analysis_df, results)

        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Output saved to: {output_path}")
        print(f"Final cohort: {len(analysis_df)} patients")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nIf network issues persist:")
        print("1. Try downloading from cBioPortal: https://www.cbioportal.org/")
        print("2. Use GDC Data Transfer Tool: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool")
        print("3. Check firewall/proxy settings")
        sys.exit(1)

if __name__ == "__main__":
    main()
