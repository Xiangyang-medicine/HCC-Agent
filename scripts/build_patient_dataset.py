#!/usr/bin/env python3
"""
Build clinical + OS dataset from TCGA-LIHC data.
Uses batch GDC API query for efficient file-to-case mapping.
"""
import os
import json
import hashlib
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
RAW_DIR = Path("data/raw/gdc/20260713")
EXPR_DIR = RAW_DIR / "raw_expression"
OUTPUT_DIR = Path("data/processed/gdc/20260713")
MANIFEST_PATH = RAW_DIR / "gdc_manifest_primary_tumor.tsv"
CASES_PATH = RAW_DIR / "cases_complete_response.json"

# 15 Metabolic Genes
METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",  # Glycolysis
    "GLS", "GLUD1",  # Glutamine
    "FASN", "SCD",  # Lipogenesis
    "CA9", "VEGFA", "HIF1A",  # Hypoxia
    "MYC", "CTNNB1"  # Oncogenic
]

def load_manifest():
    """Load manifest file."""
    downloads = []
    with open(MANIFEST_PATH, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                filename = parts[1]
                file_uuid_in_name = filename.split('.')[0]
                downloads.append({
                    'file_id': parts[0],
                    'filename': filename,
                    'file_uuid_in_name': file_uuid_in_name,
                    'md5': parts[2],
                    'size': parts[3]
                })
    return downloads

def load_cases():
    """Load cases data."""
    with open(CASES_PATH, 'r') as f:
        return json.load(f)

def query_single_file(file_id, headers):
    """Query a single file for its case mapping."""
    url = "https://api.gdc.cancer.gov/files"
    filter_query = {
        "op": "=",
        "content": {
            "field": "file_id",
            "value": file_id
        }
    }

    params = {
        "fields": "cases.case_id,cases.submitter_id,cases.samples.sample_id,cases.samples.sample_type,cases.samples.is_ffpe,cases.project.project_id",
        "format": "JSON",
        "size": 1,
        "filters": json.dumps(filter_query)
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get('data', {}).get('hits', [])
            if hits:
                hit = hits[0]
                cases = hit.get('cases', [])
                if cases:
                    case = cases[0]
                    project_id = case.get('project', {}).get('project_id')
                    samples = case.get('samples', [])
                    primary_samples = [s for s in samples if s.get('sample_type') == 'Primary Tumor']
                    if primary_samples and project_id == 'TCGA-LIHC':
                        sample_info = primary_samples[0]
                        return {
                            'file_id': file_id,
                            'case_id': case.get('case_id'),
                            'submitter_id': case.get('submitter_id'),
                            'sample_id': sample_info.get('sample_id'),
                            'sample_type': sample_info.get('sample_type'),
                            'is_ffpe': sample_info.get('is_ffpe'),
                            'project_id': project_id
                        }
    except Exception:
        pass
    return None


def get_file_to_case_mapping(file_ids):
    """Get file-to-case mappings using concurrent GDC GET API queries."""
    print("Querying GDC API for file-to-case mappings...")

    headers = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'}
    mappings = {}
    total = len(file_ids)
    completed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_single_file, fid, headers): fid for fid in file_ids}

        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == 1:
                print(f"  Progress: {completed}/{total}...")

            result = future.result()
            if result:
                mappings[result['file_id']] = result

    print(f"  Retrieved {len(mappings)}/{total} valid TCGA-LIHC Primary Tumor mappings")
    return mappings

def main():
    print("=" * 60)
    print("Building Clinical + OS Dataset")
    print("=" * 60)

    # Load data
    manifest = load_manifest()
    cases_data = load_cases()

    print(f"\nManifest files: {len(manifest)}")
    print(f"Cases loaded: {len(cases_data.get('data', {}).get('hits', []))}")

    # Build cases lookup
    cases = cases_data.get('data', {}).get('hits', [])
    cases_by_id = {c['id']: c for c in cases}

    # Get file_id list from manifest
    file_ids = [m['file_id'] for m in manifest]
    print(f"File IDs to query: {len(file_ids)}")

    # Query GDC for sample info
    sample_mappings = get_file_to_case_mapping(file_ids)

    # Save mappings for debugging
    with open(OUTPUT_DIR / "file_case_mappings.json", 'w') as f:
        json.dump(sample_mappings, f, indent=2)

    # Build expression data
    print("\nLoading expression data...")
    expression_df = pd.read_parquet(OUTPUT_DIR / "tcga_lihc_expression_counts.parquet")
    print(f"Expression matrix: {expression_df.shape}")

    # Create mapping from file_id to manifest entry
    manifest_by_file_id = {m['file_id']: m for m in manifest}

    # Build patient dataset
    print("\nBuilding patient dataset...")

    patient_records = []
    for file_id, sample_info in sample_mappings.items():
        if not sample_info:
            continue

        manifest_entry = manifest_by_file_id.get(file_id)
        if not manifest_entry:
            continue

        case_id = sample_info.get('case_id')
        if not case_id or case_id not in cases_by_id:
            continue

        case = cases_by_id[case_id]

        # Get survival data
        vital_status = case.get('demographic', {}).get('vital_status', 'Unknown')
        days_to_death = case.get('demographic', {}).get('days_to_death')
        age_at_diagnosis = case.get('demographic', {}).get('age_at_index')

        # For Alive patients, check diagnoses for days_to_last_follow_up
        days_to_followup = None
        if vital_status == 'Alive':
            diagnoses = case.get('diagnoses', [])
            for diag in diagnoses:
                dtf = diag.get('days_to_last_follow_up')
                if dtf is not None:
                    days_to_followup = dtf
                    break

        # Calculate survival months
        if vital_status == 'Dead' and days_to_death:
            survival_months = days_to_death / 30.4375
            event = 1
        elif vital_status == 'Alive' and days_to_followup:
            survival_months = days_to_followup / 30.4375
            event = 0
        else:
            continue  # Skip if no survival data

        # Get clinical data
        diagnoses = case.get('diagnoses', [])
        ajcc_stage = None
        grade = None
        for diag in diagnoses:
            if ajcc_stage is None:
                ajcc_stage = diag.get('ajcc_pathologic_stage')
            if grade is None:
                grade = diag.get('tumor_grade')

        # Get expression data
        # Note: expression_df.index uses manifest 'id' (file_id), not filename UUID
        if file_id not in expression_df.index:
            continue
        expr_row = expression_df.loc[file_id].to_dict()

        # Build record
        record = {
            'case_id': case_id,
            'sample_id': sample_info.get('sample_id', ''),
            'file_id': file_id,
            'submitter_id': case.get('submitter_id', ''),
            'vital_status': vital_status,
            'survival_months': round(survival_months, 2),
            'event': event,
            'age_at_diagnosis': age_at_diagnosis,
            'ajcc_stage': ajcc_stage,
            'tumor_grade': grade,
            'sample_type': sample_info.get('sample_type', ''),
            'is_ffpe': sample_info.get('is_ffpe', ''),
        }

        # Add expression data
        for gene in METABOLIC_GENES:
            record[f'{gene}_counts'] = expr_row.get(gene, np.nan)

        patient_records.append(record)

    # Create DataFrame
    patient_df = pd.DataFrame(patient_records)
    print(f"\nPatient records: {len(patient_df)}")

    if len(patient_df) == 0:
        print("ERROR: No patient records created!")
        return None

    alive_count = (patient_df['event'] == 0).sum()
    dead_count = (patient_df['event'] == 1).sum()
    print(f"Alive: {alive_count}")
    print(f"Dead: {dead_count}")

    # Save
    patient_path = OUTPUT_DIR / "tcga_lihc_patients.parquet"
    patient_df.to_parquet(patient_path)
    print(f"\nPatient data saved to: {patient_path}")

    # Save JSON version
    patient_json_path = OUTPUT_DIR / "tcga_lihc_patients.json"
    patient_df.to_json(patient_json_path, orient='records', indent=2)
    print(f"Patient data saved to: {patient_json_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total patients: {len(patient_df)}")
    print(f"Features: {len(patient_df.columns)}")
    print(f"\nSurvival statistics:")
    print(f"  Median survival (months): {patient_df['survival_months'].median():.1f}")
    print(f"  Mean survival (months): {patient_df['survival_months'].mean():.1f}")
    print(f"  Alive: {alive_count} ({alive_count/len(patient_df)*100:.1f}%)")
    print(f"  Dead: {dead_count} ({dead_count/len(patient_df)*100:.1f}%)")

    # Expression summary
    print(f"\nExpression summary (counts):")
    for gene in METABOLIC_GENES:
        col = f'{gene}_counts'
        if col in patient_df.columns:
            print(f"  {gene}: mean={patient_df[col].mean():.1f}, median={patient_df[col].median():.1f}")

    return patient_df

if __name__ == "__main__":
    main()
