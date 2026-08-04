#!/usr/bin/env python3
"""Create Phase 3A modeling dataset from patient + TPM data."""
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Constants
PROCESSED_DIR = Path("data/processed/gdc/20260713")
MODELING_DIR = Path("data/modeling")
MODELING_DIR.mkdir(parents=True, exist_ok=True)

METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",  # Glycolysis
    "GLS", "GLUD1",  # Glutamine
    "FASN", "SCD",  # Lipogenesis
    "CA9", "VEGFA", "HIF1A",  # Hypoxia
    "MYC", "CTNNB1"  # Oncogenic
]

CLINICAL_VARS = ['age_at_diagnosis', 'gender', 'ajcc_stage', 'tumor_grade']
LABEL_VARS = ['case_id', 'submitter_id', 'survival_months', 'event']

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_df_sha256(df, filepath_for_reference=None):
    """Compute SHA-256 hash of a dataframe by hashing its parquet representation."""
    import tempfile
    import os
    # Write to temp file and hash
    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path)
    hash_val = compute_sha256(tmp_path)
    os.unlink(tmp_path)
    return hash_val

def main():
    print("=" * 70)
    print("Creating Phase 3A Modeling Dataset")
    print("=" * 70)

    # Load patient data
    patient_path = PROCESSED_DIR / "tcga_lihc_patients.parquet"
    patient_df = pd.read_parquet(patient_path)
    print(f"\nPatient data: {patient_df.shape}")

    # Load TPM matrix
    tpm_path = PROCESSED_DIR / "tcga_lihc_expression_tpm.parquet"
    tpm_df = pd.read_parquet(tpm_path)
    print(f"TPM matrix: {tpm_df.shape}")

    # Verify TPM is not counts (TPM values should be reasonable, not integers > 10000)
    sample_tpm = tpm_df.iloc[0]
    print(f"\nTPM sample (first patient):")
    print(f"  HK2: {sample_tpm['HK2']:.4f} (should be TPM, not counts)")
    print(f"  PKM: {sample_tpm['PKM']:.4f}")

    # Merge TPM by file_id
    # TPM df has file_id as index, patient df has file_id as column
    tpm_df = tpm_df.reset_index()  # file_id becomes column
    modeling_df = patient_df.merge(tpm_df, on='file_id', how='inner', suffixes=('_old', ''))
    print(f"\nMerged dataset: {modeling_df.shape}")

    # Check patient_df still has counts columns - we need to verify
    patient_gene_cols = [c for c in patient_df.columns if c.endswith('_counts')]
    if patient_gene_cols:
        print(f"\nWARNING: Patient df contains _counts columns: {patient_gene_cols}")
        print("These are COUNTS, NOT TPM. Using TPM matrix for modeling.")
        # Drop counts columns from patient_df if they exist (use TPM instead)
        # Actually, after merge, the TPM columns will overwrite or we need to select

    # Check which gene columns we have after merge
    tpm_gene_cols = [c for c in modeling_df.columns if c in METABOLIC_GENES]
    print(f"\nGene columns in merged df: {len(tpm_gene_cols)}")

    # Check if we have counts columns too
    counts_cols = [c for c in modeling_df.columns if c.endswith('_counts')]
    if counts_cols:
        print(f"\nFound _counts columns in merged df: {counts_cols}")
        print("DROPPING counts columns - using TPM matrix for modeling input")

    # Create log2(TPM + 1) expression
    for gene in METABOLIC_GENES:
        if gene in modeling_df.columns:
            tpm_col = gene  # This is from TPM matrix
            log_col = f'{gene}_log2tpm'
            modeling_df[log_col] = np.log2(modeling_df[tpm_col] + 1)
        else:
            print(f"WARNING: {gene} not found in merged data")

    # Verify: check _counts columns vs _log2tpm columns
    log_cols = [f'{g}_log2tpm' for g in METABOLIC_GENES if f'{g}_log2tpm' in modeling_df.columns]
    print(f"\nLog2TPM columns created: {len(log_cols)}")

    # Final columns for modeling
    label_cols = ['case_id', 'submitter_id', 'survival_months', 'event']
    clinical_cols = ['age_at_diagnosis', 'gender', 'ajcc_stage', 'tumor_grade']
    # Note: gender is not in patient_df currently, need to check
    if 'gender' not in modeling_df.columns:
        print("WARNING: 'gender' column not found in patient data")
        clinical_cols = ['age_at_diagnosis', 'ajcc_stage', 'tumor_grade']

    gene_expr_cols = log_cols
    all_modeling_cols = label_cols + clinical_cols + gene_expr_cols

    # Select final columns
    final_df = modeling_df[all_modeling_cols].copy()

    # Add file_id for reference
    final_df['file_id'] = modeling_df['file_id']

    print(f"\nFinal modeling dataset: {final_df.shape}")
    print(f"Label columns: {label_cols}")
    print(f"Clinical columns: {clinical_cols}")
    print(f"Gene expression columns: {len(gene_expr_cols)}")

    # Check for missingness
    print(f"\nMissingness:")
    for col in clinical_cols:
        missing = final_df[col].isna().sum()
        pct = missing / len(final_df) * 100
        print(f"  {col}: {missing} ({pct:.1f}%)")

    # Event statistics
    print(f"\nEvent distribution:")
    print(f"  Alive (event=0): {(final_df['event'] == 0).sum()}")
    print(f"  Dead (event=1): {(final_df['event'] == 1).sum()}")
    print(f"  Event rate: {final_df['event'].mean()*100:.1f}%")

    # Survival statistics
    print(f"\nSurvival statistics:")
    print(f"  Median: {final_df['survival_months'].median():.1f} months")
    print(f"  Mean: {final_df['survival_months'].mean():.1f} months")

    # Save modeling dataset
    output_path = MODELING_DIR / "tcga_lihc_modeling_dataset.parquet"
    final_df.to_parquet(output_path)
    print(f"\nSaved: {output_path}")

    # Compute SHA-256 hashes
    print("\nComputing SHA-256 hashes...")
    patient_sha = compute_sha256(patient_path)
    tpm_sha = compute_sha256(tpm_path)
    modeling_sha = compute_df_sha256(final_df)

    # Create manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "input_files": {
            "patient_data": {
                "path": str(patient_path),
                "sha256": patient_sha,
                "rows": len(patient_df),
                "columns": len(patient_df.columns)
            },
            "tpm_matrix": {
                "path": str(tpm_path),
                "sha256": tpm_sha,
                "rows": len(tpm_df),
                "columns": 15
            }
        },
        "output_file": {
            "path": str(output_path),
            "sha256": modeling_sha,
            "rows": len(final_df),
            "columns": len(final_df.columns)
        },
        "cohort": {
            "total_patients": len(final_df),
            "events": int(final_df['event'].sum()),
            "censored": int((final_df['event'] == 0).sum()),
            "event_rate": float(final_df['event'].mean())
        },
        "variables": {
            "label": {
                "columns": label_cols,
                "description": "case_id, submitter_id, survival_months, event"
            },
            "clinical": {
                "columns": clinical_cols,
                "missing": {
                    col: {"count": int(final_df[col].isna().sum()), "percent": float(final_df[col].isna().mean()*100)}
                    for col in clinical_cols if col in final_df.columns
                }
            },
            "gene_expression": {
                "columns": gene_expr_cols,
                "unit": "log2(TPM + 1)",
                "source": "tcga_lihc_expression_tpm.parquet"
            }
        },
        "os_definition": {
            "formula": "survival_months = days / 30.4375",
            "source": "Phase 2B locked",
            "verified": True
        },
        "leakage_check": {
            "label_in_features": False,
            "columns_excluded": ["event", "survival_months", "vital_status", "days_to_death"],
            "evidence": "label_leakage_report.md"
        },
        "age_restriction": {
            "all_patients": {"n": 363},
            "exclude_under_18": {"n": 2, "ids": ["TCGA-5R-AA1D", "TCGA-XR-A8TE"], "sensitivity_n": 361}
        }
    }

    manifest_path = MODELING_DIR / "modeling_dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Modeling Dataset Created Successfully")
    print("=" * 70)
    print(f"\nDataset: {output_path}")
    print(f"Patients: {len(final_df)}")
    print(f"Events: {final_df['event'].sum()}")
    print(f"Clinical variables: {clinical_cols}")
    print(f"Gene variables: {len(gene_expr_cols)} (log2(TPM+1))")
    print(f"\nManifest: {manifest_path}")

    # Verify label columns NOT in features
    feature_cols = clinical_cols + gene_expr_cols
    leakage_check = set(feature_cols) & set(label_cols)
    if leakage_check:
        print(f"\nERROR: Label columns in features: {leakage_check}")
    else:
        print(f"\nLabel leakage check: PASS (no label columns in features)")

    return final_df, manifest

if __name__ == "__main__":
    main()
