#!/usr/bin/env python3
"""Fix os_derivation_audit.csv matching logic."""
import pandas as pd
import numpy as np

# Input files
AUDIT_PATH = 'data/processed/gdc/20260713/os_derivation_audit.csv'
PATIENT_PATH = 'data/processed/gdc/20260713/tcga_lihc_patients.parquet'
OUTPUT_PATH = 'data/processed/gdc/20260713/os_derivation_audit.csv'

# Read data
audit_df = pd.read_csv(AUDIT_PATH)
patient_df = pd.read_parquet(PATIENT_PATH)

print("=" * 70)
print("Fixing OS Derivation Audit: Comparing by case_id with tolerance")
print("=" * 70)
print(f"\nAudit rows: {len(audit_df)}")
print(f"Patient rows: {len(patient_df)}")

# Check OS matching logic
dtype_issues = []
value_differences = []

results = []
all_matched = True

for idx, row in audit_df.iterrows():
    case_id = row['case_id']

    # Skip if patient not found
    if case_id not in patient_df['case_id'].values:
        print(f" ERROR: Case {case_id} not found in patient dataset!")
        results.append({
            'case_id': case_id,
            'os_value_match': False,
            'event_match': False,
            'matches_parquet': False,
            'notes': 'Patient record missing'
        })
        all_matched = False
        continue

    patient_row = patient_df[patient_df['case_id'] == case_id].iloc[0]

    # Get OS values with type handling
    audit_os = row['final_os_months']
    patient_os = patient_row['survival_months']
    audit_event = int(row['event']) if pd.notna(row['event']) else None
    patient_event = int(patient_row['event']) if pd.notna(patient_row['event']) else None

    # Tolerance comparison for OS months (absolute error <= 0.01)
    os_tolerance = 0.01
    os_match = abs(float(audit_os) - float(patient_os)) <= os_tolerance

    # Event must be exactly equal (as integers)
    event_match = audit_event == patient_event

    # Store results
    result = {
        'case_id': case_id,
        'os_value_match': os_match,
        'event_match': event_match,
        'matches_parquet_current': bool(row['matches_parquet']),
        'os_audit': float(audit_os) if pd.notna(audit_os) else None,
        'os_patient': float(patient_os) if pd.notna(patient_os) else None,
        'os_diff': abs(float(audit_os) - float(patient_os)) if pd.notna(audit_os) and pd.notna(patient_os) else None,
        'event_audit': audit_event,
        'event_patient': patient_event,
        'matches_parquet': os_match and event_match
    }

    results.append(result)

    if not (os_match and event_match):
        all_matched = False
        value_differences.append(result)
        if abs(audit_os - patient_os) > 0.01:
            dtype_issues.append(result)

print(f"\nAudit vs Patient comparison:")
print(f" Total cases checked: {len(results)}")
print(f" OS matches within tolerance: {sum(1 for r in results if r['os_value_match'])}")
print(f" Event matches exactly: {sum(1 for r in results if r['event_match'])}")
print(f" Combined matches: {sum(1 for r in results if r['matches_parquet'])}")
print(f" Issues found: {len(value_differences)} cases")
if dtype_issues:
    print(f" Potential dtype issues: {len(dtype_issues)} cases")
    print("\nFirst dtype issue example:")
    for r in dtype_issues[:1]:
        print(f"  Case {r['case_id']}: diff={r['os_diff']}, tolerance=0.01")

# Create updated audit dataframe with new match column
results_df = pd.DataFrame(results)

# Update only the matches_parquet column in the original audit
for idx, row in results_df.iterrows():
    audit_id = results_df.iloc[idx]['case_id']
    new_match = row['matches_parquet']

print(f"\nUpdating matches_parquet for {len(results_df)} rows...")
for idx, (orig_idx, orig_row) in enumerate(results_df.iterrows()):
    audit_case_id = orig_row['case_id']
    audit_df.at[orig_idx, 'matches_parquet'] = orig_row['matches_parquet']
    # Update diagnostic columns
    audit_df.at[orig_idx, 'os_value_match'] = orig_row['os_value_match']
    audit_df.at[orig_idx, 'event_match'] = orig_row['event_match']

print(f"\nAll 363/363 should now TRUE: {all_matched}")

# Save updated file
audit_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nUpdated file saved to: {OUTPUT_PATH}")

# Show summary
print("\n" + "=" * 70)
print("Updated matches_parquet summary:")
print("=" * 70)
print(audit_df['matches_parquet'].value_counts(dropna=False))
