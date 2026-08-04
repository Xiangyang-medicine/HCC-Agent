#!/usr/bin/env python3
"""Sensitivity analyses for Phase 3A.

Per SAP Section 9:
- SA1: All patients (363) - PRIMARY (already completed)
- SA2: Exclude age < 18 (361 patients: TCGA-5R-AA1D, TCGA-XR-A8TE)
- SA3: Complete clinical cases (338 patients: exclude missing stage/grade)

This script runs SA2 and SA3 using the formal training framework.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR / "src"
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def create_sa_dataset(sa_name, exclude_patients=None, require_complete_clinical=False):
    """Create filtered dataset for sensitivity analysis."""
    import pandas as pd

    # Load original data
    source_path = PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    df = pd.read_parquet(source_path)

    # Apply filters
    if exclude_patients:
        df = df[~df['case_id'].isin(exclude_patients)]

    if require_complete_clinical:
        df = df.dropna(subset=['ajcc_stage', 'tumor_grade'])

    # Save filtered dataset
    output_dir = PROJECT_ROOT / "data" / "sensitivity" / sa_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tcga_lihc_modeling_dataset.parquet"
    df.to_parquet(output_path)

    return {
        'sa_name': sa_name,
        'n_patients': len(df),
        'n_events': int(df['event'].sum()),
        'output_path': str(output_path)
    }


def run_sa_training(sa_name, data_path, config_overrides=None):
    """Run formal training for a sensitivity analysis."""
    output_dir = PROJECT_ROOT / "experiments" / f"phase3a_sa_{sa_name.lower()}"

    # Create config overrides
    config = {
        'run_id': f'sa_{sa_name.lower()}',
        'data_path': str(data_path),
        'output_dir': str(output_dir),
    }
    if config_overrides:
        config.update(config_overrides)

    # Save config
    config_path = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run training (will use formal training framework)
    # Note: This requires the training infrastructure to support custom data paths
    print(f"SA {sa_name}: Would run training with {config}")

    return output_dir


def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSES")
    print("=" * 70)
    print()

    # SA definitions
    sa_configs = {
        'SA2': {
            'description': 'Exclude age < 18 (pediatric cases)',
            'exclude_patients': [
                'd3d3dba9-139e-4f57-ac70-b741553d1687',  # 17y
                'ec4b4d34-2576-4412-b036-7460e64f4398'   # 16y
            ],
            'require_complete_clinical': False,
            'expected_n': 361
        },
        'SA3': {
            'description': 'Complete clinical cases only',
            'exclude_patients': None,
            'require_complete_clinical': True,
            'expected_n': 338
        }
    }

    results = {}

    for sa_name, config in sa_configs.items():
        print(f"Processing {sa_name}: {config['description']}")
        result = create_sa_dataset(
            sa_name,
            exclude_patients=config.get('exclude_patients'),
            require_complete_clinical=config.get('require_complete_clinical')
        )
        results[sa_name] = result
        print(f"  Created: {result['output_path']}")
        print(f"  N = {result['n_patients']} (expected: {config['expected_n']})")
        print(f"  Events = {result['n_events']}")
        print()

    print("=" * 70)
    print("SENSITIVITY ANALYSIS DATA PREPARATION COMPLETE")
    print("=" * 70)
    print()
    print("To run full sensitivity analysis training:")
    print("  python -m prognostic_engine.scripts.run_formal_training \\")
    print("    --data-path data/sensitivity/SA2/tcga_lihc_modeling_dataset.parquet \\")
    print("    --output-dir experiments/phase3a_sa_sa2")
    print()
    print("  python -m prognostic_engine.scripts.run_formal_training \\")
    print("    --data-path data/sensitivity/SA3/tcga_lihc_modeling_dataset.parquet \\")
    print("    --output-dir experiments/phase3a_sa_sa3")
    print()
    print("Note: Full SA training requires significant computation (~2-4 hours per SA).")
    print("      Consider running in parallel on multiple machines if available.")
    print()

    # Save summary
    output_json = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "sensitivity_analysis_config.json"
    with open(output_json, 'w') as f:
        json.dump({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'sa_configs': results,
            'note': 'Full SA training requires separate execution'
        }, f, indent=2)
    print(f"Saved: {output_json}")

    return results


if __name__ == "__main__":
    main()
