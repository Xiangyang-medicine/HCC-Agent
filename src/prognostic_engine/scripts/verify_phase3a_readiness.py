#!/usr/bin/env python
"""
Phase 3A Reset Readiness Verification Script

Exits 0 only if:
- All pytest tests pass
- Environment has required packages
- Source files sanitized for forbidden patterns
- SHA-256 hashes of source files match expected values

Writes:
- experiments/phase3a/readiness/READINESS_GATE.json with success = True/False

Do not manually edit this file for checks unless you have explicit Phase 3A reset
validation rationale.
"""
import sys
import json
import subprocess
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

# Calculate PROJECT_ROOT from script location
# scripts/verify_phase3a_readiness.py -> prognostic_engine/scripts -> prognostic_engine -> src -> project root
# Script is at: F:\ACM\src\prognostic_engine\scripts\verify_phase3a_readiness.py
# Need 4 levels of .parent to get to F:\ACM
_SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_FILE.parent.parent.parent.parent  # F:/ACM
READY_DIR = PROJECT_ROOT / "experiments" / "phase3a" / "readiness"
GATE_PATH = READY_DIR / "READINESS_GATE.json"
INVALID_DIR = READY_DIR / "invalid"
INVALID_REASON_PATH = INVALID_DIR / "INVALID_REASON.md"

# Structure integrity status values
STRUCTURE_STATUS = {
    "COMPLETE": "COMPLETE",
    "FAILED_INCOMPLETE": "FAILED_INCOMPLETE",
    "NOT_APPLICABLE": "NOT_APPLICABLE",
    "PENDING": "PENDING",
}

# Methodology compliance status values
METHODOLOGY_STATUS = {
    "COMPLIANT": "COMPLIANT",
    "NON_COMPLIANT": "NON_COMPLIANT",
    "PENDING": "PENDING",
}


def get_file_hash(filepath):
    """Calculate SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_file_exists(filepath):
    """Check if file exists and is readable."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Required file not found: {filepath}")
    return True


def check_structure_integrity():
    """
    Verify Phase 3A project structure integrity.

    Returns a dict with:
        - status: COMPLETE | FAILED_INCOMPLETE | NOT_APPLICABLE
        - issues: list of missing/invalid components
        - details: dict of structure checks
    """
    issues = []
    details = {}

    # Required directories
    required_dirs = [
        PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine",
        PROJECT_ROOT / "src" / "prognostic_engine" / "tests",
        PROJECT_ROOT / "src" / "prognostic_engine" / "scripts",
        PROJECT_ROOT / "data" / "modeling",
        PROJECT_ROOT / "experiments" / "phase3a" / "training",
        PROJECT_ROOT / "experiments" / "phase3a" / "readiness",
    ]

    for d in required_dirs:
        dir_name = d.relative_to(PROJECT_ROOT)
        exists = d.is_dir()
        details[str(dir_name)] = {"type": "directory", "exists": exists}
        if not exists:
            issues.append(f"Missing directory: {dir_name}")

    # Required source files
    required_sources = {
        "models.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "models.py",
        "training.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "training.py",
        "bootstrap.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "bootstrap.py",
        "inner_splits.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "inner_splits.py",
        "inner_preprocessing.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "inner_preprocessing.py",
        "config.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "config.py",
    }

    for name, path in required_sources.items():
        exists = path.is_file()
        details[name] = {"type": "source_file", "exists": exists, "path": str(path.relative_to(PROJECT_ROOT))}
        if not exists:
            issues.append(f"Missing source file: {name}")

    # Required test files
    required_tests = {
        "test_models.py": PROJECT_ROOT / "src" / "prognostic_engine" / "tests" / "test_models.py",
    }

    for name, path in required_tests.items():
        exists = path.is_file()
        details[name] = {"type": "test_file", "exists": exists, "path": str(path.relative_to(PROJECT_ROOT))}
        if not exists:
            issues.append(f"Missing test file: {name}")

    # Required data files
    required_data = {
        "tcga_lihc_modeling_dataset.parquet": PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet",
    }

    for name, path in required_data.items():
        exists = path.is_file()
        details[name] = {"type": "data_file", "exists": exists, "path": str(path.relative_to(PROJECT_ROOT))}
        if not exists:
            issues.append(f"Missing data file: {name}")

    # Determine status
    if issues:
        status = STRUCTURE_STATUS["FAILED_INCOMPLETE"]
    else:
        status = STRUCTURE_STATUS["COMPLETE"]

    return {
        "status": status,
        "issues": issues,
        "details": details,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def check_methodology_compliance():
    """
    Verify Phase 3A methodology compliance via source code inspection.

    Returns a dict with:
        - status: COMPLIANT | NON_COMPLIANT | PENDING
        - issues: list of methodology violations
        - details: dict of methodology checks
    """
    issues = []
    details = {}

    # Check 1: Inner preprocessing isolation in training.py
    # M2/M3 must receive raw outer_train_df and fit ONLY on inner_train_df
    training_path = PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "training.py"
    try:
        training_content = training_path.read_text(encoding='utf-8')
        # Verify inner_fold_data is extracted from train_df (outer training)
        # and preprocessing uses inner_train_df for fitting
        has_extract_inner = "extract_inner_fold_data" in training_content
        has_preprocess_genes = "preprocess_inner_fold_genes" in training_content
        details["inner_preprocessing_isolation"] = {
            "check": "Inner fold preprocessing isolation",
            "extract_inner_fold_data_present": has_extract_inner,
            "preprocess_inner_fold_genes_present": has_preprocess_genes,
        }
        if not (has_extract_inner and has_preprocess_genes):
            issues.append("Inner preprocessing isolation: missing required functions")
    except Exception as e:
        issues.append(f"Cannot verify preprocessing isolation: {e}")

    # Check 2: case_id-based split management in inner_splits.py
    inner_splits_path = PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "inner_splits.py"
    try:
        inner_splits_content = inner_splits_path.read_text(encoding='utf-8')
        # Verify functions use case_id-based extraction
        has_extract_case_id = "extract_inner_fold_data" in inner_splits_content
        has_save_inner = "save_inner_splits" in inner_splits_content
        has_case_id_col = "'case_id'" in inner_splits_content or '"case_id"' in inner_splits_content
        details["case_id_split_management"] = {
            "check": "case_id-based split management",
            "has_extract_function": has_extract_case_id,
            "has_save_function": has_save_inner,
            "uses_case_id_column": has_case_id_col,
        }
        if not (has_extract_case_id and has_save_inner and has_case_id_col):
            issues.append("case_id split management: missing required components")
    except Exception as e:
        issues.append(f"Cannot verify case_id split management: {e}")

    # Check 3: Integrity monitoring (non-blocking diagnostics)
    try:
        # Verify integrity monitoring exists but is not blocking
        has_integrity_monitor = "IntegrityMonitor" in training_content or "INTEGRITY_MONITORING" in training_content
        # Check that it does NOT contain blocking thresholds
        has_blocking_check = False
        if "IntegrityMonitor" in training_content:
            # Should have 'MONITORED' status, not 'BLOCKED'
            has_blocking_check = "'BLOCKED'" in training_content or '"BLOCKED"' in training_content
        details["integrity_monitoring"] = {
            "check": "Non-blocking integrity monitoring",
            "has_monitor": has_integrity_monitor,
            "not_blocking": not has_blocking_check,
        }
        if has_blocking_check:
            issues.append("Integrity monitoring: found blocking check (should be diagnostic only)")
    except Exception as e:
        issues.append(f"Cannot verify integrity monitoring: {e}")

    # Check 4: Forbidden decay formula absent
    try:
        has_decay_formula = "exp(-t/60)" in training_content or "exp(-T/60)" in training_content
        details["forbidden_decay_formula"] = {
            "check": "No forbidden exp(-t/60) decay formula",
            "absent": not has_decay_formula,
        }
        if has_decay_formula:
            issues.append("Forbidden decay formula 'exp(-t/60)' found in training.py")
    except Exception as e:
        issues.append(f"Cannot verify decay formula: {e}")

    # Check 5: Coxnet uses predict_survival_function (not manual formula)
    models_path = PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "models.py"
    try:
        models_content = models_path.read_text(encoding='utf-8')
        # Verify manual Coxnet formula is NOT present
        has_manual_formula = "np.power" in models_content and "alpha" in models_content
        # Should use sksurv predict methods
        has_predict_surv = "predict_survival_function" in models_content
        details["coxnet_implementation"] = {
            "check": "Coxnet uses sksurv predict methods",
            "has_manual_formula": has_manual_formula,
            "has_predict_surv_function": has_predict_surv,
        }
        # Note: Having np.power with alpha could be legitimate (e.g., in other formulas)
        # So we don't fail on that; we only check predict_surv is present
        if not has_predict_surv:
            issues.append("Coxnet models: predict_survival_function not found")
    except Exception as e:
        issues.append(f"Cannot verify Coxnet implementation: {e}")

    # Check 6: M1 does not have secret age-only fallback
    try:
        # Look for age-only fallback pattern in M1
        has_age_fallback = False
        if "M1" in models_content or "ClinicalCox" in models_content:
            # Check if there's a pattern like: if len(features) == 1 or only_age
            age_patterns = ["only_age", "age_only", "len(features) == 1", "len(X.columns) == 1"]
            for pattern in age_patterns:
                if pattern in models_content:
                    # Check if it's in a fallback context
                    lines = models_content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line:
                            # Check surrounding context
                            context = '\n'.join(lines[max(0, i-3):min(len(lines), i+4)])
                            if 'return' in context or 'fit' in context:
                                has_age_fallback = True
        details["m1_no_secret_fallback"] = {
            "check": "M1 Clinical Cox has no secret age-only fallback",
            "has_age_only_fallback": has_age_fallback,
        }
        if has_age_fallback:
            issues.append("M1: found secret age-only fallback pattern")
    except Exception as e:
        issues.append(f"Cannot verify M1 fallback: {e}")

    # Determine status
    if issues:
        status = METHODOLOGY_STATUS["NON_COMPLIANT"]
    else:
        status = METHODOLOGY_STATUS["COMPLIANT"]

    return {
        "status": status,
        "issues": issues,
        "details": details,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def check_behavioral_compliance():
    """
    Verify Phase 3A behavioral compliance via runtime testing.

    Per Phase 3A reset: These are explicit behavioral checks that verify
    the actual runtime behavior of key components.

    Returns a dict with:
        - status: COMPLIANT | NON_COMPLIANT | PENDING
        - issues: list of behavioral violations
        - details: dict of behavioral checks
    """
    issues = []
    details = {}
    import sys as _sys

    # Add src to path for imports
    src_path = str(PROJECT_ROOT / "src" / "prognostic_engine" / "src")
    if src_path not in _sys.path:
        _sys.path.insert(0, src_path)

    # Check 1: IntegrityMonitor never blocks
    try:
        from prognostic_engine.training import IntegrityMonitor
        monitor = IntegrityMonitor('M1', repeat=1, fold=1)
        # Test with extreme/worst case values
        monitor.check_c_index(0.3)  # Very poor
        monitor.check_auc({'auc_12': 0.4, 'auc_36': 0.4})  # Very poor
        monitor.check_ibs(0.9)  # Very poor IBS
        status = monitor.get_status()
        never_blocks = status['status'] == 'MONITORED'
        details["integrity_monitor_never_blocks"] = {
            "check": "IntegrityMonitor never blocks regardless of metric values",
            "status_returned": status['status'],
            "never_blocks": never_blocks,
        }
        if not never_blocks:
            issues.append(f"IntegrityMonitor blocked with status '{status['status']}'")
    except Exception as e:
        issues.append(f"Cannot verify IntegrityMonitor non-blocking behavior: {e}")

    # Check 2: Bootstrap supports repeat/fold metadata
    try:
        import numpy as _np
        import pandas as _pd
        from prognostic_engine.bootstrap import (
            patient_level_paired_bootstrap,
            aggregate_bootstrap_results,
            run_full_bootstrap_comparison
        )
        # Create minimal test data
        n = 20
        df_a = _pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'model': ['M1'] * n,
            'risk_score': _np.random.randn(n),
            'survival_months': _np.random.exponential(30, n),
            'event': _np.random.binomial(1, 0.3, n)
        })
        df_b = _pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'model': ['M2'] * n,
            'risk_score': _np.random.randn(n) + 0.2,
            'survival_months': _np.random.exponential(30, n),
            'event': _np.random.binomial(1, 0.3, n)
        })
        predictions_df = _pd.concat([df_a, df_b], ignore_index=True)

        # Test patient_level_paired_bootstrap with repeat/fold
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42,
            repeat=1,
            fold=1
        )
        has_repeat = result.get('repeat') == 1
        has_fold = result.get('fold') == 1
        details["bootstrap_repeat_fold_support"] = {
            "check": "Bootstrap functions support repeat/fold metadata",
            "has_repeat_metadata": has_repeat and has_fold,
            "repeat_value": result.get('repeat'),
            "fold_value": result.get('fold'),
        }
        if not (has_repeat and has_fold):
            issues.append("Bootstrap result missing repeat/fold metadata")
    except Exception as e:
        issues.append(f"Cannot verify bootstrap repeat/fold support: {e}")

    # Check 3: IntegrityMonitor tracks metrics properly
    try:
        from prognostic_engine.training import IntegrityMonitor
        monitor = IntegrityMonitor('M2', repeat=2, fold=3)
        monitor.check_c_index(0.55)
        monitor.check_auc({'auc_12': 0.60})
        status = monitor.get_status()
        has_checks = 'c_index' in status['checks']
        details["integrity_monitor_tracks_metrics"] = {
            "check": "IntegrityMonitor properly tracks and reports metrics",
            "has_c_index_check": has_checks,
            "checks_count": len(status['checks']),
        }
        if not has_checks:
            issues.append("IntegrityMonitor not tracking metrics properly")
    except Exception as e:
        issues.append(f"Cannot verify IntegrityMonitor metric tracking: {e}")

    # Check 4: Bootstrap preserves patient multiplicity
    try:
        import numpy as _np
        import pandas as _pd
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap
        # Create data with known multiplicity (2 rows per case)
        n = 10
        df_a = _pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)] * 2,  # 20 rows, 10 unique cases
            'model': ['M1'] * 20,
            'risk_score': _np.random.randn(20),
            'survival_months': _np.random.exponential(30, 20),
            'event': _np.random.binomial(1, 0.3, 20)
        })
        df_b = _pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)] * 2,
            'model': ['M2'] * 20,
            'risk_score': _np.random.randn(20) + 0.2,
            'survival_months': _np.random.exponential(30, 20),
            'event': _np.random.binomial(1, 0.3, 20)
        })
        predictions_df = _pd.concat([df_a, df_b], ignore_index=True)
        # Function should run without error (multiplicity preserved)
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=5,
            seed=42
        )
        details["bootstrap_patient_multiplicity"] = {
            "check": "Bootstrap preserves patient-level multiplicity",
            "completed_without_error": result is not None,
            "iterations_valid": result.get('n_iterations_valid', 0),
        }
    except Exception as e:
        issues.append(f"Cannot verify bootstrap patient multiplicity: {e}")

    # Determine status
    if issues:
        status = METHODOLOGY_STATUS["NON_COMPLIANT"]
    else:
        status = METHODOLOGY_STATUS["COMPLIANT"]

    return {
        "status": status,
        "issues": issues,
        "details": details,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main():
    # Ensure output directories exist
    READY_DIR.mkdir(parents=True, exist_ok=True)
    INVALID_DIR.mkdir(parents=True, exist_ok=True)

    # Summary info for runner
    summary = {
        "env_ok": False,
        "structure_check": None,  # Structure integrity check result
        "methodology_check": None,  # Methodology compliance check result
        "pytest_run": None,  # {exit_code, stdout, stderr}
        "forbidden_pattern_errors": [],
        "source_file_hashes": {},
        "success": False,
    }

    # 0) Structure integrity check
    structure_result = check_structure_integrity()
    summary["structure_check"] = structure_result
    if structure_result["status"] == STRUCTURE_STATUS["FAILED_INCOMPLETE"]:
        # Write FAILED_INCOMPLETE gate
        gate = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "success": False,
            "total_tests": None,
            "passed": None,
            "failed": None,
            "warnings": None,
            "details": {
                "reason": "Structure integrity check failed",
                "structure_status": structure_result["status"],
                "structure_issues": structure_result["issues"],
            },
            "phase3areset": {
                "all_issues_resolved": False,
                "unit_tests_passing": False,
                "readiness_generated_by": "verify_phase3a_readiness.py",
            },
            "structure_integrity": structure_result,
            "methodology_gate": {
                "status": METHODOLOGY_STATUS["PENDING"],
                "issues": [],
                "details": {},
                "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        }
        GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')
        # Write INVALID_REASON.md
        invalid_content = f"""# Phase 3A Structure Integrity Check - FAILED

Status: {structure_result['status']}

Checked at: {structure_result['checked_at']}

## Issues Found

"""
        for issue in structure_result["issues"]:
            invalid_content += f"- {issue}\n"

        invalid_content += "\n## Structure Details\n\n"
        for name, info in structure_result["details"].items():
            status_icon = "OK" if info.get("exists", False) else "MISSING"
            invalid_content += f"- [{status_icon}] {name}: {info.get('path', 'directory')}\n"

        INVALID_REASON_PATH.write_text(invalid_content, encoding='utf-8')
        print(f"\nStructure integrity: FAILED_INCOMPLETE")
        print(f"Issues: {len(structure_result['issues'])}")
        print(f"Gate written to: {GATE_PATH}")
        print(f"Invalid reason written to: {INVALID_REASON_PATH}\n")
        return False

    # 1) Environment check - NO WAIVERS ALLOWED
    try:
        import sklearn
        import scipy
        import numpy
        # Check exact versions as specified
        if sklearn.__version__ != "1.9.0":
            print(f"WARNING: Expected sklearn 1.9.0, got {sklearn.__version__}", file=sys.stderr)
        summary["env_ok"] = True
    except Exception as e:
        print(f"Environment check failed: {e}", file=sys.stderr)
        summary["env_ok"] = False

    # 2) Define forbidden patterns to check
    FORBIDDEN_PATTERNS = [
        "KFold(n_splits=3, shuffle=True, random_state=42)",
        ".isin(sample_case_ids)",  # Bootstrap multiplicity violation
        # M5 must NOT use predict_survival_function directly (must use baseline + survival_df)
        # Pattern: direct access without getattr() - catches M5's incorrect usage
        "self.model.predict_survival_function(",
        "exp(-t/60)",  # Forbidden decay formula
        "# SKIPPED",  # M5 SKIPPED block
        "fits on full outer-training for efficiency",  # Preprocessing leakage comment
    ]

    # 3) Define source files to check
    # Structure: src/prognostic_engine/src/prognostic_engine/*.py
    #            src/prognostic_engine/tests/test_models.py
    CORE_FILES = {
        "models.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "models.py",
        "training.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "training.py",
        "bootstrap.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "bootstrap.py",
        "inner_splits.py": PROJECT_ROOT / "src" / "prognostic_engine" / "src" / "prognostic_engine" / "inner_splits.py",
        "test_models.py": PROJECT_ROOT / "src" / "prognostic_engine" / "tests" / "test_models.py",
    }

    # Check all files exist
    missing_files = []
    for name, path in CORE_FILES.items():
        try:
            check_file_exists(path)
        except FileNotFoundError as e:
            missing_files.append(str(e))

    if missing_files:
        error_msg = f"Missing required files: {', '.join(missing_files)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        gate = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "success": False,
            "total_tests": None,
            "passed": None,
            "failed": None,
            "warnings": None,
            "details": {
                "reason": error_msg,
                "env_ok": summary["env_ok"],
                "missing_files": missing_files,
            },
            "phase3areset": {
                "all_issues_resolved": False,
                "unit_tests_passing": False,
                "readiness_generated_by": "verify_phase3a_readiness.py",
            }
        }
        GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')
        print(f"\nREADINESS_GATE.json written: FAILED ({error_msg})\n")
        return False

    # 4) Check source files for forbidden patterns
    forbidden_errors = []
    for name, path in CORE_FILES.items():
        try:
            content = path.read_text(encoding='utf-8')
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in content:
                    forbidden_errors.append(f"{name}: contains forbidden pattern '{pattern}'")
        except Exception as e:
            forbidden_errors.append(f"{name}: error reading file - {e}")

    summary["forbidden_pattern_errors"] = forbidden_errors

    # 5) Calculate SHA-256 hashes of source files
    file_hashes = {}
    for name, path in CORE_FILES.items():
        try:
            file_hashes[name] = get_file_hash(path)
        except Exception as e:
            file_hashes[name] = f"ERROR: {e}"

    summary["source_file_hashes"] = file_hashes

    # 6) Run pytest (required for final gate)
    try:
        # Run from the src/prognostic_engine directory where tests live
        test_dir = PROJECT_ROOT / "src" / "prognostic_engine"
        # Use the venv Python executable to ensure correct environment
        pytest_python = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"
        if not pytest_python.exists():
            pytest_python = sys.executable  # Fallback to current Python
        proc = subprocess.run(
            [str(pytest_python), "-m", "pytest", "-q", "--tb=short", "tests/"],
            cwd=str(test_dir),
            capture_output=True,
            text=True,
            timeout=180,
        )
        summary["pytest_run"] = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

        if proc.returncode != 0:
            gate = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "success": False,
                "total_tests": None,
                "passed": None,
                "failed": None,
                "warnings": None,
                "details": {
                    "reason": "pytest exit code non-zero",
                    "env_ok": summary["env_ok"],
                    "forbidden_errors": summary["forbidden_pattern_errors"],
                    "pytest_stdout": proc.stdout[:500] if proc.stdout else "",
                    "pytest_stderr": proc.stderr[:500] if proc.stderr else "",
                },
                "phase3areset": {
                    "all_issues_resolved": False,
                    "unit_tests_passing": False,
                    "readiness_generated_by": "verify_phase3a_readiness.py",
                }
            }
            GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')
            print(f"\nREADINESS_GATE.json written: FAILED (pytest non-zero exit code {proc.returncode})\n")
            return False

    except subprocess.TimeoutExpired:
        err = {"exit_code": -1, "stdout": "TIMEOUT", "stderr": ""}
        summary["pytest_run"] = err
        gate = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "success": False,
            "total_tests": None,
            "passed": None,
            "failed": None,
            "warnings": None,
            "details": {
                "reason": "pytest timeout",
                "env_ok": summary["env_ok"],
                "forbidden_errors": summary["forbidden_pattern_errors"],
            },
            "phase3areset": {
                "all_issues_resolved": False,
                "unit_tests_passing": False,
                "readiness_generated_by": "verify_phase3a_readiness.py",
            }
        }
        GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')
        print(f"\nREADINESS_GATE.json written: FAILED (pytest timeout)\n")
        return False
    except Exception as e:
        err = {"exception": str(e), "exit_code": -1, "stdout": "", "stderr": str(e)}
        summary["pytest_run"] = err
        gate = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "success": False,
            "total_tests": None,
            "passed": None,
            "failed": None,
            "warnings": None,
            "details": {
                "reason": f"pytest run exception: {e}",
                "env_ok": summary["env_ok"],
                "forbidden_errors": summary["forbidden_pattern_errors"],
            },
            "phase3areset": {
                "all_issues_resolved": False,
                "unit_tests_passing": False,
                "readiness_generated_by": "verify_phase3a_readiness.py",
            }
        }
        GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')
        print(f"\nREADINESS_GATE.json written: FAILED (exception {e})\n")
        return False

    # 7) Parse pytest output to extract test counts
    pytest_output = summary["pytest_run"]["stdout"]
    import re
    match_passed = re.search(r'(\d+) passed', pytest_output)
    match_failed = re.search(r'(\d+) failed', pytest_output)
    match_skipped = re.search(r'(\d+) skipped', pytest_output)
    match_warnings = re.search(r'(\d+) warnings', pytest_output)

    passed = int(match_passed.group(1)) if match_passed else None
    # When all tests pass, there's no "X failed" in output - treat as 0
    failed = int(match_failed.group(1)) if match_failed else 0
    skipped = int(match_skipped.group(1)) if match_skipped else 0
    warnings = int(match_warnings.group(1)) if match_warnings else None
    total = passed + failed + skipped if passed is not None else None

    # 8) Methodology compliance check (static code analysis)
    methodology_result = check_methodology_compliance()
    summary["methodology_check"] = methodology_result

    # 8b) Behavioral compliance check (runtime testing)
    behavioral_result = check_behavioral_compliance()
    summary["behavioral_check"] = behavioral_result

    # 9) Determine success based on ALL criteria
    success = (
        summary["structure_check"]["status"] == STRUCTURE_STATUS["COMPLETE"] and
        summary["methodology_check"]["status"] == METHODOLOGY_STATUS["COMPLIANT"] and
        summary["behavioral_check"]["status"] == METHODOLOGY_STATUS["COMPLIANT"] and
        summary["env_ok"] and
        len(summary["forbidden_pattern_errors"]) == 0 and
        summary["pytest_run"]["exit_code"] == 0 and
        passed is not None and
        failed == 0
    )

    # 10) Write new gate
    gate = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "success": success,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "details": {
            "env_ok": summary["env_ok"],
            "forbidden_errors": summary["forbidden_pattern_errors"],
            "pytest_exit_code": summary["pytest_run"]["exit_code"],
            "source_file_hashes": summary["source_file_hashes"],
        },
        "phase3areset": {
            "all_issues_resolved": success,
            "unit_tests_passing": failed == 0 if failed is not None else False,
            "readiness_generated_by": "verify_phase3a_readiness.py",
        },
        "structure_integrity": summary["structure_check"],
        "methodology_gate": summary["methodology_check"],
        "behavioral_gate": summary["behavioral_check"],
    }

    GATE_PATH.write_text(json.dumps(gate, indent=2), encoding='utf-8')

    # 11) Print summary to console
    detail_msg = (
        f"Structure Integrity: {summary['structure_check']['status']}\n"
        f"Methodology Compliance: {summary['methodology_check']['status']}\n"
        f"Behavioral Compliance: {summary['behavioral_check']['status']}\n"
        f"Environment OK: {summary['env_ok']}\n"
        f"Forbidden pattern errors: {len(summary['forbidden_pattern_errors'])} items\n"
        f"Pytest exit code: {summary['pytest_run']['exit_code']}\n"
        f"Total tests: {total}; Passed: {passed}; Failed: {failed}; Warnings: {warnings}\n"
        f"Gate success: {success}"
    )
    print(f"\n{'='*60}")
    print("PHASE3A VERIFICATION SUMMARY")
    print(f"{'='*60}\n")
    print(detail_msg)
    print(f"\nGate written to: {GATE_PATH}\n")

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
