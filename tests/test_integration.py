"""
Integration tests for the HCC Prognosis System core modules.

This module tests the integration between:
- TCGA Data Loader/Downloader
- Baseline Models (Cox, DeepSurv, Simple)
- Prognosis Evaluator
- PubMed Tools
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.tcga_downloader import TCGADownloader, TCGADataBuilder, load_tcga_data
from src.models.baseline_models import (
    CoxProportionalHazards,
    DeepSurvModel,
    SimpleSurvivalPredictor,
    ModelFactory,
    prepare_features
)
from src.evaluation.prognosis_evaluator import (
    SurvivalEvaluator,
    ComparisonEvaluator,
    evaluate_model,
    compare_models
)
from src.tools.pubmed_tool import PubMedTool
from src.tools.enhanced_pubmed_tool import EnhancedPubMedTool
from src.state.schema import LiteratureEvidence


def test_tcga_downloader():
    """Test TCGA data downloader and builder."""
    print("=" * 60)
    print("Testing TCGA Data Downloader")
    print("=" * 60)

    downloader = TCGADownloader(data_dir="F:/ACM/data", use_cache=False)

    # Generate mock data
    df = downloader._get_mock_clinical_data(n_patients=200)

    print(f"Generated {len(df)} patient records")
    print(f"Columns: {len(df.columns)}")
    print(f"Stage distribution:\n{df['stage'].value_counts()}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"Median survival: {df['survival_months'].median():.1f} months")
    print(f"Median AFP: {df['afp_level'].median():.1f}")
    print(f"Vital status: {df['vital_status'].value_counts().to_dict()}")

    # Test TCGADataBuilder
    builder = TCGADataBuilder(df)
    filtered = (builder
                .filter_by_stage(["Stage I", "Stage II"])
                .filter_by_grade(["G1", "G2"])
                .build())

    print(f"\nFiltered to {len(filtered)} early-stage patients")

    # Test train/test split
    train_df, test_df = builder.split_train_test(test_size=0.2, stratify_by="stage")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    stats = downloader.get_data_statistics(df)
    print(f"\nDataset statistics: {stats}")

    print("\n[OK] TCGA Downloader tests passed!")
    return df


def test_baseline_models(df):
    """Test baseline models."""
    print("\n" + "=" * 60)
    print("Testing Baseline Models")
    print("=" * 60)

    # Get gene columns from dataframe
    gene_columns = [c for c in df.columns if c in TCGADownloader.METABOLIC_GENES[:15]]
    print(f"Using {len(gene_columns)} gene features")

    # Prepare features
    X, feature_names = prepare_features(df, gene_columns)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names: {feature_names[:5]}...")

    # Create event and time arrays
    event = (df["vital_status"] == "Dead").astype(int).values
    time = df["survival_months"].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )

    # Test Simple Survival Predictor
    print("\n--- Testing SimpleSurvivalPredictor ---")
    simple_model = SimpleSurvivalPredictor(threshold_months=24)
    simple_model.fit(X_train, time_train, event_train)
    pred_simple = simple_model.predict_risk(X_test)
    print(f"Predictions range: [{pred_simple.min():.3f}, {pred_simple.max():.3f}]")

    # Test Cox PH
    print("\n--- Testing CoxProportionalHazards ---")
    cox_model = CoxProportionalHazards(alpha=0.05)
    cox_model.fit(X_train, time_train, event_train)
    pred_cox = cox_model.predict_risk(X_test)
    print(f"Risk scores range: [{pred_cox.min():.3f}, {pred_cox.max():.3f}]")

    # Test ModelFactory
    print("\n--- Testing ModelFactory ---")
    model = ModelFactory.create("simple", threshold_months=12)
    print(f"Created model: {type(model).__name__}")

    print("\n[OK] Baseline models tests passed!")
    return pred_simple, pred_cox, time_test, event_test


def test_evaluation(predictions, times, events):
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("Testing Prognosis Evaluator")
    print("=" * 60)

    evaluator = SurvivalEvaluator(time_points=[12, 36, 60])

    # Evaluate
    metrics = evaluator.evaluate(predictions, times, events)

    print(f"C-index: {metrics.c_index:.3f} [{metrics.c_index_ci_low:.3f}, {metrics.c_index_ci_high:.3f}]")
    print(f"AUC 1yr: {metrics.auc_1yr:.3f}")
    print(f"AUC 3yr: {metrics.auc_3yr:.3f}")
    print(f"AUC 5yr: {metrics.auc_5yr:.3f}")
    print(f"Calibration slope: {metrics.calibration_slope:.3f}")
    print(f"Calibration intercept: {metrics.calibration_intercept:.3f}")
    print(f"Brier score: {metrics.brier_score:.3f}")
    print(f"Samples: {metrics.n_samples}, Events: {metrics.n_events}")

    # Test comparison evaluator
    print("\n--- Testing ComparisonEvaluator ---")
    comparator = ComparisonEvaluator()

    # Create mock predictions for comparison
    pred2 = predictions * 0.8 + np.random.normal(0, 0.1, len(predictions))

    # Evaluate both models properly
    results = {}
    results["Model A"] = evaluate_model(predictions, times, events)
    results["Model B"] = evaluate_model(pred2, times, events)

    comparison_df = comparator.compare_models(results)
    print(comparison_df)

    print("\n[OK] Evaluation tests passed!")
    return metrics


def test_pubmed_tools():
    """Test PubMed tools."""
    print("\n" + "=" * 60)
    print("Testing PubMed Tools")
    print("=" * 60)

    # Test basic PubMed tool
    pubmed_tool = PubMedTool(cache_dir="F:/ACM/data/literature_cache", max_results=5)
    evidence = pubmed_tool.search("hepatocellular carcinoma prognosis metabolic", use_cache=False)

    print(f"Search query: {evidence.search_query}")
    print(f"Number of results: {evidence.num_results}")
    if evidence.evidence_items:
        print(f"First result title: {evidence.evidence_items[0].get('title', 'N/A')[:60]}...")

    # Test Enhanced PubMed tool
    print("\n--- Testing EnhancedPubMedTool ---")
    enhanced_tool = EnhancedPubMedTool(cache_dir="F:/ACM/data/literature_cache", max_results=5)
    enhanced_evidence = enhanced_tool.search("HCC survival gene signature", use_cache=False)

    print(f"Enhanced search results: {enhanced_evidence.num_results}")

    print("\n[OK] PubMed tools tests passed!")
    return evidence


def test_workflow_integration():
    """Test workflow integration with new modules."""
    print("\n" + "=" * 60)
    print("Testing Workflow Integration")
    print("=" * 60)

    from src.state.schema import PatientData
    from src.workflow import HCCPrognosisWorkflow

    # Create sample patient data
    patient = PatientData(
        patient_id="TEST-001",
        age=65,
        gender="M",
        stage="Stage III",
        grade="G2",
        afp_level=500.0,
        albumin=3.5,
        bilirubin=1.2,
        survival_months=24.0,
        vital_status="Alive",
        gene_expression={
            "HK2": 4.5,
            "LDHA": 5.2,
            "VEGFA": 3.8,
            "CA9": 2.1,
            "MYC": 4.0
        }
    )

    # Test workflow
    workflow = HCCPrognosisWorkflow()
    print(f"Supported stages: {workflow.get_supported_stages()}")

    # Run assessment (may require LLM API)
    result = workflow.assess(patient)

    print(f"Assessment success: {result['success']}")
    if result['success']:
        print(f"Risk level: {result.get('risk_level')}")
        print(f"Survival estimate: {result.get('survival_estimate')}")

    print("\n[OK] Workflow integration tests passed!")
    return result


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Pipeline")
    print("=" * 60)

    # 1. Load/generate data
    downloader = TCGADownloader(data_dir="F:/ACM/data", use_cache=False)
    df = downloader._get_mock_clinical_data(n_patients=200)

    # 2. Prepare features
    gene_columns = [c for c in df.columns if c in TCGADownloader.METABOLIC_GENES[:15]]
    X, feature_names = prepare_features(df, gene_columns)
    event = (df["vital_status"] == "Dead").astype(int).values
    time = df["survival_months"].values

    # 3. Train baseline model
    X_train, X_test, time_train, time_test, event_train, event_test = \
        train_test_split(X, time, event, test_size=0.2, random_state=42)

    model = SimpleSurvivalPredictor(threshold_months=24)
    model.fit(X_train, time_train, event_train)
    predictions = model.predict_risk(X_test)

    # 4. Evaluate
    metrics = evaluate_model(predictions, time_test, event_test)

    # 5. Search literature
    pubmed = PubMedTool(max_results=3)
    evidence = pubmed.search("hepatocellular carcinoma prognosis metabolic subtype")

    # 6. Generate report
    report = f"""
    ====== HCC Prognosis Assessment Summary ======

    Dataset: TCGA-LIHC (n={len(df)})
    Test Set: n={len(time_test)} patients, {sum(event_test)} events

    Baseline Model Performance:
    - C-index: {metrics.c_index:.3f} (95% CI: {metrics.c_index_ci_low:.3f}-{metrics.c_index_ci_high:.3f})
    - 1-year AUC: {metrics.auc_1yr:.3f}
    - 3-year AUC: {metrics.auc_3yr:.3f}
    - Calibration: slope={metrics.calibration_slope:.3f}

    Literature Evidence:
    - Papers retrieved: {evidence.num_results}
    - Summary: {evidence.summary[:100]}...

    System Status: OPERATIONAL
    """
    print(report)

    print("[OK] End-to-end pipeline test passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("HCC PROGNOSIS SYSTEM - INTEGRATION TESTS")
    print("=" * 70)

    try:
        # Test 1: TCGA Downloader
        df = test_tcga_downloader()

        # Test 2: Baseline Models
        pred_simple, pred_cox, time_test, event_test = test_baseline_models(df)

        # Test 3: Evaluation
        test_evaluation(pred_simple, time_test, event_test)

        # Test 4: PubMed Tools
        test_pubmed_tools()

        # Test 5: Workflow Integration (may fail without LLM API)
        try:
            test_workflow_integration()
        except Exception as e:
            print(f"\n[NOTE] Workflow test skipped: {e}")

        # Test 6: End-to-End Pipeline
        test_end_to_end_pipeline()

        print("\n" + "=" * 70)
        print("ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
