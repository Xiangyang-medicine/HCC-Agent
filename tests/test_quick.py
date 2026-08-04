"""
Quick integration tests for the HCC Prognosis System core modules.
Skip slow models for faster testing.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.tcga_downloader import TCGADownloader, TCGADataBuilder
from src.models.baseline_models import (
    SimpleSurvivalPredictor,
    ModelFactory,
    prepare_features
)
from src.evaluation.prognosis_evaluator import (
    SurvivalEvaluator,
    evaluate_model
)
from src.tools.pubmed_tool import PubMedTool


def test_tcga_downloader():
    """Test TCGA data downloader."""
    print("Testing TCGA Data Downloader...")
    downloader = TCGADownloader(data_dir="F:/ACM/data", use_cache=False)
    df = downloader._get_mock_clinical_data(n_patients=100)
    print(f"  Generated {len(df)} patient records")
    print(f"  Stage distribution: {df['stage'].value_counts().to_dict()}")
    print("  [OK]")
    return df


def test_baseline_models(df):
    """Test baseline models."""
    print("\nTesting Baseline Models...")

    gene_columns = [c for c in df.columns if c in TCGADownloader.METABOLIC_GENES[:10]]
    X, feature_names = prepare_features(df, gene_columns)
    event = (df["vital_status"] == "Dead").astype(int).values
    time = df["survival_months"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )

    # Test Simple model (fast)
    model = SimpleSurvivalPredictor(threshold_months=24)
    model.fit(X_train, time_train, event_train)
    predictions = model.predict_risk(X_test)
    print(f"  SimpleSurvivalPredictor predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Test ModelFactory
    model2 = ModelFactory.create("simple", threshold_months=12)
    print(f"  ModelFactory created: {type(model2).__name__}")

    print("  [OK]")
    return predictions, time_test, event_test


def test_evaluation(predictions, times, events):
    """Test evaluation metrics."""
    print("\nTesting Prognosis Evaluator...")
    evaluator = SurvivalEvaluator(time_points=[12, 36])
    metrics = evaluator.evaluate(predictions, times, events)
    print(f"  C-index: {metrics.c_index:.3f} [{metrics.c_index_ci_low:.3f}, {metrics.c_index_ci_high:.3f}]")
    print(f"  AUC 1yr: {metrics.auc_1yr:.3f}")
    print(f"  Brier score: {metrics.brier_score:.3f}")
    print("  [OK]")
    return metrics


def test_pubmed_tools():
    """Test PubMed tools."""
    print("\nTesting PubMed Tools...")
    pubmed_tool = PubMedTool(cache_dir="F:/ACM/data/literature_cache", max_results=3)
    evidence = pubmed_tool.search("hepatocellular carcinoma prognosis", use_cache=False)
    print(f"  Search returned {evidence.num_results} results")
    print("  [OK]")
    return evidence


def test_end_to_end():
    """Quick end-to-end test."""
    print("\nTesting End-to-End Pipeline...")

    # Load data
    downloader = TCGADownloader(data_dir="F:/ACM/data", use_cache=False)
    df = downloader._get_mock_clinical_data(n_patients=100)

    # Prepare features
    gene_columns = [c for c in df.columns if c in TCGADownloader.METABOLIC_GENES[:10]]
    X, feature_names = prepare_features(df, gene_columns)
    event = (df["vital_status"] == "Dead").astype(int).values
    time = df["survival_months"].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )

    # Train model
    model = SimpleSurvivalPredictor(threshold_months=24)
    model.fit(X_train, time_train, event_train)
    predictions = model.predict_risk(X_test)

    # Evaluate
    metrics = evaluate_model(predictions, time_test, event_test)

    # Search literature
    pubmed = PubMedTool(max_results=3)
    evidence = pubmed.search("HCC metabolic subtype prognosis")

    print(f"  Pipeline completed successfully")
    print(f"  Model C-index: {metrics.c_index:.3f}")
    print(f"  Literature results: {evidence.num_results}")
    print("  [OK]")


def run_quick_tests():
    """Run quick integration tests."""
    print("=" * 60)
    print("HCC PROGNOSIS SYSTEM - QUICK INTEGRATION TESTS")
    print("=" * 60)

    try:
        df = test_tcga_downloader()
        pred, times, events = test_baseline_models(df)
        test_evaluation(pred, times, events)
        test_pubmed_tools()
        test_end_to_end()

        print("\n" + "=" * 60)
        print("ALL QUICK TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_quick_tests()
