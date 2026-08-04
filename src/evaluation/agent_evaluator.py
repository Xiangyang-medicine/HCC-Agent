"""
LLM Agent Evaluation Module for HCC Prognosis System.

This module provides utilities for evaluating the multi-agent LLM system
against traditional baseline models (Cox PH, DeepSurv, Logistic Regression).

Key evaluation dimensions:
1. Prediction Accuracy: C-index, time-dependent AUC
2. Clinical Utility: Decision curve analysis
3. Explainability: Report quality, doctor preferences
4. Response Quality: LLM output evaluation
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from src.state.schema import PatientData, AgentState, RiskLevel
from src.models.baseline_models import (
    CoxProportionalHazards,
    DeepSurvModel,
    SimpleSurvivalPredictor,
    prepare_features
)
from src.evaluation.prognosis_evaluator import (
    SurvivalEvaluator,
    ComparisonEvaluator,
    DecisionCurveAnalysis,
    evaluate_model,
    EvaluationMetrics
)


class EvaluationMode(Enum):
    """Evaluation mode enumeration."""
    BASELINE_ONLY = "baseline_only"
    LLM_AGENT_ONLY = "llm_agent_only"
    COMPARISON = "comparison"


@dataclass
class AgentPrediction:
    """Prediction output from LLM agent."""
    risk_score: float
    risk_level: str
    survival_months_estimate: float
    confidence: float
    reasoning: str
    report_text: str


@dataclass
class AgentEvaluationResult:
    """Results from agent evaluation."""
    patient_id: str
    true_survival_months: float
    true_event: int
    predicted_risk_score: float
    predicted_risk_level: str
    predicted_survival_months: float
    prediction_confidence: float
    report_quality_score: float
    clinical_validity_score: float
    error: Optional[str] = None


@dataclass
class ComparativeEvaluationResult:
    """Results comparing agent vs baseline models."""
    # Patient info
    patient_id: str
    true_survival_months: float
    true_event: int

    # Baseline model predictions
    baseline_predictions: Dict[str, float]
    baseline_risk_levels: Dict[str, str]

    # LLM Agent predictions
    agent_risk_score: float
    agent_risk_level: str
    agent_survival_estimate: float
    agent_confidence: float

    # Evaluation metrics
    baseline_c_index: float
    agent_c_index_approx: float

    # Quality metrics
    agent_report_quality: float
    agent_clinical_validity: float

    # Differences
    risk_level_agreement: bool
    survival_estimate_error: float


class PatientGenerator:
    """
    Generate synthetic patients for evaluation.

    Creates realistic patient profiles that test various
    clinical scenarios and edge cases.
    """

    METABOLIC_GENES = [
        "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",
        "GLS", "GLUD1", "FASN", "SCD", "CA9", "VEGFA",
        "MYC", "HIF1A", "CTNNB1", "BCL2", "CCND1"
    ]

    def __init__(self, seed: int = 42):
        """Initialize generator."""
        self.rng = np.random.RandomState(seed)

    def generate_patient(
        self,
        patient_id: str = None,
        age: int = None,
        gender: str = None,
        stage: str = None,
        include_expression: bool = True
    ) -> PatientData:
        """
        Generate a single patient.

        Args:
            patient_id: Patient ID
            age: Age (random if None)
            gender: Gender (random if None)
            stage: Cancer stage (random if None)
            include_expression: Include gene expression

        Returns:
            PatientData object
        """
        patient_id = patient_id or f"TEST-{self.rng.randint(1000,9999)}"

        # Clinical features
        if age is None:
            age = self.rng.normal(65, 12)
            age = int(np.clip(age, 35, 85))

        if gender is None:
            gender = self.rng.choice(["M", "F"], p=[0.65, 0.35])

        if stage is None:
            stage = self.rng.choice(
                ["Stage I", "Stage II", "Stage III", "Stage IV"],
                p=[0.35, 0.20, 0.30, 0.15]
            )

        grade = self.rng.choice(["G1", "G2", "G3", "G4"], p=[0.10, 0.40, 0.35, 0.15])

        # Lab values
        afp_level = self.rng.lognormal(4.5, 1.5)
        afp_level = float(np.clip(afp_level, 2, 50000))

        albumin = float(np.clip(self.rng.normal(3.8, 0.5), 2.5, 4.8))
        bilirubin = float(np.clip(self.rng.exponential(1.2), 0.3, 8.0))

        # Survival outcome (for ground truth)
        survival_base = {"Stage I": 60, "Stage II": 45, "Stage III": 24, "Stage IV": 12}
        base_months = survival_base.get(stage, 24)
        survival_months = self.rng.exponential(base_months * 0.8)

        # Vital status
        if self.rng.random() < 0.4:
            vital_status = "Dead"
        else:
            vital_status = "Alive"
            survival_months = max(survival_months, 12)  # Minimum follow-up

        # Gene expression (as z-scores from normal liver)
        # Positive z-score = upregulated in tumor vs normal
        gene_expression = {}
        if include_expression:
            for gene in self.METABOLIC_GENES[:10]:
                # Base z-score: centered around 0
                base_expr = self.rng.normal(0, 0.8)

                # Adjust based on stage (advanced = more metabolic reprogramming)
                if stage in ["Stage III", "Stage IV"]:
                    base_expr += self.rng.normal(0.8, 0.3)  # Shift up
                elif stage == "Stage II":
                    base_expr += self.rng.normal(0.3, 0.3)  # Slight shift

                # Adjust based on outcome (dead = more aggressive metabolism)
                if vital_status == "Dead":
                    base_expr += self.rng.normal(0.3, 0.2)

                # Add individual variation
                base_expr += self.rng.normal(0, 0.4)

                gene_expression[gene] = round(float(base_expr), 3)

        return PatientData(
            patient_id=patient_id,
            age=age,
            gender=gender,
            stage=stage,
            grade=grade,
            afp_level=afp_level,
            albumin=albumin,
            bilirubin=bilirubin,
            survival_months=float(survival_months),
            vital_status=vital_status,
            gene_expression=gene_expression
        )

    def generate_cohort(
        self,
        n_patients: int = 100,
        stratified: bool = True
    ) -> Tuple[List[PatientData], pd.DataFrame]:
        """
        Generate a patient cohort for evaluation.

        Args:
            n_patients: Number of patients
            stratified: Stratify by stage

        Returns:
            Tuple of (list of PatientData, DataFrame for modeling)
        """
        patients = []

        if stratified:
            # Stratified sampling by stage - ensure at least 1 per stage
            stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
            weights = [0.35, 0.20, 0.30, 0.15]

            # For small cohorts, use proportional allocation with minimum of 1
            stage_counts = []
            raw_counts = np.array(weights) * n_patients
            for i, (stage, raw) in enumerate(zip(stages, raw_counts)):
                if n_patients <= 4:
                    # For small cohorts, allocate at least 1 to each stage
                    stage_counts.append(1)
                else:
                    stage_counts.append(int(round(raw)))

            # Adjust to match exact n_patients
            total = sum(stage_counts)
            if total != n_patients:
                stage_counts[0] += n_patients - total  # Add/subtract from Stage I

            patient_id = 0
            for stage, count in zip(stages, stage_counts):
                for _ in range(count):
                    patient = self.generate_patient(
                        patient_id=f"P{patient_id:04d}",
                        stage=stage
                    )
                    patients.append(patient)
                    patient_id += 1
        else:
            for i in range(n_patients):
                patient = self.generate_patient(patient_id=f"P{i:04d}")
                patients.append(patient)

        # Create DataFrame for modeling
        df = self.patients_to_dataframe(patients)

        return patients, df

    def patients_to_dataframe(self, patients: List[PatientData]) -> pd.DataFrame:
        """Convert patient list to DataFrame."""
        records = []

        for p in patients:
            record = {
                "patient_id": p.patient_id,
                "age": p.age,
                "gender": p.gender,
                "stage": p.stage,
                "grade": p.grade,
                "afp_level": p.afp_level,
                "albumin": p.albumin,
                "bilirubin": p.bilirubin,
                "survival_months": p.survival_months,
                "vital_status": p.vital_status,
            }
            # Add gene expression
            for gene, expr in p.gene_expression.items():
                record[gene] = expr
            records.append(record)

        return pd.DataFrame(records)


class BaselineModelWrapper:
    """Wrapper for baseline models to provide consistent interface."""

    def __init__(self, model_type: str = "cox", **kwargs):
        """
        Initialize wrapper.

        Args:
            model_type: Type of model ('cox', 'deepsurv', 'simple')
            **kwargs: Model-specific arguments
        """
        self.model_type = model_type

        if model_type == "cox":
            self.model = CoxProportionalHazards(**kwargs)
        elif model_type == "deepsurv":
            self.model = DeepSurvModel(**kwargs)
        elif model_type == "simple":
            self.model = SimpleSurvivalPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        """Fit the model."""
        self.model.fit(X, time, event)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        return self.model.predict_risk(X)

    def get_risk_level(self, risk_scores: np.ndarray) -> np.ndarray:
        """Convert risk scores to risk levels."""
        # Simple threshold-based classification
        low_mask = risk_scores < np.percentile(risk_scores, 33)
        high_mask = risk_scores > np.percentile(risk_scores, 66)

        levels = np.array(["Medium"] * len(risk_scores))
        levels[low_mask] = "Low"
        levels[high_mask] = "High"

        return levels


class AgentEvaluator:
    """
    Evaluator for LLM agent predictions.

    This class provides utilities for:
    - Evaluating agent predictions against ground truth
    - Scoring report quality
    - Assessing clinical validity
    """

    def __init__(
        self,
        evaluator: SurvivalEvaluator = None,
        time_points: List[float] = None
    ):
        """
        Initialize evaluator.

        Args:
            evaluator: SurvivalEvaluator instance
            time_points: Time points for evaluation
        """
        self.evaluator = evaluator or SurvivalEvaluator(time_points=time_points or [12, 36, 60])

    def evaluate_prediction(
        self,
        agent_result: AgentEvaluationResult,
        ground_truth_time: float,
        ground_truth_event: int
    ) -> AgentEvaluationResult:
        """
        Evaluate a single agent prediction.

        Args:
            agent_result: Agent evaluation result
            ground_truth_time: True survival time
            ground_truth_event: True event indicator

        Returns:
            Updated AgentEvaluationResult with error metrics
        """
        # Calculate survival estimate error
        agent_result.survival_estimate_error = abs(
            agent_result.predicted_survival_months - ground_truth_time
        )

        return agent_result

    def score_report_quality(
        self,
        report_text: str,
        patient: PatientData
    ) -> float:
        """
        Score the quality of agent-generated report.

        Args:
            report_text: Generated report text
            patient: Patient data

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        # Check for required sections
        required_sections = [
            "risk", "prognosis", "stage", "survival",
            "recommendation", "treatment"
        ]

        report_lower = report_text.lower()

        # Section coverage
        section_score = sum(1 for s in required_sections if s in report_lower)
        score += (section_score / len(required_sections)) * 0.5

        # Patient-specific mentions
        patient_mentions = 0
        if patient.stage.lower() in report_lower:
            patient_mentions += 1
        if patient.gender.lower() in report_lower:
            patient_mentions += 0.5
        if any(gene.lower() in report_lower for gene in patient.gene_expression.keys()):
            patient_mentions += 1

        score += min(patient_mentions / 3, 0.3)

        # Report length (not too short, not too long)
        word_count = len(report_text.split())
        if 100 <= word_count <= 500:
            score += 0.2
        elif 50 <= word_count < 100:
            score += 0.1

        return min(score, 1.0)

    def score_clinical_validity(
        self,
        risk_level: str,
        patient: PatientData
    ) -> float:
        """
        Score clinical validity of risk assessment.

        Args:
            risk_level: Predicted risk level
            patient: Patient data

        Returns:
            Validity score (0-1)
        """
        score = 0.5

        # Stage-based expected risk
        expected_risk = {
            "Stage I": "Low",
            "Stage II": "Low-Medium",
            "Stage III": "Medium-High",
            "Stage IV": "High"
        }

        expected = expected_risk.get(patient.stage, "Medium")

        # Risk level agreement
        risk_order = {"Low": 0, "Low-Medium": 1, "Medium": 2, "Medium-High": 3, "High": 4}
        expected_order = risk_order.get(expected, 2)
        predicted_order = risk_order.get(risk_level, 2)

        # Within one level is good
        if abs(expected_order - predicted_order) <= 1:
            score += 0.4
        elif abs(expected_order - predicted_order) == 2:
            score += 0.2

        # AFP level correlation (high AFP = higher risk)
        if patient.afp_level > 1000 and risk_level in ["High", "Medium-High"]:
            score += 0.1

        return min(score, 1.0)

    def evaluate_batch(
        self,
        results: List[AgentEvaluationResult],
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of agent predictions.

        Args:
            results: List of agent evaluation results
            predictions: Agent risk predictions
            times: Ground truth survival times
            events: Ground truth events

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = self.evaluator.evaluate(predictions, times, events)

        # Calculate additional quality metrics
        report_qualities = [r.report_quality_score for r in results]
        clinical_validities = [r.clinical_validity_score for r in results]

        return {
            "survival_metrics": metrics,
            "c_index": metrics.c_index,
            "c_index_ci": (metrics.c_index_ci_low, metrics.c_index_ci_high),
            "auc_1yr": metrics.auc_1yr,
            "auc_3yr": metrics.auc_3yr,
            "brier_score": metrics.brier_score,
            "avg_report_quality": np.mean(report_qualities),
            "avg_clinical_validity": np.mean(clinical_validities),
            "n_patients": len(results),
            "n_events": sum(r.true_event for r in results)
        }


class ComparativeEvaluator:
    """
    Compare LLM agent against baseline models.

    This class provides comprehensive comparison including:
    - Prediction accuracy
    - Clinical utility
    - Report quality
    - Statistical significance
    """

    def __init__(
        self,
        baseline_models: List[str] = None,
        time_points: List[float] = None
    ):
        """
        Initialize comparative evaluator.

        Args:
            baseline_models: List of baseline model types
            time_points: Time points for evaluation
        """
        self.baseline_models = baseline_models or ["simple", "cox", "deepsurv"]
        self.time_points = time_points or [12, 36, 60]
        self.evaluator = SurvivalEvaluator(time_points=self.time_points)
        self.comparator = ComparisonEvaluator()
        self.patient_generator = PatientGenerator()

    def prepare_baseline_models(
        self,
        df: pd.DataFrame,
        gene_columns: List[str]
    ) -> Dict[str, BaselineModelWrapper]:
        """
        Prepare and train baseline models.

        Args:
            df: Patient DataFrame
            gene_columns: Gene expression columns

        Returns:
            Dictionary of trained models
        """
        X, feature_names = prepare_features(df, gene_columns)
        event = (df["vital_status"] == "Dead").astype(int).values
        time = df["survival_months"].values

        models = {}

        for model_type in self.baseline_models:
            try:
                if model_type == "deepsurv":
                    wrapper = BaselineModelWrapper(
                        model_type=model_type,
                        hidden_sizes=[16, 8],
                        max_epochs=30
                    )
                else:
                    wrapper = BaselineModelWrapper(model_type=model_type)

                wrapper.fit(X, time, event)
                models[model_type] = wrapper
                print(f"  Trained {model_type} model")
            except Exception as e:
                print(f"  Failed to train {model_type}: {e}")

        return models

    def run_comparison(
        self,
        test_df: pd.DataFrame,
        baseline_models: Dict[str, BaselineModelWrapper],
        agent_predictions: List[AgentEvaluationResult] = None,
        agent_risk_scores: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Run comprehensive comparison.

        Args:
            test_df: Test DataFrame
            baseline_models: Trained baseline models
            agent_predictions: Agent predictions (optional)
            agent_risk_scores: Agent risk scores (optional)

        Returns:
            DataFrame with comparison results
        """
        # Prepare features
        gene_cols = [c for c in test_df.columns if c in [
            "HK2", "PKM", "LDHA", "GPI", "PFKL", "GLS", "GLUD1",
            "FASN", "SCD", "CA9", "VEGFA", "MYC"
        ]]
        X_test, _ = prepare_features(test_df, gene_cols)

        times = test_df["survival_months"].values
        events = (test_df["vital_status"] == "Dead").astype(int).values

        results = []

        # Evaluate baseline models
        for model_type, model in baseline_models.items():
            predictions = model.predict_risk(X_test)
            metrics = self.evaluator.evaluate(predictions, times, events)

            results.append({
                "model": model_type.upper(),
                "type": "Baseline",
                "c_index": metrics.c_index,
                "c_index_ci_low": metrics.c_index_ci_low,
                "c_index_ci_high": metrics.c_index_ci_high,
                "auc_1yr": metrics.auc_1yr,
                "auc_3yr": metrics.auc_3yr,
                "brier_score": metrics.brier_score,
                "report_quality": np.nan,
                "clinical_validity": np.nan
            })

        # Evaluate agent (if available)
        if agent_risk_scores is not None:
            metrics = self.evaluator.evaluate(agent_risk_scores, times, events)

            report_qualities = [p.report_quality_score for p in agent_predictions] if agent_predictions else [np.nan]
            clinical_validities = [p.clinical_validity_score for p in agent_predictions] if agent_predictions else [np.nan]

            results.append({
                "model": "LLM Agent",
                "type": "Agent",
                "c_index": metrics.c_index,
                "c_index_ci_low": metrics.c_index_ci_low,
                "c_index_ci_high": metrics.c_index_ci_high,
                "auc_1yr": metrics.auc_1yr,
                "auc_3yr": metrics.auc_3yr,
                "brier_score": metrics.brier_score,
                "report_quality": np.nan if np.isnan(report_qualities[0]) else np.mean(report_qualities),
                "clinical_validity": np.nan if np.isnan(clinical_validities[0]) else np.mean(clinical_validities)
            })

        return pd.DataFrame(results)

    def generate_comparison_report(
        self,
        comparison_df: pd.DataFrame
    ) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison_df: Comparison results DataFrame

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("HCC PROGNOSIS MODEL COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")

        # Sort by C-index
        comparison_df = comparison_df.sort_values("c_index", ascending=False)

        for _, row in comparison_df.iterrows():
            report.append(f"Model: {row['model']} ({row['type']})")
            report.append(f"  C-index: {row['c_index']:.3f} "
                         f"[{row['c_index_ci_low']:.3f}, {row['c_index_ci_high']:.3f}]")
            report.append(f"  AUC 1yr: {row['auc_1yr']:.3f}")
            report.append(f"  AUC 3yr: {row['auc_3yr']:.3f}")
            report.append(f"  Brier Score: {row['brier_score']:.3f}")

            if not np.isnan(row['report_quality']):
                report.append(f"  Report Quality: {row['report_quality']:.3f}")
            if not np.isnan(row['clinical_validity']):
                report.append(f"  Clinical Validity: {row['clinical_validity']:.3f}")

            report.append("")

        # Statistical comparison
        report.append("-" * 70)
        report.append("Key Findings:")
        best_model = comparison_df.iloc[0]['model']
        best_cindex = comparison_df.iloc[0]['c_index']

        if len(comparison_df) > 1:
            second_cindex = comparison_df.iloc[1]['c_index']
            improvement = (best_cindex - second_cindex) / second_cindex * 100
            report.append(f"  Best model: {best_model} (C-index: {best_cindex:.3f})")
            report.append(f"  Improvement over runner-up: {improvement:.1f}%")

        report.append("=" * 70)

        return "\n".join(report)


class ExperimentRunner:
    """
    Runner for comprehensive evaluation experiments.

    This class orchestrates the complete evaluation pipeline:
    1. Data preparation
    2. Model training
    3. Evaluation
    4. Result saving
    """

    def __init__(
        self,
        data_dir: str = "F:/ACM/data",
        output_dir: str = "F:/ACM/experiments"
    ):
        """
        Initialize experiment runner.

        Args:
            data_dir: Data directory
            output_dir: Output directory for results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.patient_generator = PatientGenerator()
        self.comparative_evaluator = ComparativeEvaluator()

    def run_full_experiment(
        self,
        n_train: int = 240,
        n_test: int = 60,
        agent_evaluator: callable = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation experiment.

        Args:
            n_train: Number of training patients
            n_test: Number of test patients
            agent_evaluator: Optional function to evaluate agent on patient
            verbose: Print progress

        Returns:
            Dictionary with all results
        """
        if verbose:
            print("\n" + "=" * 70)
            print("HCC PROGNOSIS MODEL EVALUATION EXPERIMENT")
            print("=" * 70)
            print(f"Training samples: {n_train}")
            print(f"Test samples: {n_test}")
            print("=" * 70)

        # Generate data
        if verbose:
            print("\n[1/5] Generating patient cohort...")

        all_patients, df = self.patient_generator.generate_cohort(
            n_patients=n_train + n_test,
            stratified=True
        )

        # Split train/test
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]

        if verbose:
            print(f"  Training set: {len(train_df)} patients")
            print(f"  Test set: {len(test_df)} patients")

        # Prepare baseline models
        if verbose:
            print("\n[2/5] Training baseline models...")

        gene_cols = [c for c in df.columns if c in [
            "HK2", "PKM", "LDHA", "GPI", "PFKL", "GLS", "GLUD1",
            "FASN", "SCD", "CA9", "VEGFA", "MYC", "HIF1A", "CTNNB1"
        ]]

        baseline_models = self.comparative_evaluator.prepare_baseline_models(
            train_df, gene_cols
        )

        # Run comparison
        if verbose:
            print("\n[3/5] Running model comparison...")

        comparison_df = self.comparative_evaluator.run_comparison(
            test_df, baseline_models
        )

        if verbose:
            print("\n[4/5] Generating comparison report...")
            print(self.comparative_evaluator.generate_comparison_report(comparison_df))

        # Save results
        if verbose:
            print("\n[5/5] Saving results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comparison table
        comparison_df.to_csv(
            self.output_dir / f"model_comparison_{timestamp}.csv",
            index=False
        )

        # Save test predictions
        test_df.to_csv(
            self.output_dir / f"test_patients_{timestamp}.csv",
            index=False
        )

        results = {
            "timestamp": timestamp,
            "n_train": n_train,
            "n_test": n_test,
            "comparison_results": comparison_df,
            "train_data": train_df,
            "test_data": test_df,
            "baseline_models": list(baseline_models.keys())
        }

        if verbose:
            print(f"\nResults saved to: {self.output_dir}")
            print("=" * 70)

        return results


# Convenience functions
def run_agent_vs_baseline_comparison(
    data_dir: str = "F:/ACM/data",
    output_dir: str = "F:/ACM/experiments"
) -> Dict[str, Any]:
    """
    Run agent vs baseline comparison.

    Args:
        data_dir: Data directory
        output_dir: Output directory

    Returns:
        Comparison results
    """
    runner = ExperimentRunner(data_dir=data_dir, output_dir=output_dir)
    return runner.run_full_experiment()


if __name__ == "__main__":
    # Run quick evaluation
    runner = ExperimentRunner()
    results = runner.run_full_experiment(n_train=200, n_test=50)
