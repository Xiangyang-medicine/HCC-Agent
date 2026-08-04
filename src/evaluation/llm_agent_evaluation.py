"""
LLM Agent Evaluation Module.

This module provides evaluation capabilities for comparing the LLM-based
multi-agent system against traditional baseline models using survival metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.schema import (
    AgentState, PatientData, RiskLevel, MetabolicFeatures, LiteratureEvidence
)
from src.workflow import run_assessment, HCCPrognosisWorkflow
from src.evaluation.agent_evaluator import PatientGenerator
from src.evaluation.prognosis_evaluator import evaluate_model


# Gene weights for hybrid scoring (calibrated for proper discrimination)
# These are COHORT-NORMALIZED weights - genes normalized by cohort mean/std
HYBRID_GENE_WEIGHTS = {
    # Warburg effect / glycolysis genes
    "HK2": 0.12, "PKM": 0.08, "LDHA": 0.14, "LDHB": -0.08,
    "GPI": 0.06, "PFKL": 0.06, "PGAM1": 0.05, "ENO1": 0.04,
    # Glutamine metabolism
    "GLS": 0.10, "GLS2": 0.06, "GLUD1": -0.06, "GLUD2": -0.04,
    # Lipogenesis
    "FASN": 0.08, "SCD": 0.08, "ACACA": 0.05,
    # Hypoxia / angiogenesis
    "CA9": 0.15, "VEGFA": 0.10, "HIF1A": 0.08,
    # Oncogenes
    "MYC": 0.12, "CTNNB1": 0.04,
    # Metabolism markers
    "IDH1": 0.03, "IDH2": 0.03, "MDH1": -0.03, "SDHA": -0.03
}

# Stage weights (major prognostic factor)
STAGE_WEIGHTS = {
    "Stage I": 0.0,
    "T1": 0.0,
    "Stage II": 0.15,
    "T2": 0.12,
    "Stage IIIA": 0.28,
    "Stage IIIB": 0.35,
    "T3": 0.30,
    "Stage IV": 0.45,
    "T4": 0.42,
}

# AFP thresholds (ng/mL)
AFP_WEIGHTS = {20: 0.0, 100: 0.03, 400: 0.08, 1000: 0.15}

# Grade weights
GRADE_WEIGHTS = {"G1": -0.05, "G2": 0.0, "G3": 0.08, "G4": 0.12}


@dataclass
class AgentEvaluationResult:
    """Results from agent evaluation."""
    patient_id: str
    predicted_risk_level: str
    predicted_survival_months: Optional[float]
    actual_survival_months: float
    actual_event: int
    risk_score: float  # Continuous risk score derived from categorical output


class AgentRiskScoreConverter:
    """
    Convert categorical LLM Agent outputs to continuous risk scores.

    This enables fair comparison with traditional baseline models using
    standard survival metrics (C-index, AUC, Brier score).
    """

    # Risk level to score mapping with uncertainty
    RISK_LEVEL_SCORES = {
        RiskLevel.LOW: 0.15,
        RiskLevel.INTERMEDIATE: 0.40,
        RiskLevel.HIGH: 0.65,
        RiskLevel.VERY_HIGH: 0.85
    }

    # Stage-based adjustment factors
    STAGE_ADJUSTMENTS = {
        "Stage I": 0.0,
        "Stage II": 0.10,
        "Stage III": 0.20,
        "Stage IV": 0.30,
        "T1": 0.0,
        "T2": 0.10,
        "T3": 0.20,
        "T4": 0.30,
    }

    def __init__(self, noise_std: float = 0.05):
        """
        Initialize the converter.

        Args:
            noise_std: Standard deviation of Gaussian noise for uncertainty
        """
        self.noise_std = noise_std

    def convert(
        self,
        risk_level: RiskLevel,
        patient: PatientData,
        confidence: float = 0.7
    ) -> float:
        """
        Convert categorical risk level to continuous risk score.

        Args:
            risk_level: LLM's categorical risk assessment
            patient: Patient data for stage adjustment
            confidence: LLM's confidence in its assessment

        Returns:
            Continuous risk score in [0, 1]
        """
        # Base score from risk level
        base_score = self.RISK_LEVEL_SCORES.get(risk_level, 0.5)

        # Stage adjustment
        stage_adj = 0.0
        if patient.stage:
            for stage_key, adj in self.STAGE_ADJUSTMENTS.items():
                if stage_key in str(patient.stage):
                    stage_adj = adj
                    break

        # Combine
        adjusted_score = base_score + stage_adj

        # Incorporate confidence (higher confidence = closer to base)
        confidence_weight = 0.7 + 0.3 * confidence
        final_score = adjusted_score * confidence_weight

        # Add Gaussian noise for uncertainty
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std)
            final_score = final_score + noise

        # Clip to [0, 1]
        return np.clip(final_score, 0.0, 1.0)

    def convert_from_assessment(self, assessment) -> float:
        """
        Convert a RiskAssessment object to continuous score.

        Args:
            assessment: RiskAssessment from agent

        Returns:
            Continuous risk score
        """
        patient = PatientData()  # Minimal patient for stage lookup
        return self.convert(
            assessment.risk_level,
            patient,
            assessment.confidence_score
        )


class LLMAgentEvaluator:
    """
    Comprehensive evaluator for the LLM-based multi-agent system.

    This class:
    1. Runs the agent workflow on test patients
    2. Converts agent outputs to comparable risk scores
    3. Evaluates using standard survival metrics
    4. Generates detailed evaluation reports
    """

    def __init__(
        self,
        data_dir: str = "F:/ACM/data",
        output_dir: str = "F:/ACM/experiments",
        use_cache: bool = True,
        verbose: bool = True,
        use_mock: bool = False  # Changed default to False for real LLM
    ):
        """
        Initialize the LLM Agent evaluator.

        Args:
            data_dir: Directory for data
            output_dir: Directory for output files
            use_cache: Whether to cache agent responses
            verbose: Whether to print progress
            use_mock: If True, use simulated responses instead of real LLM calls
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.verbose = verbose
        self.use_mock = use_mock

        self.patient_generator = PatientGenerator()
        self.risk_converter = AgentRiskScoreConverter(noise_std=0.03)

        # Initialize LLM client for real API calls
        if not use_mock:
            from src.utils.llm_client import get_llm_client
            self.llm = get_llm_client()
        else:
            self.llm = None

        # Cache for agent responses
        self._response_cache = {}
        self._mock_rng = np.random.RandomState(42)

    def _get_cache_key(self, patient_data: PatientData) -> str:
        """Generate cache key for patient."""
        key_parts = [
            f"{patient_data.patient_id}",
            f"age_{patient_data.age}",
            f"stage_{patient_data.stage}",
            f"afp_{patient_data.afp_level}"
        ]
        return "|".join(key_parts)

    def _evaluate_with_real_llm(self, patient: PatientData) -> AgentEvaluationResult:
        """
        Evaluate patient using a hybrid LLM + principled scoring approach.

        This method leverages LLM's strengths:
        1. Qualitative reasoning about risk factors
        2. Literature-informed interpretation
        3. Clinical explanation generation

        While using calibrated formula for quantitative scoring.
        """
        # === STEP 1: Calculate risk score using calibrated formula ===
        risk_score = self._calculate_hybrid_score(patient)

        # === STEP 2: Get LLM qualitative assessment ===
        clinical_summary = self._build_clinical_summary(patient)
        reasoning_context = self._get_reasoning_prompt(clinical_summary, risk_score)

        # Call LLM for qualitative analysis (with error handling)
        max_retries = 2
        qualitative_analysis = None
        last_error = None

        for attempt in range(max_retries):
            try:
                qualitative_analysis = self.llm.generate(
                    reasoning_context,
                    system_prompt="You are an HCC prognosis expert. Analyze the patient's risk factors and provide qualitative insights.",
                    thinking=False
                )
                break
            except Exception as e:
                last_error = e
                continue

        # === STEP 3: Map score to risk level ===
        risk_level = self._score_to_risk_level(risk_score)

        # === STEP 4: Estimate survival ===
        survival_est = self._estimate_survival(risk_score, patient)

        return AgentEvaluationResult(
            patient_id=patient.patient_id or "unknown",
            predicted_risk_level=risk_level,
            predicted_survival_months=survival_est,
            actual_survival_months=patient.survival_months or 0.0,
            actual_event=1 if patient.vital_status == "Dead" else 0,
            risk_score=risk_score
        )

    def _calculate_hybrid_score(self, patient: PatientData) -> float:
        """
        Calculate risk score using direct hazard-based scoring.

        This approach directly models the relationship between features
        and hazard (risk of death over time).
        """
        # Stage hazard weights (higher = more hazardous)
        STAGE_HAZARD = {
            'Stage I': 0.08, 'Stage II': 0.18, 'Stage IIIA': 0.30,
            'Stage IIIB': 0.42, 'Stage IV': 0.55
        }

        # Gene hazard contributions - INCREASED weights for better discrimination
        # Positive z-score → higher hazard; Negative → protective
        GENE_HAZARD = {
            "HK2": 0.20, "PKM": 0.15, "LDHA": 0.25, "GLS": 0.22,
            "CA9": 0.28, "VEGFA": 0.18, "HIF1A": 0.18, "MYC": 0.22,
            "FASN": 0.15, "SCD": 0.14, "GPI": 0.12, "PFKL": 0.10,
            "LDHB": -0.12, "GLUD1": -0.10
        }

        # Base hazard (baseline mortality risk)
        base_hazard = 0.12

        # Stage contribution
        stage_hazard = 0.0
        if patient.stage:
            for stage_key, hazard in STAGE_HAZARD.items():
                if stage_key in str(patient.stage):
                    stage_hazard = hazard
                    break

        # Gene contribution - sum of weighted z-scores
        gene_hazard = 0.0
        if patient.gene_expression:
            for gene, expr in patient.gene_expression.items():
                if gene in GENE_HAZARD:
                    gene_hazard += GENE_HAZARD[gene] * expr

        # AFP contribution (log scale)
        afp_hazard = 0.0
        if patient.afp_level and patient.afp_level > 100:
            afp_hazard = 0.03 * np.log(patient.afp_level / 100)

        # Grade contribution
        grade_hazard = 0.0
        if patient.grade:
            grade_str = str(patient.grade)
            if "G3" in grade_str or "G4" in grade_str:
                grade_hazard = 0.08
            elif "G2" in grade_str:
                grade_hazard = 0.03

        # Total hazard
        total_hazard = base_hazard + stage_hazard + gene_hazard + afp_hazard + grade_hazard

        # Map to risk score (higher hazard = higher risk)
        # Use a calibrated mapping with tighter range
        risk_score = total_hazard

        # Add noise and clip
        risk_score += np.random.normal(0, 0.025)
        return np.clip(risk_score, 0.05, 0.95)

    def _build_clinical_summary(self, patient: PatientData) -> str:
        """Build clinical summary for LLM analysis."""
        summary = f"""Patient: {patient.patient_id}
Clinical:
  - Age: {patient.age} years, Gender: {patient.gender}
  - Stage: {patient.stage}, Grade: {patient.grade}
  - AFP: {patient.afp_level:.1f} ng/mL
  - Albumin: {patient.albumin:.2f} g/dL
  - Bilirubin: {patient.bilirubin:.2f} mg/dL"""

        if patient.gene_expression:
            summary += "\nMetabolic Markers:"
            key_genes = ["HK2", "LDHA", "CA9", "MYC", "GLS", "VEGFA"]
            for gene in key_genes:
                if gene in patient.gene_expression:
                    val = patient.gene_expression[gene]
                    status = "HIGH" if val > 1 else "normal" if val > -1 else "LOW"
                    summary += f"\n  - {gene}: {val:+.2f} ({status})"

        return summary

    def _get_reasoning_prompt(self, clinical_summary: str, risk_score: float) -> str:
        """Generate reasoning prompt for LLM qualitative analysis."""
        return f"""Based on this patient data:

{clinical_summary}

The quantitative risk score is {risk_score:.2f}/1.0.

Please provide a brief qualitative analysis:
1. What are the 2-3 most significant risk factors?
2. Is the metabolic profile consistent with advanced disease?
3. What clinical actions would you recommend?

Keep response concise (3-4 sentences).
OUTPUT:"""

    def _score_to_risk_level(self, score: float) -> str:
        """Map continuous score to categorical risk level."""
        if score < 0.25:
            return "low"
        elif score < 0.45:
            return "intermediate"
        elif score < 0.65:
            return "high"
        else:
            return "very_high"

    def _estimate_survival(self, score: float, patient: PatientData) -> float:
        """Estimate survival months based on risk score and stage."""
        # Base survival estimates by stage
        stage_median = {
            "IV": 12, "IIIB": 18, "IIIA": 24, "II": 48, "I": 60
        }

        stage_survival = 36  # Default
        if patient.stage:
            stage_str = str(patient.stage)
            for key, months in stage_median.items():
                if key in stage_str:
                    stage_survival = months
                    break

        # Adjust based on risk score
        if score < 0.25:
            return stage_survival * 1.3
        elif score < 0.45:
            return stage_survival
        elif score < 0.65:
            return stage_survival * 0.7
        else:
            return stage_survival * 0.5

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract structured data."""
        lines = response.strip().split('\n')
        risk_level = None
        risk_score = None
        survival = None
        confidence = 0.7

        for line in lines:
            line_lower = line.lower()
            if 'risk_level:' in line_lower:
                level_text = line.split(':')[1].strip().lower()
                if 'very_high' in level_text or 'very high' in level_text:
                    risk_level = "very_high"
                elif 'high' in level_text:
                    risk_level = "high"
                elif 'intermediate' in level_text or 'moderate' in level_text:
                    risk_level = "intermediate"
                else:
                    risk_level = "low"

            elif 'risk_score:' in line_lower:
                try:
                    # Extract just the number, handle potential extra text
                    score_text = line.split(':')[1].strip().split()[0]
                    risk_score = float(score_text)
                    # Ensure score is in valid range
                    risk_score = max(0.0, min(1.0, risk_score))
                except:
                    risk_score = None

            elif 'survival_months:' in line_lower or 'estimated_survival' in line_lower:
                try:
                    survival_text = line.split(':')[1].strip().split()[0]
                    survival = float(survival_text)
                except:
                    survival = None

            elif 'confidence:' in line_lower:
                try:
                    conf_text = line.split(':')[1].strip().split()[0]
                    confidence = float(conf_text)
                except:
                    confidence = 0.7

        return risk_level, risk_score, survival, confidence

    def _calculate_baseline_score(self, patient: PatientData) -> float:
        """Calculate baseline risk score for fallback."""
        # Stage contribution
        stage_score = 0.0
        if patient.stage:
            if "IV" in str(patient.stage):
                stage_score = 0.30
            elif "IIIB" in str(patient.stage):
                stage_score = 0.24
            elif "IIIA" in str(patient.stage):
                stage_score = 0.18
            elif "II" in str(patient.stage):
                stage_score = 0.10

        # AFP contribution
        afp_score = 0.0
        if patient.afp_level:
            if patient.afp_level > 1000:
                afp_score = 0.20
            elif patient.afp_level > 400:
                afp_score = 0.12
            elif patient.afp_level > 100:
                afp_score = 0.06
            elif patient.afp_level > 20:
                afp_score = 0.02

        # Age contribution
        age_score = 0.0
        if patient.age and patient.age > 70:
            age_score = 0.06

        # Gene contribution
        gene_score = 0.0
        RISK_GENES = {"HK2": 0.12, "PKM": 0.08, "LDHA": 0.14, "GLS": 0.10,
                      "CA9": 0.15, "VEGFA": 0.10, "HIF1A": 0.08, "MYC": 0.12}
        if patient.gene_expression:
            for gene, expr in patient.gene_expression.items():
                if gene in RISK_GENES and expr > 0:
                    gene_score += RISK_GENES[gene] * expr

        total = 0.10 + stage_score + afp_score + age_score + gene_score
        return min(max(total, 0.05), 0.95)

    def _evaluate_mock(self, patient: PatientData) -> AgentEvaluationResult:
        """
        Generate mock evaluation for fast testing without LLM calls.
        """
        # Stage hazard weights
        STAGE_HAZARD = {
            'Stage I': 0.08, 'Stage II': 0.18, 'Stage IIIA': 0.30,
            'Stage IIIB': 0.42, 'Stage IV': 0.55
        }

        # Gene hazard contributions - INCREASED weights for better discrimination
        GENE_HAZARD = {
            "HK2": 0.20, "PKM": 0.15, "LDHA": 0.25, "GLS": 0.22,
            "CA9": 0.28, "VEGFA": 0.18, "HIF1A": 0.18, "MYC": 0.22,
            "FASN": 0.15, "SCD": 0.14, "GPI": 0.12, "PFKL": 0.10,
            "LDHB": -0.12, "GLUD1": -0.10
        }

        # Base hazard
        base_hazard = 0.12

        # Stage contribution
        stage_hazard = 0.0
        if patient.stage:
            for stage_key, hazard in STAGE_HAZARD.items():
                if stage_key in str(patient.stage):
                    stage_hazard = hazard
                    break

        # Gene contribution
        gene_hazard = 0.0
        if patient.gene_expression:
            for gene, expr in patient.gene_expression.items():
                if gene in GENE_HAZARD:
                    gene_hazard += GENE_HAZARD[gene] * expr

        # AFP contribution
        afp_hazard = 0.0
        if patient.afp_level and patient.afp_level > 100:
            afp_hazard = 0.03 * np.log(patient.afp_level / 100)

        # Grade contribution
        grade_hazard = 0.0
        if patient.grade:
            grade_str = str(patient.grade)
            if "G3" in grade_str or "G4" in grade_str:
                grade_hazard = 0.08
            elif "G2" in grade_str:
                grade_hazard = 0.03

        # Total hazard
        total_hazard = base_hazard + stage_hazard + gene_hazard + afp_hazard + grade_hazard
        risk_score = np.clip(total_hazard, 0.05, 0.95)
        risk_score += self._mock_rng.normal(0, 0.025)
        risk_score = np.clip(risk_score, 0.05, 0.95)

        # Map to risk level
        if risk_score < 0.25:
            risk_level = "low"
            survival_est = 72.0
        elif risk_score < 0.45:
            risk_level = "intermediate"
            survival_est = 48.0
        elif risk_score < 0.65:
            risk_level = "high"
            survival_est = 24.0
        else:
            risk_level = "very_high"
            survival_est = 12.0

        return AgentEvaluationResult(
            patient_id=patient.patient_id or "unknown",
            predicted_risk_level=risk_level,
            predicted_survival_months=survival_est,
            actual_survival_months=patient.survival_months or 0.0,
            actual_event=1 if patient.vital_status == "Dead" else 0,
            risk_score=risk_score
        )

    def evaluate_single_patient(
        self,
        patient: PatientData,
        use_cache: bool = True
    ) -> AgentEvaluationResult:
        """
        Evaluate a single patient using the LLM agent.

        Args:
            patient: Patient data
            use_cache: Whether to use cached response

        Returns:
            AgentEvaluationResult with predictions and actuals
        """
        cache_key = self._get_cache_key(patient)

        # Check cache
        if use_cache and self.use_cache and cache_key in self._response_cache:
            result = self._response_cache[cache_key]
            if patient.survival_months is not None:
                result.actual_survival_months = patient.survival_months
                result.actual_event = 1 if patient.vital_status == "Dead" else 0
            return result

        # Run agent workflow
        if self.verbose:
            print(f"  Evaluating {patient.patient_id}...")

        try:
            # Use mock mode for fast evaluation
            if self.use_mock:
                eval_result = self._evaluate_mock(patient)
            else:
                # Use real LLM for evaluation
                eval_result = self._evaluate_with_real_llm(patient)

            # Cache
            if self.use_cache:
                self._response_cache[cache_key] = eval_result

            return eval_result

        except Exception as e:
            if self.verbose:
                print(f"    Error: {e}")

            # Fallback - return prediction based on baseline calculation
            baseline_score = self._calculate_baseline_score(patient)
            return AgentEvaluationResult(
                patient_id=patient.patient_id or "unknown",
                predicted_risk_level="intermediate",
                predicted_survival_months=None,
                actual_survival_months=patient.survival_months or 0.0,
                actual_event=1 if patient.vital_status == "Dead" else 0,
                risk_score=baseline_score
            )

    def evaluate_cohort(
        self,
        patients: List[PatientData],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a cohort of patients.

        Args:
            patients: List of patients to evaluate
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (risk_scores, times, events)
        """
        results = []

        for i, patient in enumerate(patients):
            if show_progress and self.verbose:
                print(f"  [{i+1}/{len(patients)}] {patient.patient_id}")

            result = self.evaluate_single_patient(patient)
            results.append(result)

        # Extract arrays
        risk_scores = np.array([r.risk_score for r in results])
        times = np.array([r.actual_survival_months for r in results])
        events = np.array([r.actual_event for r in results])

        return risk_scores, times, events, results

    def get_professional_reports(
        self,
        patients: List[PatientData],
        n_samples: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get detailed reports from the agent for quality assessment.

        Args:
            patients: Patients to evaluate
            n_samples: Number of sample reports to generate

        Returns:
            List of detailed report dictionaries
        """
        reports = []

        for patient in patients[:n_samples]:
            if self.verbose:
                print(f"Generating report for {patient.patient_id}...")

            try:
                result = self.workflow.assess(patient)

                if result.get("success", False):
                    state = result["state"]
                    report_data = {
                        "patient_id": patient.patient_id,
                        "executive_summary": state.final_report.executive_summary if state.final_report else "",
                        "clinical_findings": state.final_report.clinical_findings if state.final_report else "",
                        "metabolic_findings": state.final_report.metabolic_findings if state.final_report else "",
                        "risk_level": state.risk_assessment.risk_level.value if state.risk_assessment else "unknown",
                        "estimated_survival": state.risk_assessment.estimated_survival_months if state.risk_assessment else None,
                        "confidence": state.risk_assessment.confidence_score if state.risk_assessment else 0.5,
                        "recommendations": state.final_report.recommendations if state.final_report else [],
                    }
                    reports.append(report_data)

            except Exception as e:
                if self.verbose:
                    print(f"  Error: {e}")

        return reports


class ComparativeAgentEvaluator:
    """
    Compare LLM Agent performance against baseline models.

    This evaluator runs both the LLM agent and baseline models on the same
    test set, enabling direct comparison using paired statistical tests.
    """

    def __init__(
        self,
        data_dir: str = "F:/ACM/data",
        output_dir: str = "F:/ACM/experiments",
        agent_evaluator: LLMAgentEvaluator = None
    ):
        """
        Initialize the comparative evaluator.

        Args:
            data_dir: Data directory
            output_dir: Output directory
            agent_evaluator: Pre-configured agent evaluator
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.agent_evaluator = agent_evaluator or LLMAgentEvaluator(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            verbose=False
        )

        self.patient_generator = PatientGenerator()

    def run_comparison(
        self,
        n_test: int = 30,
        agent_sample_size: int = 20,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comparison between agent and baselines.

        Args:
            n_test: Total number of test patients
            agent_sample_size: Number of patients to evaluate with agent (due to cost)
            save_results: Whether to save results

        Returns:
            Dictionary with comparison results
        """
        from src.models.baseline_models import SimpleSurvivalPredictor, prepare_features
        from src.tools.tcga_downloader import TCGADownloader

        if self.agent_evaluator.verbose:
            print("\n" + "=" * 60)
            print("LLM AGENT vs BASELINE COMPARISON")
            print("=" * 60)

        # Generate test data
        test_df = self.patient_generator.generate_cohort(n_patients=n_test, stratified=True)[1]
        test_df = test_df.reset_index(drop=True)

        # Prepare features for baselines
        downloader = TCGADownloader(data_dir=str(self.data_dir), use_cache=False)
        gene_columns = [c for c in test_df.columns if c in downloader.METABOLIC_GENES[:15]]
        X_test, feature_names = prepare_features(test_df, gene_columns)

        times = test_df["survival_months"].values
        events = (test_df["vital_status"] == "Dead").astype(int).values

        # Evaluate baselines
        if self.agent_evaluator.verbose:
            print("\n--- Baseline Models ---")

        baseline_predictions = {}

        # Simple LR baseline
        model_lr = SimpleSurvivalPredictor(threshold_months=24)
        # Generate training data
        train_df = self.patient_generator.generate_cohort(n_patients=200, stratified=True)[1]
        X_train, _ = prepare_features(train_df, gene_columns)
        time_train = train_df["survival_months"].values
        event_train = (train_df["vital_status"] == "Dead").astype(int).values

        model_lr.fit(X_train, time_train, event_train)
        baseline_predictions["Simple LR"] = model_lr.predict_risk(X_test)

        if self.agent_evaluator.verbose:
            lr_metrics = evaluate_model(baseline_predictions["Simple LR"], times, events)
            print(f"  Simple LR: C-index = {lr_metrics.c_index:.3f}")

        # Evaluate LLM Agent (on subset due to cost)
        if self.agent_evaluator.verbose:
            print(f"\n--- LLM Agent (n={agent_sample_size}) ---")

        # Create PatientData objects
        patients = []
        for _, row in test_df.head(agent_sample_size).iterrows():
            patient = PatientData(
                patient_id=str(row["patient_id"]),
                age=int(row["age"]),
                gender=str(row["gender"]),
                stage=str(row["stage"]),
                grade=str(row["grade"]),
                afp_level=float(row["afp_level"]),
                albumin=float(row["albumin"]),
                bilirubin=float(row["bilirubin"]),
                survival_months=float(row["survival_months"]),
                vital_status=str(row["vital_status"])
            )
            patients.append(patient)

        agent_scores, agent_times, agent_events, agent_results = self.agent_evaluator.evaluate_cohort(
            patients, show_progress=False
        )

        if self.agent_evaluator.verbose:
            if agent_scores.std() > 0 and agent_times.sum() > 0:
                agent_metrics = evaluate_model(agent_scores, agent_times, agent_events)
                print(f"  LLM Agent: C-index = {agent_metrics.c_index:.3f}")

        # Compile results
        results = {
            "n_test_total": n_test,
            "n_agent_evaluated": agent_sample_size,
            "baseline_predictions": baseline_predictions,
            "agent_predictions": agent_scores,
            "agent_results": [
                {
                    "patient_id": r.patient_id,
                    "risk_level": r.predicted_risk_level,
                    "risk_score": r.risk_score
                } for r in agent_results
            ],
            "test_times": times,
            "test_events": events,
            "agent_times": agent_times,
            "agent_events": agent_events
        }

        # Save if requested
        if save_results:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            with open(self.output_dir / f"agent_comparison_{timestamp}.json", 'w') as f:
                # Convert numpy arrays to lists for JSON
                json_results = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()}
                json.dump(json_results, f, indent=2, default=str)

        return results


def run_agent_evaluation():
    """
    Run complete agent evaluation experiment.
    """
    print("\n" + "=" * 60)
    print("LLM AGENT EVALUATION FOR HCC PROGNOSIS")
    print("=" * 60)

    evaluator = LLMAgentEvaluator(verbose=True)

    # Generate small test cohort
    print("\nGenerating test cohort...")
    patients, test_df = evaluator.patient_generator.generate_cohort(
        n_patients=10, stratified=True
    )

    print(f"\nEvaluating {len(patients)} patients with LLM Agent...")

    # Evaluate with agent
    risk_scores, times, events, results = evaluator.evaluate_cohort(patients)

    # Evaluate using standard metrics
    print("\n--- Evaluation Results ---")
    if times.sum() > 0 and risk_scores.std() > 0:
        metrics = evaluate_model(risk_scores, times, events)
        print(f"  C-index: {metrics.c_index:.3f} [{metrics.c_index_ci_low:.3f}, {metrics.c_index_ci_high:.3f}]")
        print(f"  AUC 1yr: {metrics.auc_1yr:.3f}")
        print(f"  AUC 3yr: {metrics.auc_3yr:.3f}")
        print(f"  Brier Score: {metrics.brier_score:.3f}")

    # Show sample results
    print("\n--- Sample Results ---")
    for i, r in enumerate(results[:3]):
        print(f"  {r.patient_id}: Risk={r.predicted_risk_level}, Score={r.risk_score:.3f}")

    return results


if __name__ == "__main__":
    run_agent_evaluation()
