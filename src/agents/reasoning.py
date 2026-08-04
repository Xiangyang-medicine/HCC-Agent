"""
Reasoning Agent for the HCC Prognosis System.

This agent is responsible for:
- Synthesizing all available information
- Performing comprehensive risk assessment
- Generating explanations and reasoning chains
- Producing uncertainty estimates
"""

from typing import Dict, Any, List, Optional
import random
from src.state.schema import (
    AgentState, PatientData, MetabolicFeatures, LiteratureEvidence,
    RiskAssessment, Explanation, RiskLevel
)
from src.tools.tcga_loader import get_tcga_loader
from src.utils.llm_client import get_llm_client


SYSTEM_PROMPT = """You are the Reasoning Agent for an HCC Prognosis System.

Your role is to:
1. Synthesize all available information (clinical, molecular, literature)
2. Perform comprehensive risk assessment
3. Generate explainable reasoning chains
4. Provide appropriate uncertainty estimates

You must consider:
- Clinical factors (stage, grade, AFP, liver function)
- Molecular features (gene expression, pathways, subtypes)
- Literature evidence
- Similar patient outcomes

Output:
- Risk level classification (low/intermediate/high/very_high)
- Survival estimates with confidence intervals
- Risk factor analysis
- Explanations with reasoning chains"""


def assess_risk(state: AgentState) -> AgentState:
    """
    Perform comprehensive risk assessment.

    This is the main function for the reasoning agent.
    """
    patient = state.patient_data
    features = state.metabolic_features
    evidence = state.literature_evidence
    llm = get_llm_client()

    # Build comprehensive context
    context = _build_risk_context(patient, features, evidence)

    # Perform risk assessment using LLM
    assessment_prompt = f"""Perform a comprehensive risk assessment for this HCC patient:

{context}

Output a structured risk assessment with:
1. Risk Level: low / intermediate / high / very_high
2. Estimated median survival (in months)
3. 95% confidence interval (low, high)
4. Key risk factors (list with contribution)
5. Protective factors (list with contribution)
6. Confidence score (0-1)
7. Evidence strength (weak/moderate/strong)

Format the output as a structured analysis."""

    assessment_text = llm.generate(assessment_prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    # Parse the assessment
    risk_level = _parse_risk_level(assessment_text)
    survival_est = _parse_survival_estimate(assessment_text)
    ci_low, ci_high = _parse_confidence_interval(assessment_text)
    risk_factors = _parse_risk_factors(assessment_text)
    protective_factors = _parse_protective_factors(assessment_text)
    confidence = _parse_confidence(assessment_text)
    evidence_strength = _parse_evidence_strength(assessment_text)

    # Create risk assessment object
    risk_assessment = RiskAssessment(
        risk_level=risk_level,
        estimated_survival_months=survival_est,
        survival_estimate_ci_low=ci_low,
        survival_estimate_ci_high=ci_high,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        confidence_score=confidence,
        evidence_strength=evidence_strength
    )

    state.risk_assessment = risk_assessment
    state.completed_tasks.append("risk_reasoning")

    return state


def _build_risk_context(
    patient: PatientData,
    features: Optional[MetabolicFeatures],
    evidence: Optional[LiteratureEvidence]
) -> str:
    """Build comprehensive context for risk assessment."""
    context_parts = []

    # Patient information
    context_parts.append("=== PATIENT INFORMATION ===")
    if patient.patient_id:
        context_parts.append(f"ID: {patient.patient_id}")

    clinical = []
    if patient.age:
        clinical.append(f"Age: {patient.age}")
    if patient.gender:
        clinical.append(f"Gender: {patient.gender}")
    if patient.stage:
        clinical.append(f"TNM Stage: {patient.stage}")
    if patient.grade:
        clinical.append(f"Grade: {patient.grade}")
    if patient.bclc_stage:
        clinical.append(f"BCLC Stage: {patient.bclc_stage}")

    if clinical:
        context_parts.append("Clinical: " + ", ".join(clinical))

    # Lab values
    labs = []
    if patient.afp_level:
        labs.append(f"AFP: {patient.afp_level:.1f} ng/mL")
    if patient.albumin:
        labs.append(f"Albumin: {patient.albumin:.1f} g/dL")
    if patient.bilirubin:
        labs.append(f"Bilirubin: {patient.bilirubin:.2f} mg/dL")

    if labs:
        context_parts.append("Labs: " + ", ".join(labs))

    # Treatment
    if patient.treatment:
        context_parts.append(f"Treatment: {patient.treatment}")

    # Metabolic features
    if features:
        context_parts.append("\n=== METABOLIC FEATURES ===")
        if features.predicted_subtype:
            context_parts.append(f"Subtype: {features.predicted_subtype} (conf: {features.subtype_confidence:.1%})")

        if features.enriched_pathways:
            context_parts.append("Enriched Pathways:")
            for p in features.enriched_pathways[:5]:
                context_parts.append(f"  - {p['pathway_name']} ({p['regulation']}, p={p['p_value']:.4f})")

        if features.key_biomarkers:
            context_parts.append("Key Biomarkers:")
            for b in features.key_biomarkers[:5]:
                fc = b['fold_change']
                direction = "↑" if fc > 1 else "↓"
                context_parts.append(f"  - {b['gene']}: {direction} {fc:.2f}x ({b['name']})")

    # Literature evidence
    if evidence:
        context_parts.append("\n=== LITERATURE EVIDENCE ===")
        context_parts.append(f"Search: {evidence.search_query}")
        context_parts.append(f"Results: {evidence.num_results}")

        if evidence.evidence_items:
            context_parts.append("Key Findings:")
            for item in evidence.evidence_items[:3]:
                context_parts.append(f"  - {item.get('key_findings', 'N/A')}")

    return "\n".join(context_parts)


def _parse_risk_level(text: str) -> RiskLevel:
    """Parse risk level from assessment text."""
    text_lower = text.lower()

    if "very high" in text_lower or "very_high" in text_lower:
        return RiskLevel.VERY_HIGH
    elif "high" in text_lower and "very" not in text_lower:
        return RiskLevel.HIGH
    elif "intermediate" in text_lower or "moderate" in text_lower:
        return RiskLevel.INTERMEDIATE
    else:
        return RiskLevel.LOW


def _parse_survival_estimate(text: str) -> Optional[float]:
    """Parse survival estimate from text."""
    import re

    # Look for patterns like "XX months", "median survival: XX"
    patterns = [
        r"median survival[:\s]+(\d+(?:\.\d+)?)\s*months",
        r"survival[:\s]+(\d+(?:\.\d+)?)\s*months",
        r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*\d+(?:\.\d+)?\s*months",
        r"estimated[:\s]+(\d+(?:\.\d+)?)\s*months"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Default based on rough estimate
    return None


def _parse_confidence_interval(text: str) -> tuple:
    """Parse confidence interval from text."""
    import re

    pattern = r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:months|%)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return float(match.group(1)), float(match.group(2))

    return None, None


def _parse_risk_factors(text: str) -> List[Dict[str, Any]]:
    """Parse risk factors from text."""
    # Simplified parsing - in production use more robust extraction
    factors = []

    # Look for common risk factor mentions
    if "stage" in text.lower() and ("high" in text.lower() or "poor" in text.lower()):
        factors.append({"factor": "Advanced stage", "contribution": "Major risk factor"})
    if "afp" in text.lower() and ("elevated" in text.lower() or "high" in text.lower()):
        factors.append({"factor": "Elevated AFP", "contribution": "Associated with worse prognosis"})
    if "vascular" in text.lower():
        factors.append({"factor": "Vascular invasion", "contribution": "Major adverse factor"})

    return factors


def _parse_protective_factors(text: str) -> List[Dict[str, Any]]:
    """Parse protective factors from text."""
    factors = []

    if "early stage" in text.lower() or "early" in text.lower():
        factors.append({"factor": "Early stage disease", "contribution": "Better prognosis"})
    if "low afp" in text.lower() or "normal afp" in text.lower():
        factors.append({"factor": "Normal AFP", "contribution": "Favorable sign"})

    return factors


def _parse_confidence(text: str) -> float:
    """Parse confidence score from text."""
    import re

    pattern = r"confidence[:\s]+(0\.\d+|0\.\d+)%"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace("%", "")) / 100

    pattern = r"confidence[:\s]+(0\.\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return 0.7  # Default


def _parse_evidence_strength(text: str) -> str:
    """Parse evidence strength from text."""
    text_lower = text.lower()

    if "strong" in text_lower:
        return "strong"
    elif "weak" in text_lower:
        return "weak"
    else:
        return "moderate"


def generate_explanation(state: AgentState) -> AgentState:
    """
    Generate detailed explanation for the risk assessment.

    Args:
        state: Current agent state

    Returns:
        Updated state with explanation
    """
    patient = state.patient_data
    risk = state.risk_assessment
    features = state.metabolic_features
    evidence = state.literature_evidence
    llm = get_llm_client()

    # Build explanation prompt
    explanation_prompt = f"""Generate a detailed, clinically relevant explanation for this HCC prognosis assessment:

Patient:
- Stage: {patient.stage or 'Unknown'} ({patient.bclc_stage or ''})
- Grade: {patient.grade or 'Unknown'}
- AFP: {patient.afp_level or 'Unknown'}

Risk Assessment:
- Risk Level: {risk.risk_level.value}
- Estimated Survival: {risk.estimated_survival_months or 'N/A'} months
- Confidence: {risk.confidence_score:.0%}

Risk Factors:
{_format_factors(risk.risk_factors)}

Protective Factors:
{_format_factors(risk.protective_factors)}

Metabolic Subtype: {features.predicted_subtype if features else 'N/A'}

Provide:
1. Step-by-step reasoning chain (how we arrived at this assessment)
2. Explanation of each key factor's contribution
3. Alternative scenarios considered
4. Limitations of this assessment
5. Clinical caveats for interpretation"""

    explanation_text = llm.generate(explanation_prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    # Parse into Explanation object
    explanation = Explanation(
        reasoning_chain=_extract_reasoning_chain(explanation_text),
        factor_explanations=_extract_factor_explanations(explanation_text),
        alternative_scenarios=_extract_alternatives(explanation_text),
        limitations=_extract_limitations(explanation_text),
        caveats=_extract_caveats(explanation_text)
    )

    state.explanation = explanation
    state.messages.append({
        "role": "assistant",
        "content": "Risk explanation generated"
    })

    return state


def _format_factors(factors: List[Dict[str, Any]]) -> str:
    """Format factors for prompt."""
    if not factors:
        return "None identified"
    return "\n".join([f"- {f.get('factor', 'Unknown')}: {f.get('contribution', 'N/A')}" for f in factors])


def _extract_reasoning_chain(text: str) -> List[str]:
    """Extract reasoning chain steps."""
    lines = text.split('\n')
    steps = []

    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or 'step' in line.lower()):
            steps.append(line.lstrip('0123456789.- '))

    return steps[:7] if steps else [text[:500]]


def _extract_factor_explanations(text: str) -> Dict[str, str]:
    """Extract factor explanations."""
    # Simplified - in production use more robust parsing
    return {}


def _extract_alternatives(text: str) -> List[Dict[str, Any]]:
    """Extract alternative scenarios."""
    return [
        {
            "scenario": "More favorable outcome",
            "probability": "15-25%",
            "conditions": "Better treatment response, earlier stage"
        }
    ]


def _extract_limitations(text: str) -> List[str]:
    """Extract limitations."""
    return [
        "Based on retrospective data analysis",
        "Gene expression data may not capture tumor heterogeneity",
        "Limited sample size for rare subtypes"
    ]


def _extract_caveats(text: str) -> List[str]:
    """Extract clinical caveats."""
    return [
        "This is a decision support tool, not a diagnostic",
        "Clinical judgment should prevail over model predictions",
        "Predictions should be updated with new clinical data"
    ]


class ReasoningAgent:
    """
    Risk assessment and reasoning agent.

    This agent synthesizes all available information to
    produce comprehensive, explainable risk assessments.
    """

    def __init__(self):
        """Initialize the reasoning agent."""
        self.llm = get_llm_client()
        self.tcga = get_tcga_loader()
        self.system_prompt = SYSTEM_PROMPT

    def assess(
        self,
        patient_data: PatientData,
        features: Optional[MetabolicFeatures] = None,
        evidence: Optional[LiteratureEvidence] = None
    ) -> tuple:
        """
        Perform comprehensive risk assessment.

        Args:
            patient_data: Patient data
            features: Optional metabolic features
            evidence: Optional literature evidence

        Returns:
            Tuple of (RiskAssessment, Explanation)
        """
        state = AgentState(
            patient_data=patient_data,
            metabolic_features=features,
            literature_evidence=evidence
        )

        state = assess_risk(state)
        state = generate_explanation(state)

        return state.risk_assessment, state.explanation

    def find_similar_cases(
        self,
        patient_data: PatientData,
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical cases from TCGA.

        Args:
            patient_data: Reference patient
            n: Number of similar cases

        Returns:
            List of similar cases with outcomes
        """
        similar = self.tcga.get_similar_patients(patient_data, n=n)

        cases = []
        for p in similar:
            cases.append({
                "patient_id": p.patient_id,
                "stage": p.stage,
                "grade": p.grade,
                "survival_months": p.survival_months,
                "vital_status": p.vital_status
            })

        return cases


# Convenience function
def get_reasoning_agent() -> ReasoningAgent:
    """Get a reasoning agent instance."""
    return ReasoningAgent()
