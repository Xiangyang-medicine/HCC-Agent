"""
Report Generation Agent for the HCC Prognosis System.

This agent is responsible for:
- Generating comprehensive prognosis reports
- Formatting results for clinical use
- Including appropriate disclaimers
"""

from typing import Dict, Any
from datetime import datetime
from src.state.schema import (
    AgentState, PatientData, MetabolicFeatures, LiteratureEvidence,
    RiskAssessment, Explanation, FinalReport
)
from src.utils.llm_client import get_llm_client


SYSTEM_PROMPT = """You are the Report Generation Agent for an HCC Prognosis System.

Your role is to:
1. Generate comprehensive, clinically relevant prognosis reports
2. Present complex analyses in accessible formats
3. Include appropriate context and limitations

You produce reports that:
- Are suitable for healthcare professionals
- Include appropriate uncertainty quantification
- Have clear structure and formatting
- Include necessary clinical disclaimers"""


def generate_report(state: AgentState) -> AgentState:
    """
    Generate the final prognosis report.

    This is the final step in the workflow, producing
    a comprehensive report from all available data.
    """
    patient = state.patient_data
    features = state.metabolic_features
    evidence = state.literature_evidence
    risk = state.risk_assessment
    explanation = state.explanation
    llm = get_llm_client()

    # Generate report sections
    executive_summary = _generate_executive_summary(
        patient, features, risk, llm
    )

    clinical_findings = _generate_clinical_findings(
        patient, features, llm
    )

    metabolic_findings = _generate_metabolic_findings(
        features, llm
    )

    literature_support = _generate_literature_support(
        evidence, llm
    )

    recommendations = _generate_recommendations(
        patient, risk, features, llm
    )

    # Create final report
    report = FinalReport(
        patient_id=patient.patient_id,
        generated_at=datetime.now(),
        executive_summary=executive_summary,
        clinical_findings=clinical_findings,
        metabolic_findings=metabolic_findings,
        literature_support=literature_support,
        recommendations=recommendations,
        risk_assessment=risk or RiskAssessment(),
        explanation=explanation or Explanation()
    )

    state.final_report = report
    state.completed_tasks.append("report_generation")

    state.messages.append({
        "role": "assistant",
        "content": "Final report generated successfully"
    })

    return state


def _generate_executive_summary(
    patient: PatientData,
    features,
    risk: RiskAssessment,
    llm
) -> str:
    """Generate executive summary section."""
    prompt = f"""Generate a concise executive summary for an HCC prognosis assessment report.

Patient Overview:
- ID: {patient.patient_id or 'Unknown'}
- Age: {patient.age or 'Unknown'}, Gender: {patient.gender or 'Unknown'}
- Stage: {patient.stage or 'Unknown'} ({patient.bclc_stage or ''})
- Grade: {patient.grade or 'Unknown'}

Risk Assessment:
- Risk Level: {risk.risk_level.value if risk else 'Unknown'}
- Estimated Survival: {risk.estimated_survival_months if risk and risk.estimated_survival_months else 'N/A'} months
- Confidence: {risk.confidence_score if risk else 0:.0%}

Metabolic Subtype: {features.predicted_subtype if features else 'N/A'}

Write a 2-3 sentence executive summary suitable for busy clinicians.
Focus on key findings and risk level."""

    return llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)


def _generate_clinical_findings(
    patient: PatientData,
    features,
    llm
) -> str:
    """Generate clinical findings section."""
    prompt = f"""Generate a clinical findings section for an HCC prognosis report.

Patient Clinical Data:
- Stage: {patient.stage or 'Unknown'} ({patient.bclc_stage or ''})
- Grade: {patient.grade or 'Unknown'}
- AFP: {patient.afp_level if patient.afp_level else 'N/A'} ng/mL
- Albumin: {patient.albumin if patient.albumin else 'N/A'} g/dL
- Bilirubin: {patient.bilirubin if patient.bilirubin else 'N/A'} mg/dL
- Treatment: {patient.treatment or 'Not specified'}

Key Clinical Considerations:
- Interpret AFP in context of tumor burden
- Liver function reflected in albumin/bilirubin
- Stage and grade inform baseline prognosis

Write a concise clinical findings summary (3-4 sentences)."""

    return llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)


def _generate_metabolic_findings(features, llm) -> str:
    """Generate metabolic findings section."""
    if not features:
        return "No metabolic analysis available due to missing gene expression data."

    pathways = features.enriched_pathways[:5] if features.enriched_pathways else []
    biomarkers = features.key_biomarkers[:5] if features.key_biomarkers else []

    pathways_text = "\n".join([f"- {p['pathway_name']} ({p['regulation']})" for p in pathways])
    biomarkers_text = "\n".join([f"- {b['gene']}: {b['fold_change']:.2f}x ({b['name']})" for b in biomarkers])

    prompt = f"""Generate a metabolic findings section for an HCC prognosis report.

Metabolic Subtype: {features.predicted_subtype or 'Unknown'} (confidence: {features.subtype_confidence if features.subtype_confidence else 0:.0%})

Enriched Pathways:
{pathways_text if pathways_text else 'No significantly enriched pathways'}

Key Biomarkers:
{biomarkers_text if biomarkers_text else 'No significant biomarkers identified'}

Write a 3-4 sentence summary of metabolic findings with clinical implications."""

    return llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)


def _generate_literature_support(evidence, llm) -> str:
    """Generate literature support section."""
    if not evidence or not evidence.evidence_items:
        return "Limited literature evidence available for this case."

    items = evidence.evidence_items[:3]
    items_text = "\n".join([
        f"- {item.get('title', 'Unknown')} ({item.get('year', 'N/A')}): {item.get('key_findings', 'N/A')}"
        for item in items
    ])

    prompt = f"""Generate a literature support section for an HCC prognosis report.

Relevant Evidence:
{items_text}

Write a 2-3 sentence summary connecting literature findings to this patient's assessment."""

    return llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)


def _generate_recommendations(
    patient: PatientData,
    risk: RiskAssessment,
    features,
    llm
) -> str:
    """Generate clinical recommendations section."""
    risk_factors = risk.risk_factors if risk else []
    risk_text = "\n".join([f"- {f.get('factor', 'Unknown')}" for f in risk_factors]) if risk_factors else "See risk assessment"

    prompt = f"""Generate clinical recommendations for an HCC patient with the following profile:

Risk Level: {risk.risk_level.value if risk else 'Unknown'}
Key Risk Factors:
{risk_text}

Metabolic Subtype: {features.predicted_subtype if features else 'Unknown'}

Generate 3-5 actionable recommendations suitable for a multidisciplinary tumor board.
Consider: surveillance, treatment planning, and follow-up strategies.
Keep recommendations concise and evidence-based."""

    recommendations_text = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    # Parse recommendations into list
    recommendations = [
        line.strip().lstrip('0123456789.- ')
        for line in recommendations_text.split('\n')
        if line.strip() and (line[0].isdigit() or line.startswith('-'))
    ]

    if not recommendations:
        recommendations = [recommendations_text]

    return recommendations


def format_report_text(report: FinalReport) -> str:
    """
    Format the final report as readable text.

    Args:
        report: FinalReport object

    Returns:
        Formatted report text
    """
    lines = [
        "=" * 60,
        "HCC PROGNOSIS ASSESSMENT REPORT",
        "=" * 60,
        "",
        f"Patient ID: {report.patient_id or 'N/A'}",
        f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        "EXECUTIVE SUMMARY",
        "-" * 60,
        report.executive_summary or "N/A",
        "",
    ]

    # Risk assessment
    if report.risk_assessment:
        risk = report.risk_assessment
        lines.extend([
            "-" * 60,
            "RISK ASSESSMENT",
            "-" * 60,
            f"Risk Level: {risk.risk_level.value.upper()}",
            f"Confidence: {risk.confidence_score:.0%}",
            "",
        ])

        if risk.estimated_survival_months:
            lines.append(f"Estimated Median Survival: {risk.estimated_survival_months:.1f} months")
            if risk.survival_estimate_ci_low and risk.survival_estimate_ci_high:
                lines.append(f"95% CI: {risk.survival_estimate_ci_low:.1f} - {risk.survival_estimate_ci_high:.1f} months")
            lines.append("")

        if risk.risk_factors:
            lines.append("Risk Factors:")
            for f in risk.risk_factors:
                lines.append(f"  - {f.get('factor', 'Unknown')}: {f.get('contribution', 'N/A')}")
            lines.append("")

    # Clinical findings
    if report.clinical_findings:
        lines.extend([
            "-" * 60,
            "CLINICAL FINDINGS",
            "-" * 60,
            report.clinical_findings,
            "",
        ])

    # Metabolic findings
    if report.metabolic_findings:
        lines.extend([
            "-" * 60,
            "METABOLIC FINDINGS",
            "-" * 60,
            report.metabolic_findings,
            "",
        ])

    # Literature support
    if report.literature_support:
        lines.extend([
            "-" * 60,
            "LITERATURE SUPPORT",
            "-" * 60,
            report.literature_support,
            "",
        ])

    # Recommendations
    if report.recommendations:
        lines.extend([
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60,
        ])
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Disclaimer
    lines.extend([
        "=" * 60,
        "DISCLAIMER",
        "=" * 60,
        report.disclaimer,
        "",
    ])

    return "\n".join(lines)


class ReportAgent:
    """
    Report generation agent.

    This agent creates comprehensive, clinically relevant
    prognosis reports from assessment results.
    """

    def __init__(self):
        """Initialize the report agent."""
        self.llm = get_llm_client()
        self.system_prompt = SYSTEM_PROMPT

    def generate(
        self,
        patient_data: PatientData,
        risk_assessment: RiskAssessment,
        explanation: Explanation,
        features: MetabolicFeatures = None,
        evidence: LiteratureEvidence = None
    ) -> FinalReport:
        """
        Generate a final prognosis report.

        Args:
            patient_data: Patient information
            risk_assessment: Risk assessment results
            explanation: Reasoning explanation
            features: Optional metabolic features
            evidence: Optional literature evidence

        Returns:
            FinalReport object
        """
        state = AgentState(
            patient_data=patient_data,
            metabolic_features=features,
            literature_evidence=evidence,
            risk_assessment=risk_assessment,
            explanation=explanation
        )

        state = generate_report(state)

        return state.final_report

    def format_markdown(self, report: FinalReport) -> str:
        """Format report as Markdown."""
        return format_report_text(report)
