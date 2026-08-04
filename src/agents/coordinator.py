"""
Coordinator Agent for the HCC Prognosis Assessment System.

This agent is responsible for:
- Understanding the user's intent
- Decomposing tasks into sub-tasks
- Coordinating specialist agents
- Integrating results into final reports
"""

from typing import Dict, Any, List
from src.state.schema import AgentState, PatientData, FinalReport, RiskAssessment, Explanation
from src.utils.llm_client import get_llm_client


SYSTEM_PROMPT = """You are the Coordinator Agent for a Hepatocellular Carcinoma (HCC) Prognosis Assessment System.

Your role is to:
1. Understand the user's intent and decompose tasks
2. Coordinate specialist agents (feature extraction, literature, reasoning)
3. Integrate results into coherent prognosis reports

You work with patient data including:
- Clinical information (stage, grade, lab values)
- Gene expression data (RNA-seq)
- Survival outcomes

You must produce:
- Structured risk assessments (low/intermediate/high/very high)
- Evidence-based explanations
- Clinically actionable reports

Always maintain:
- Clinical relevance
- Appropriate uncertainty quantification
- Alignment with medical best practices

Remember: This is a decision support tool. The final clinical decisions remain with the healthcare professional."""


def analyze_intent(state: AgentState) -> AgentState:
    """
    Analyze user intent and determine required actions.

    This is the entry point for the agent workflow.
    """
    llm = get_llm_client()

    # Build context from current state
    patient = state.patient_data
    context = _build_context(state)

    prompt = f"""Analyze the following patient data and determine the prognosis assessment workflow:

{context}

Available tasks:
1. feature_extraction - Analyze metabolic features from gene expression
2. literature_search - Find relevant research literature
3. risk_reasoning - Perform comprehensive risk assessment
4. report_generation - Generate final prognosis report

Determine which tasks are needed and in what order. Output a structured plan."""

    response = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    # Update state with current task
    state.current_task = "intent_analysis"
    state.messages.append({
        "role": "assistant",
        "content": f"Intent analysis: {response}"
    })

    return state


def _build_context(state: AgentState) -> str:
    """Build context string from state."""
    patient = state.patient_data

    context_parts = []

    # Patient identifier
    if patient.patient_id:
        context_parts.append(f"Patient ID: {patient.patient_id}")

    # Clinical info
    clinical_info = []
    if patient.age:
        clinical_info.append(f"Age: {patient.age}")
    if patient.gender:
        clinical_info.append(f"Gender: {patient.gender}")
    if patient.stage:
        clinical_info.append(f"TNM Stage: {patient.stage}")
    if patient.grade:
        clinical_info.append(f"Grade: {patient.grade}")
    if patient.bclc_stage:
        clinical_info.append(f"BCLC Stage: {patient.bclc_stage}")

    if clinical_info:
        context_parts.append("Clinical Information: " + ", ".join(clinical_info))

    # Lab values
    lab_info = []
    if patient.afp_level:
        lab_info.append(f"AFP: {patient.afp_level:.1f} ng/mL")
    if patient.albumin:
        lab_info.append(f"Albumin: {patient.albumin:.1f} g/dL")
    if patient.bilirubin:
        lab_info.append(f"Bilirubin: {patient.bilirubin:.2f} mg/dL")

    if lab_info:
        context_parts.append("Laboratory Values: " + ", ".join(lab_info))

    # Gene expression
    if patient.gene_expression:
        gene_count = len(patient.gene_expression)
        context_parts.append(f"Gene Expression Data: {gene_count} genes available")

    # Completed tasks
    if state.completed_tasks:
        context_parts.append(f"Completed Tasks: {', '.join(state.completed_tasks)}")

    # Errors
    if state.error_messages:
        context_parts.append(f"Warnings: {'; '.join(state.error_messages)}")

    return "\n".join(context_parts) if context_parts else "No patient data available"


def plan_task_decomposition(state: AgentState) -> AgentState:
    """
    Plan the decomposition of work into sub-tasks.

    Called after intent analysis to plan the workflow.
    """
    llm = get_llm_client()

    context = _build_context(state)

    prompt = f"""Based on the following patient context, plan the prognosis assessment workflow:

{context}

Task options:
1. feature_extraction - Required if gene expression data available
2. literature_search - Always useful for evidence
3. risk_reasoning - Required for final assessment
4. report_generation - Required to produce output

Output:
1. List of tasks to execute (in order)
2. Dependencies between tasks
3. Estimated complexity (simple/moderate/complex)"""

    response = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    state.messages.append({
        "role": "assistant",
        "content": f"Task decomposition: {response}"
    })

    return state


def should_continue(state: AgentState) -> str:
    """
    Determine if the workflow should continue to the next stage.

    Returns:
        "feature_extraction" to continue to feature extraction
        "literature_search" to skip to literature search
        "report_generation" to generate report directly
        "end" if workflow is complete
    """
    patient = state.patient_data

    # Check what data we have
    has_gene_expression = patient.gene_expression is not None and len(patient.gene_expression) > 0
    has_clinical = any([patient.stage, patient.grade, patient.bclc_stage, patient.afp_level])

    if not has_clinical and not has_gene_expression:
        # Not enough data
        return "end"

    if has_gene_expression and state.metabolic_features is None:
        return "feature_extraction"

    if state.literature_evidence is None:
        return "literature_search"

    if state.risk_assessment is None:
        return "risk_reasoning"

    if state.final_report is None:
        return "report_generation"

    return "end"


def summarize_results(state: AgentState) -> AgentState:
    """
    Summarize the assessment results for the user.

    Called at the end of the workflow.
    """
    llm = get_llm_client()

    # Build summary
    summary_parts = ["# HCC Prognosis Assessment Summary\n"]

    if state.final_report:
        summary_parts.append(state.final_report.executive_summary)

        if state.risk_assessment:
            risk = state.risk_assessment
            summary_parts.append(f"\n## Risk Level: {risk.risk_level.value.upper()}")
            summary_parts.append(f"Confidence: {risk.confidence_score:.0%}")

            if risk.estimated_survival_months:
                summary_parts.append(f"\nEstimated Median Survival: {risk.estimated_survival_months:.1f} months")
                if risk.survival_estimate_ci_low and risk.survival_estimate_ci_high:
                    summary_parts.append(f"95% CI: {risk.survival_estimate_ci_low:.1f} - {risk.survival_estimate_ci_high:.1f} months")

    state.messages.append({
        "role": "assistant",
        "content": "\n".join(summary_parts)
    })

    return state


class CoordinatorAgent:
    """
    Main coordinator agent for the HCC prognosis system.

    This class orchestrates the entire workflow from patient data
    input to final prognosis report generation.
    """

    def __init__(self):
        """Initialize the coordinator agent."""
        self.llm = get_llm_client()
        self.system_prompt = SYSTEM_PROMPT

    def process(self, patient_data: PatientData) -> FinalReport:
        """
        Process patient data through the full prognosis pipeline.

        Args:
            patient_data: Input patient data

        Returns:
            Final prognosis report
        """
        # Initialize state
        state = AgentState(patient_data=patient_data)

        # Run workflow stages
        state = analyze_intent(state)
        state = plan_task_decomposition(state)

        # Note: Actual agent calls are handled by the LangGraph workflow
        # This method provides a high-level interface

        return state.final_report

    def generate_clinical_summary(
        self,
        patient_data: PatientData,
        risk_assessment: RiskAssessment,
        explanation: Explanation
    ) -> str:
        """
        Generate a clinical summary for healthcare professionals.

        Args:
            patient_data: Patient information
            risk_assessment: Risk assessment results
            explanation: Reasoning explanation

        Returns:
            Clinical summary text
        """
        prompt = f"""Generate a clinical summary for the following HCC prognosis assessment.

Patient Information:
- Age: {patient_data.age or 'Unknown'}
- Gender: {patient_data.gender or 'Unknown'}
- Stage: {patient_data.stage or 'Unknown'} ({patient_data.bclc_stage or ''})
- Grade: {patient_data.grade or 'Unknown'}

Risk Assessment:
- Risk Level: {risk_assessment.risk_level.value}
- Estimated Survival: {risk_assessment.estimated_survival_months or 'N/A'} months
- Confidence: {risk_assessment.confidence_score:.0%}

Key Risk Factors:
{self._format_factors(risk_assessment.risk_factors)}

Protective Factors:
{self._format_factors(risk_assessment.protective_factors)}

Generate a concise clinical summary suitable for healthcare professionals.
Focus on actionable insights and key findings."""

        response = self.llm.generate(prompt, system_prompt=self.system_prompt, thinking=False)
        return response

    def _format_factors(self, factors: List[Dict[str, Any]]) -> str:
        """Format risk/protective factors for prompt."""
        if not factors:
            return "None identified"

        lines = []
        for i, factor in enumerate(factors, 1):
            name = factor.get("factor", "Unknown")
            contribution = factor.get("contribution", factor.get("effect", "Unknown"))
            lines.append(f"{i}. {name}: {contribution}")

        return "\n".join(lines)


# Convenience function
def get_coordinator() -> CoordinatorAgent:
    """Get a coordinator agent instance."""
    return CoordinatorAgent()
