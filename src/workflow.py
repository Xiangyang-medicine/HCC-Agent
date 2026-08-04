"""
HCC Prognosis Assessment Multi-Agent Workflow.

This module implements the LangGraph workflow that orchestrates
the multi-agent system for HCC prognosis assessment.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from src.state.schema import AgentState
from src.agents.coordinator import (
    analyze_intent, plan_task_decomposition, should_continue, summarize_results
)
from src.agents.feature_extraction import extract_metabolic_features
from src.agents.literature import search_literature, synthesize_evidence
from src.agents.reasoning import assess_risk, generate_explanation
from src.agents.report import generate_report


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for HCC prognosis assessment.

    The workflow consists of:
    1. Intent analysis and task planning
    2. Feature extraction (if gene expression available)
    3. Literature search
    4. Risk reasoning
    5. Report generation

    Returns:
        Compiled LangGraph StateGraph
    """
    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("plan_tasks", plan_task_decomposition)
    workflow.add_node("feature_extraction", extract_metabolic_features)
    workflow.add_node("literature_search", search_literature)
    workflow.add_node("synthesize_evidence", synthesize_evidence)
    workflow.add_node("risk_reasoning", assess_risk)
    workflow.add_node("generate_explanation", generate_explanation)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("summarize", summarize_results)

    # Set entry point
    workflow.set_entry_point("analyze_intent")

    # Linear workflow with conditional skip of feature_extraction
    workflow.add_edge("analyze_intent", "plan_tasks")
    workflow.add_edge("plan_tasks", "feature_extraction")
    workflow.add_edge("feature_extraction", "literature_search")
    workflow.add_edge("literature_search", "synthesize_evidence")
    workflow.add_edge("synthesize_evidence", "risk_reasoning")
    workflow.add_edge("risk_reasoning", "generate_explanation")
    workflow.add_edge("generate_explanation", "generate_report")
    workflow.add_edge("generate_report", "summarize")
    workflow.add_edge("summarize", END)

    # Compile the graph
    return workflow.compile()


# Global workflow instance
_workflow = None


def get_workflow() -> StateGraph:
    """Get the global workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow


def run_assessment(patient_data, config: dict = None) -> AgentState:
    """
    Run the full prognosis assessment workflow.

    Args:
        patient_data: PatientData object with patient information
        config: Optional configuration dictionary

    Returns:
        AgentState with all results including final_report
    """
    workflow = get_workflow()

    # Initialize state
    initial_state = AgentState(patient_data=patient_data)

    # Run workflow - LangGraph returns dict, convert back to AgentState
    result = workflow.invoke(initial_state)

    if isinstance(result, dict):
        return AgentState(**result)
    return result


def run_assessment_stream(patient_data, config: dict = None):
    """
    Run the assessment workflow with streaming output.

    Args:
        patient_data: PatientData object
        config: Optional configuration

    Yields:
        State updates as the workflow progresses
    """
    workflow = get_workflow()

    initial_state = AgentState(patient_data=patient_data)

    for state_update in workflow.stream(initial_state):
        yield state_update


class HCCPrognosisWorkflow:
    """
    High-level interface to the HCC prognosis workflow.

    This class provides a simpler interface for running
    prognosis assessments with proper error handling.
    """

    def __init__(self):
        """Initialize the workflow."""
        self.workflow = get_workflow()

    def assess(self, patient_data) -> dict:
        """
        Run prognosis assessment on patient data.

        Args:
            patient_data: PatientData object

        Returns:
            Dictionary with assessment results
        """
        try:
            result = self.workflow.invoke(AgentState(patient_data=patient_data))

            # LangGraph returns a dict, convert back to AgentState
            if isinstance(result, dict):
                state = AgentState(**result)
            else:
                state = result

            return {
                "success": True,
                "patient_id": patient_data.patient_id,
                "risk_level": state.risk_assessment.risk_level.value if state.risk_assessment else None,
                "survival_estimate": state.risk_assessment.estimated_survival_months if state.risk_assessment else None,
                "confidence": state.risk_assessment.confidence_score if state.risk_assessment else None,
                "report": state.final_report,
                "state": state
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "patient_id": patient_data.patient_id
            }

    def get_supported_stages(self) -> list:
        """Get list of supported workflow stages."""
        return [
            "intent_analysis",
            "feature_extraction",
            "literature_search",
            "risk_reasoning",
            "report_generation"
        ]
