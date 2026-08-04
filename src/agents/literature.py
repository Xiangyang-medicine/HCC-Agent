"""
Literature Agent for the HCC Prognosis System.

This agent is responsible for:
- Searching PubMed for relevant literature
- Extracting evidence for prognosis assessment
- Synthesizing findings from multiple papers
"""

from typing import Dict, Any, List, Optional
from src.state.schema import (
    AgentState, PatientData, LiteratureEvidence, MetabolicFeatures
)
from src.tools.pubmed_tool import get_pubmed_tool
from src.utils.llm_client import get_llm_client


SYSTEM_PROMPT = """You are the Literature Agent for an HCC Prognosis System.

Your role is to:
1. Search for relevant research literature
2. Extract key evidence for prognosis assessment
3. Synthesize findings from multiple sources

You have access to:
- PubMed/PubMed Central literature database
- Cached literature results for efficiency

Always provide:
- Evidence-based conclusions
- Appropriate citation context
- Clinical relevance of findings"""


def search_literature(state: AgentState) -> AgentState:
    """
    Search for relevant literature for prognosis assessment.

    This is the main function for the literature agent.
    """
    patient = state.patient_data
    features = state.metabolic_features
    llm = get_llm_client()
    pubmed = get_pubmed_tool()

    # Build search context
    search_context = _build_search_context(patient, features)

    # Perform search
    prompt = f"""Based on the following patient context, formulate an optimal PubMed search strategy:

{search_context}

Generate a search query that will retrieve the most relevant literature for:
1. Prognostic biomarkers in HCC
2. Metabolic features and survival
3. Risk stratification factors
4. Clinical prediction models

Output a single, optimized search query."""

    search_query = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, thinking=False).strip()

    # Execute search
    evidence = pubmed.search(search_query, max_results=10)

    state.literature_evidence = evidence
    state.completed_tasks.append("literature_search")
    state.messages.append({
        "role": "assistant",
        "content": f"Literature search completed: {evidence.num_results} papers found"
    })

    return state


def _build_search_context(patient: PatientData, features: Optional[MetabolicFeatures]) -> str:
    """Build search context from patient data."""
    context_parts = []

    # Patient info
    if patient.patient_id:
        context_parts.append(f"Patient: {patient.patient_id}")

    if patient.stage:
        context_parts.append(f"Stage: {patient.stage}")
    if patient.grade:
        context_parts.append(f"Grade: {patient.grade}")
    if patient.bclc_stage:
        context_parts.append(f"BCLC: {patient.bclc_stage}")

    if patient.afp_level:
        context_parts.append(f"AFP: {patient.afp_level:.1f} ng/mL")

    # Metabolic features
    if features:
        if features.predicted_subtype:
            context_parts.append(f"Metabolic Subtype: {features.predicted_subtype}")

        if features.enriched_pathways:
            pathways = [p['pathway_name'] for p in features.enriched_pathways[:3]]
            context_parts.append(f"Pathways: {', '.join(pathways)}")

        if features.key_biomarkers:
            genes = [b['gene'] for b in features.key_biomarkers[:5]]
            context_parts.append(f"Key Genes: {', '.join(genes)}")

    # Gene expression
    if patient.gene_expression:
        top_genes = sorted(patient.gene_expression.items(), key=lambda x: x[1], reverse=True)[:5]
        gene_str = ", ".join([f"{g} ({v:.2f})" for g, v in top_genes])
        context_parts.append(f"Top Expressed Genes: {gene_str}")

    return "\n".join(context_parts) if context_parts else "No patient context available"


def synthesize_evidence(state: AgentState) -> AgentState:
    """
    Synthesize evidence from literature search results.

    Args:
        state: Current agent state

    Returns:
        Updated state with synthesized evidence
    """
    evidence = state.literature_evidence
    llm = get_llm_client()

    if not evidence or not evidence.evidence_items:
        return state

    # Format evidence items for synthesis
    evidence_text = _format_evidence_for_synthesis(evidence.evidence_items)

    synthesis_prompt = f"""Synthesize the following literature evidence for HCC prognosis assessment:

Evidence Items:
{evidence_text}

Provide:
1. Summary of key prognostic factors identified
2. Consistency of findings across studies
3. Clinical implications
4. Level of evidence (weak/moderate/strong)

Keep the synthesis concise and clinically relevant."""

    synthesis = llm.generate(synthesis_prompt, system_prompt=SYSTEM_PROMPT, thinking=False)

    # Update evidence summary
    evidence.summary = synthesis

    state.literature_evidence = evidence
    state.messages.append({
        "role": "assistant",
        "content": f"Evidence synthesis completed"
    })

    return state


def _format_evidence_for_synthesis(evidence_items: List[Dict[str, Any]]) -> str:
    """Format evidence items for synthesis prompt."""
    lines = []

    for i, item in enumerate(evidence_items[:5], 1):
        lines.append(f"""
--- Paper {i} ---
PMID: {item.get('pmid', 'Unknown')}
Title: {item.get('title', 'Unknown')}
Journal: {item.get('journal', 'Unknown')} ({item.get('year', 'Unknown')})
Key Findings: {item.get('key_findings', 'N/A')}
Relevance: {item.get('relevance_score', 0):.2f}
""")

    return "\n".join(lines)


def get_relevant_evidence(
    state: AgentState,
    topic: str
) -> List[Dict[str, Any]]:
    """
    Get evidence items relevant to a specific topic.

    Args:
        state: Current agent state
        topic: Topic of interest (e.g., "CA9", "glycolysis")

    Returns:
        List of relevant evidence items
    """
    evidence = state.literature_evidence

    if not evidence or not evidence.evidence_items:
        return []

    # Filter by relevance to topic
    relevant = []
    for item in evidence.evidence_items:
        title = item.get('title', '').lower()
        findings = item.get('key_findings', '').lower()
        abstract = item.get('abstract', '').lower()

        if topic.lower() in title or topic.lower() in findings or topic.lower() in abstract:
            relevant.append(item)

    return relevant


class LiteratureAgent:
    """
    Literature search and synthesis agent.

    This agent searches for relevant research literature
    and synthesizes evidence for prognosis assessment.
    """

    def __init__(self):
        """Initialize the literature agent."""
        self.llm = get_llm_client()
        self.pubmed = get_pubmed_tool()
        self.system_prompt = SYSTEM_PROMPT

    def search_and_synthesize(
        self,
        patient_data: PatientData,
        features: Optional[MetabolicFeatures] = None
    ) -> LiteratureEvidence:
        """
        Search for literature and synthesize findings.

        Args:
            patient_data: Patient data
            features: Optional metabolic features

        Returns:
            Literature evidence with synthesis
        """
        # Create state for processing
        state = AgentState(
            patient_data=patient_data,
            metabolic_features=features
        )

        # Search
        state = search_literature(state)

        # Synthesize
        state = synthesize_evidence(state)

        return state.literature_evidence

    def search_by_genes(
        self,
        genes: List[str],
        max_results: int = 5
    ) -> LiteratureEvidence:
        """
        Search literature by specific genes.

        Args:
            genes: List of gene names
            max_results: Maximum results

        Returns:
            Literature evidence
        """
        query = " OR ".join([f"{gene}[Title/Abstract]" for gene in genes[:5]])
        query = f"({query}) AND (hepatocellular carcinoma[Title/Abstract]) AND (prognosis[Title/Abstract])"

        return self.pubmed.search(query, max_results=max_results)


# Convenience function
def get_literature_agent() -> LiteratureAgent:
    """Get a literature agent instance."""
    return LiteratureAgent()
