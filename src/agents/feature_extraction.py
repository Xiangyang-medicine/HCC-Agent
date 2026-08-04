"""
Feature Extraction Agent for the HCC Prognosis System.

This agent is responsible for:
- Analyzing metabolic pathways from gene expression data
- Extracting key metabolic features
- Identifying biomarkers and subtypes
"""

from typing import Dict, Any
from src.state.schema import (
    AgentState, PatientData, MetabolicFeatures,
    RiskLevel, LiteratureEvidence
)
from src.tools.kegg_analyzer import get_kegg_analyzer
from src.utils.llm_client import get_llm_client


SYSTEM_PROMPT = """You are the Feature Extraction Agent for an HCC Prognosis System.

Your role is to:
1. Analyze gene expression data to extract metabolic features
2. Identify enriched metabolic pathways
3. Predict metabolic subtypes
4. Highlight key biomarkers

You work with:
- RNA-seq gene expression data
- Known metabolic pathways (KEGG)
- Published metabolic subtypes

Always provide:
- Clear interpretations of the data
- Appropriate statistical context
- Clinical relevance of findings"""


def extract_metabolic_features(state: AgentState) -> AgentState:
    """
    Extract metabolic features from patient gene expression data.

    This is the main function for the feature extraction agent.
    It analyzes metabolic pathways, identifies biomarkers, and predicts subtypes.
    """
    patient = state.patient_data
    llm = get_llm_client()
    kegg = get_kegg_analyzer()

    # Initialize metabolic features
    features = MetabolicFeatures()

    # Get gene expression data
    gene_expression = patient.gene_expression or {}

    if not gene_expression:
        # No gene expression data available
        features.summary = (
            "No gene expression data available for metabolic analysis. "
            "Analysis based on clinical features only."
        )
        state.metabolic_features = features
        state.completed_tasks.append("feature_extraction")
        return state

    # Analyze pathways
    pathway_results = kegg.analyze_pathways(gene_expression, top_n=8)

    # Store pathway activities
    pathway_activities = {}
    enriched_pathways = []

    for result in pathway_results:
        pathway_activities[result.pathway_id] = result.effect_size

        enriched_pathways.append({
            "pathway_id": result.pathway_id,
            "pathway_name": result.pathway_name,
            "p_value": result.p_value,
            "regulation": result.regulation,
            "enriched_genes": result.enriched_genes,
            "effect_size": result.effect_size
        })

    features.pathway_activities = pathway_activities
    features.enriched_pathways = enriched_pathways

    # Get metabolic subtype prediction
    subtype_result = kegg.get_metabolic_subtype(gene_expression)
    features.predicted_subtype = subtype_result["predicted_subtype"]
    features.subtype_confidence = subtype_result["subtype_confidence"]

    # Get key biomarkers
    biomarkers = kegg.get_key_biomarkers(gene_expression)
    features.key_biomarkers = biomarkers

    # Get metabolic genes
    known_metabolic_genes = kegg.HCC_METABOLIC_PATHWAYS
    metabolic_gene_list = []

    for pathway_id, pathway_info in known_metabolic_genes.items():
        for gene in pathway_info["genes"]:
            if gene in gene_expression:
                metabolic_gene_list.append({
                    "gene": gene,
                    "expression": float(gene_expression[gene]),
                    "pathway": pathway_info["name"]
                })

    features.metabolic_genes = metabolic_gene_list[:20]  # Top 20

    # Generate summary using LLM
    summary_prompt = f"""Summarize the metabolic features analysis for this HCC patient:

Gene Expression Data:
- Number of genes analyzed: {len(gene_expression)}
- Top expressed genes: {sorted(gene_expression.items(), key=lambda x: x[1], reverse=True)[:5]}

Pathway Analysis:
{_format_pathways(enriched_pathways[:5])}

Metabolic Subtype: {subtype_result['predicted_subtype']} (confidence: {subtype_result['subtype_confidence']:.1%})

Key Biomarkers:
{_format_biomarkers(biomarkers[:5])}

Provide a concise clinical summary of these metabolic findings."""

    summary = llm.generate(summary_prompt, system_prompt=SYSTEM_PROMPT, thinking=False)
    features.summary = summary

    state.metabolic_features = features
    state.completed_tasks.append("feature_extraction")
    state.messages.append({
        "role": "assistant",
        "content": f"Feature extraction completed: {len(enriched_pathways)} pathways analyzed"
    })

    return state


def _format_pathways(pathways: list) -> str:
    """Format pathway results for prompt."""
    if not pathways:
        return "No significant pathways identified"

    lines = []
    for p in pathways:
        lines.append(f"- {p['pathway_name']}: {p['regulation'].upper()} (p={p['p_value']:.4f})")
    return "\n".join(lines)


def _format_biomarkers(biomarkers: list) -> str:
    """Format biomarkers for prompt."""
    if not biomarkers:
        return "No significant biomarkers identified"

    lines = []
    for b in biomarkers:
        direction = "↑" if b['fold_change'] > 1 else "↓"
        lines.append(f"- {b['gene']} ({b['name']}): {direction} {b['fold_change']:.2f}x")
    return "\n".join(lines)


def summarize_features(state: AgentState) -> str:
    """
    Generate a summary of extracted features for the user.

    Args:
        state: Current agent state

    Returns:
        Summary string
    """
    features = state.metabolic_features

    if not features:
        return "No metabolic features extracted."

    parts = ["## Metabolic Feature Analysis\n"]

    if features.predicted_subtype:
        parts.append(f"**Predicted Subtype:** {features.predicted_subtype}")
        parts.append(f"**Confidence:** {features.subtype_confidence:.1%}\n")

    if features.enriched_pathways:
        parts.append(f"**Enriched Pathways:** {len(features.enriched_pathways)}")
        for p in features.enriched_pathways[:3]:
            parts.append(f"- {p['pathway_name']} ({p['regulation']})")

    if features.key_biomarkers:
        parts.append(f"\n**Key Biomarkers:**")
        for b in features.key_biomarkers[:5]:
            direction = "↑" if b['fold_change'] > 1 else "↓"
            parts.append(f"- {b['gene']}: {direction} {b['fold_change']:.2f}x")

    if features.summary:
        parts.append(f"\n**Summary:**\n{features.summary}")

    return "\n".join(parts)


class FeatureExtractionAgent:
    """
    Feature extraction agent for metabolic analysis.

    This agent processes gene expression data to extract
    meaningful metabolic features for prognosis assessment.
    """

    def __init__(self):
        """Initialize the feature extraction agent."""
        self.llm = get_llm_client()
        self.kegg = get_kegg_analyzer()
        self.system_prompt = SYSTEM_PROMPT

    def analyze(self, patient_data: PatientData) -> MetabolicFeatures:
        """
        Analyze patient data and extract metabolic features.

        Args:
            patient_data: Patient data with gene expression

        Returns:
            Extracted metabolic features
        """
        # Create a minimal state for processing
        state = AgentState(patient_data=patient_data)
        state = extract_metabolic_features(state)

        return state.metabolic_features

    def get_clinical_interpretation(
        self,
        features: MetabolicFeatures
    ) -> str:
        """
        Generate clinical interpretation of metabolic features.

        Args:
            features: Extracted metabolic features

        Returns:
            Clinical interpretation text
        """
        prompt = f"""Provide clinical interpretation of the following metabolic features for HCC prognosis:

Metabolic Subtype: {features.predicted_subtype or 'Unknown'}

Top Enriched Pathways:
{_format_pathways(features.enriched_pathways[:5])}

Key Biomarkers:
{_format_biomarkers(features.key_biomarkers[:5])}

Provide clinical interpretation focusing on:
1. Prognostic implications
2. Potential treatment considerations
3. Monitoring recommendations"""

        return self.llm.generate(prompt, system_prompt=self.system_prompt)


# Convenience function
def get_feature_extraction_agent() -> FeatureExtractionAgent:
    """Get a feature extraction agent instance."""
    return FeatureExtractionAgent()
