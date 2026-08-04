"""
LangGraph state schema for the HCC Prognosis Assessment Multi-Agent System.

This module defines the shared state that flows through the agent graph,
including patient data, extracted features, literature evidence, and
final assessment results.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification for prognosis."""
    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PrognosisType(str, Enum):
    """Type of prognosis being assessed."""
    OVERALL_SURVIVAL = "overall_survival"
    DISEASE_FREE_SURVIVAL = "disease_free_survival"
    RECURRENCE_RISK = "recurrence_risk"


class PatientData(BaseModel):
    """Input patient data structure."""
    # Clinical information
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, description="Patient age at diagnosis")
    gender: Optional[str] = Field(None, description="Patient gender (M/F)")
    stage: Optional[str] = Field(None, description="TNM stage (I-IV)")
    grade: Optional[str] = Field(None, description="Tumor grade (G1-G4)")
    bclc_stage: Optional[str] = Field(None, description="BCLC stage (0/A/B/C/D)")

    # Treatment information
    treatment: Optional[str] = Field(None, description="Primary treatment received")
    resection_type: Optional[str] = Field(None, description="Surgical resection type")

    # Lab values
    afp_level: Optional[float] = Field(None, description="Alpha-fetoprotein level (ng/mL)")
    albumin: Optional[float] = Field(None, description="Serum albumin (g/dL)")
    bilirubin: Optional[float] = Field(None, description="Total bilirubin (mg/dL)")
    inr: Optional[float] = Field(None, description="International normalized ratio")
    platelet_count: Optional[int] = Field(None, description="Platelet count")

    # Genomics (for TCGA data)
    gene_expression: Optional[Dict[str, float]] = Field(
        None,
        description="Gene expression values (gene_id -> expression)"
    )
    mutations: Optional[List[str]] = Field(
        None,
        description="List of detected mutations"
    )
    methylation_data: Optional[Dict[str, float]] = Field(
        None,
        description="DNA methylation beta values"
    )

    # Outcomes (for evaluation)
    survival_months: Optional[float] = Field(None, description="Overall survival in months")
    vital_status: Optional[str] = Field(None, description="Vital status (alive/dead)")
    recurrence: Optional[bool] = Field(None, description="Recurrence status")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "TCGA-CC-A7WJ",
                "age": 65,
                "gender": "M",
                "stage": "T2N0M0",
                "bclc_stage": "A",
                "afp_level": 250.5,
                "gene_expression": {"CA9": 2.5, "VEGFA": 3.2, "CXCL12": 1.8},
            }
        }


class MetabolicFeatures(BaseModel):
    """Extracted metabolic features from patient data."""
    # Pathway activity scores
    pathway_activities: Dict[str, float] = Field(
        default_factory=dict,
        description="Activity scores for metabolic pathways (pathway_id -> score)"
    )

    # Key metabolic genes
    metabolic_genes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of key metabolic genes with expression and fold change"
    )

    # Metabolic subtypes
    predicted_subtype: Optional[str] = Field(
        None,
        description="Predicted metabolic subtype"
    )
    subtype_confidence: Optional[float] = Field(
        None,
        description="Confidence score for subtype prediction"
    )

    # Enrichment results
    enriched_pathways: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Significantly enriched pathways"
    )

    # Key biomarkers
    key_biomarkers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key metabolic biomarkers identified"
    )

    # Feature summary
    summary: str = Field(
        default="",
        description="Text summary of extracted features"
    )


class LiteratureEvidence(BaseModel):
    """Literature evidence retrieved for prognosis assessment."""
    # Search metadata
    search_query: str = Field(
        default="",
        description="The search query used"
    )
    num_results: int = Field(
        default=0,
        description="Number of papers found"
    )

    # Evidence items
    evidence_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of relevant literature with extracted evidence"
    )

    # Summary
    summary: str = Field(
        default="",
        description="Summary of key findings from literature"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "search_query": "HCC metabolic genes prognosis survival",
                "num_results": 10,
                "evidence_items": [
                    {
                        "pmid": "12345678",
                        "title": "Metabolic gene signature...",
                        "key_findings": "High CA9 expression associated with poor prognosis"
                    }
                ]
            }
        }


class RiskAssessment(BaseModel):
    """Final risk assessment result."""
    # Risk classification
    risk_level: RiskLevel = Field(
        RiskLevel.INTERMEDIATE,
        description="Overall risk level"
    )

    # Prognosis estimates
    estimated_survival_months: Optional[float] = Field(
        None,
        description="Estimated median survival in months"
    )
    survival_estimate_ci_low: Optional[float] = Field(
        None,
        description="Lower bound of 95% CI"
    )
    survival_estimate_ci_high: Optional[float] = Field(
        None,
        description="Upper bound of 95% CI"
    )

    # Risk factors
    risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified risk factors with their contributions"
    )

    # Protective factors
    protective_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified protective factors"
    )

    # Confidence
    confidence_score: float = Field(
        default=0.5,
        description="Confidence in the assessment (0-1)"
    )

    # Evidence strength
    evidence_strength: str = Field(
        default="moderate",
        description="Strength of supporting evidence (weak/moderate/strong)"
    )


class Explanation(BaseModel):
    """Explanations for the risk assessment."""
    # Reasoning chain
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning chain"
    )

    # Key factors explained
    factor_explanations: Dict[str, str] = Field(
        default_factory=dict,
        description="Explanations for each key factor"
    )

    # Alternative scenarios
    alternative_scenarios: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative risk scenarios considered"
    )

    # Limitations
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations of this assessment"
    )

    # Caveats
    caveats: List[str] = Field(
        default_factory=list,
        description="Important caveats for clinical interpretation"
    )


class FinalReport(BaseModel):
    """Final prognosis assessment report."""
    # Report metadata
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Report generation timestamp"
    )

    # Executive summary
    executive_summary: str = Field(
        default="",
        description="Brief executive summary of the assessment"
    )

    # Detailed findings
    clinical_findings: str = Field(
        default="",
        description="Clinical findings section"
    )
    metabolic_findings: str = Field(
        default="",
        description="Metabolic analysis findings"
    )
    literature_support: str = Field(
        default="",
        description="Literature support section"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Clinical recommendations"
    )

    # Risk level
    risk_assessment: RiskAssessment = Field(
        default_factory=RiskAssessment,
        description="Risk assessment details"
    )

    # Explanation
    explanation: Explanation = Field(
        default_factory=Explanation,
        description="Detailed explanations"
    )

    # Disclaimer
    disclaimer: str = Field(
        default="This assessment is for research and decision support purposes only. "
                "Clinical decisions should be made by qualified healthcare professionals.",
        description="Clinical disclaimer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "TCGA-CC-A7WJ",
                "executive_summary": "Patient with intermediate-high risk profile...",
                "recommendations": ["Consider adjuvant therapy", "Frequent follow-up monitoring"]
            }
        }


class AgentState(BaseModel):
    """
    Main state class for LangGraph agent workflow.

    This defines all the state that is passed between agents in the system.
    """
    # Input
    patient_data: PatientData = Field(
        default_factory=PatientData,
        description="Input patient data"
    )

    # Processing state
    current_task: Optional[str] = Field(
        None,
        description="Current task being processed"
    )
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="List of completed tasks"
    )
    error_messages: List[str] = Field(
        default_factory=list,
        description="Any error messages encountered"
    )

    # Intermediate results
    metabolic_features: Optional[MetabolicFeatures] = Field(
        None,
        description="Extracted metabolic features"
    )
    literature_evidence: Optional[LiteratureEvidence] = Field(
        None,
        description="Retrieved literature evidence"
    )

    # Final outputs
    risk_assessment: Optional[RiskAssessment] = Field(
        None,
        description="Final risk assessment"
    )
    explanation: Optional[Explanation] = Field(
        None,
        description="Risk explanation"
    )
    final_report: Optional[FinalReport] = Field(
        None,
        description="Final prognosis report"
    )

    # Conversation context
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history"
    )

    class Config:
        arbitrary_types_allowed = True
