"""
HCC Prognosis Assessment Application.

This is the main entry point for the multi-agent prognosis system.
It provides a CLI interface for running assessments.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.schema import (
    PatientData, MetabolicFeatures, LiteratureEvidence,
    RiskAssessment, RiskLevel, Explanation, FinalReport
)
from src.workflow import HCCPrognosisWorkflow
from src.agents.report import format_report_text
from src.tools.tcga_loader import get_tcga_loader
from src.tools.kegg_analyzer import get_kegg_analyzer
from src.utils.llm_client import init_llm_client
from config.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HCC Prognosis Assessment Multi-Agent System"
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        help="TCGA patient ID to assess (e.g., TCGA-CC-A7WJ)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Claude API key (or set CLAUDE_API_KEY env var)"
    )

    parser.add_argument(
        "--api-base",
        type=str,
        help="API base URL for proxy"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample patient data"
    )

    parser.add_argument(
        "--list-patients",
        action="store_true",
        help="List available patient IDs from TCGA"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for report (default: print to stdout)"
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls and use mock data (for testing)"
    )

    return parser.parse_args()


def run_demo_with_mock():
    """Run a demo assessment with mock data (no LLM required)."""
    print("\n" + "=" * 60)
    print("HCC PROGNOSIS ASSESSMENT - DEMO MODE (Mock)")
    print("=" * 60 + "\n")

    # Create sample patient data
    patient = PatientData(
        patient_id="DEMO-001",
        age=62,
        gender="M",
        stage="T2N0M0",
        grade="G2",
        bclc_stage="A",
        afp_level=350.0,
        albumin=3.8,
        bilirubin=1.2,
        treatment="Surgical resection",
        gene_expression={
            "CA9": 3.5,  # Hypoxia marker - elevated
            "VEGFA": 4.2,  # Angiogenesis - elevated
            "HK2": 4.0,  # Glycolysis - elevated
            "PKM2": 3.8,
            "LDHA": 4.1,
            "IDH1": 2.5,
            "GLS": 3.2,
            "SDHA": 2.8,
            "TP53": 1.5,  # Tumor suppressor - reduced
            "CTNNB1": 3.0,
        }
    )

    print("Sample Patient Data:")
    print(f"  Age: {patient.age}, Gender: {patient.gender}")
    print(f"  Stage: {patient.stage} ({patient.bclc_stage})")
    print(f"  Grade: {patient.grade}")
    print(f"  AFP: {patient.afp_level} ng/mL")
    print(f"  Gene Expression: {len(patient.gene_expression)} genes")
    print()

    # Run metabolic analysis (without LLM)
    kegg = get_kegg_analyzer()

    # Extract metabolic features
    pathway_results = kegg.analyze_pathways(patient.gene_expression, top_n=5)

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

    subtype_result = kegg.get_metabolic_subtype(patient.gene_expression)
    biomarkers = kegg.get_key_biomarkers(patient.gene_expression)

    features = MetabolicFeatures(
        pathway_activities=pathway_activities,
        enriched_pathways=enriched_pathways,
        predicted_subtype=subtype_result["predicted_subtype"],
        subtype_confidence=subtype_result["subtype_confidence"],
        key_biomarkers=biomarkers,
        metabolic_genes=[
            {"gene": g, "expression": v, "pathway": "Metabolic"}
            for g, v in patient.gene_expression.items()
        ],
        summary=f"Patient shows {subtype_result['predicted_subtype']} metabolic subtype with elevated glycolysis and hypoxia markers."
    )

    # Mock literature evidence
    literature = LiteratureEvidence(
        search_query="HCC metabolic prognosis glycolysis hypoxia",
        num_results=5,
        evidence_items=[
            {
                "pmid": "32540321",
                "title": "Metabolic reprogramming in hepatocellular carcinoma",
                "year": 2020,
                "key_findings": "Glycolysis and glutamine metabolism are key metabolic pathways in HCC progression"
            },
            {
                "pmid": "31254232",
                "title": "Hypoxia-induced carbonic anhydrase IX in HCC",
                "year": 2019,
                "key_findings": "CA9 expression correlates with poor prognosis in HCC patients"
            }
        ],
        summary="Literature supports the prognostic significance of metabolic markers in HCC."
    )

    # Mock risk assessment
    risk = RiskAssessment(
        risk_level=RiskLevel.INTERMEDIATE,
        estimated_survival_months=36.0,
        survival_estimate_ci_low=24.0,
        survival_estimate_ci_high=48.0,
        risk_factors=[
            {"factor": "Elevated AFP (350 ng/mL)", "contribution": "High", "description": "AFP > 400 is associated with worse prognosis"},
            {"factor": "T2 Stage", "contribution": "Moderate", "description": "Multifocal tumor without vascular invasion"},
            {"factor": "Hypoxic metabolic subtype", "contribution": "Moderate", "description": "CA9 and VEGFA elevated"},
        ],
        protective_factors=[
            {"factor": "BCLC Stage A", "contribution": "Favorable", "description": "Early stage with preserved liver function"},
            {"factor": "Surgical resection performed", "contribution": "Favorable", "description": "Curative intent treatment"},
        ],
        confidence_score=0.75,
        evidence_strength="moderate"
    )

    # Mock explanation
    explanation = Explanation(
        reasoning_chain=[
            "1. Clinical stage analysis: T2N0M0 indicates multifocal tumor without vascular invasion",
            "2. AFP level of 350 ng/mL is elevated but below the high-risk threshold of 400",
            "3. Metabolic analysis shows hypoxic subtype with elevated CA9 and VEGFA",
            "4. Glycolysis pathway is upregulated (HK2, PKM2, LDHA elevated)",
            "5. Combined clinical and molecular factors suggest intermediate risk"
        ],
        factor_explanations={
            "AFP": "Alpha-fetoprotein is a traditional HCC biomarker. Levels >400 ng/mL are associated with worse outcomes.",
            "Metabolic subtype": "The hypoxic metabolic subtype is associated with aggressive tumor behavior and poorer prognosis."
        },
        alternative_scenarios=[
            {
                "scenario": "High-risk scenario",
                "probability": "20%",
                "description": "If metabolic markers were more elevated and AFP > 1000"
            }
        ],
        limitations=[
            "Analysis based on gene expression data without histological confirmation",
            "Mock survival estimates based on published literature, not patient-specific",
            "Further clinical correlation required for definitive assessment"
        ],
        caveats=[
            "This is a research tool, not a clinical diagnostic device",
            "All recommendations should be verified by qualified oncologists"
        ]
    )

    # Generate final report
    report = FinalReport(
        patient_id=patient.patient_id,
        generated_at=datetime.now(),
        executive_summary=(
            f"This is a {risk.risk_level.value} risk HCC patient. "
            f"Based on T2N0M0 staging, elevated AFP (350 ng/mL), and hypoxic metabolic profile, "
            f"estimated median survival is {risk.estimated_survival_months:.0f} months. "
            f"Curative surgical resection was performed, which is a favorable prognostic factor."
        ),
        clinical_findings=(
            f"Patient presents with T2N0M0 stage ({patient.bclc_stage} BCLC) HCC. "
            f"AFP is elevated at {patient.afp_level} ng/mL. "
            f"Liver function markers (albumin {patient.albumin} g/dL, bilirubin {patient.bilirubin} mg/dL) are within acceptable ranges. "
            f"Patient underwent surgical resection, indicating potentially resectable disease."
        ),
        metabolic_findings=(
            f"Metabolic analysis reveals a {subtype_result['predicted_subtype']} subtype. "
            f"Glycolysis pathway is upregulated with elevated HK2 ({patient.gene_expression['HK2']}), PKM2 ({patient.gene_expression['PKM2']}), and LDHA ({patient.gene_expression['LDHA']}). "
            f"Hypoxia markers CA9 ({patient.gene_expression['CA9']}) and VEGFA ({patient.gene_expression['VEGFA']}) are elevated, suggesting aggressive tumor metabolism."
        ),
        literature_support=(
            f"Published literature supports the prognostic significance of metabolic markers in HCC. "
            f"Elevated glycolysis and hypoxia markers are associated with worse outcomes. "
            f"The patient's metabolic profile is consistent with intermediate-risk disease."
        ),
        recommendations=[
            "Continue regular surveillance with imaging every 3-6 months",
            "Monitor AFP trends as a recurrence indicator",
            "Consider adjuvant therapy consultation given intermediate risk profile",
            "Evaluate liver function periodically",
            "Multidisciplinary discussion at tumor board recommended"
        ],
        risk_assessment=risk,
        explanation=explanation
    )

    # Format and print report
    print("Assessment completed successfully!\n")
    print(format_report_text(report))

    return {
        "success": True,
        "patient_id": patient.patient_id,
        "report": report
    }


def run_demo_with_llm():
    """Run a demo assessment with full LLM integration."""
    print("\n" + "=" * 60)
    print("HCC PROGNOSIS ASSESSMENT - DEMO MODE")
    print("=" * 60 + "\n")

    # Create sample patient data
    patient = PatientData(
        patient_id="DEMO-001",
        age=62,
        gender="M",
        stage="T2N0M0",
        grade="G2",
        bclc_stage="A",
        afp_level=350.0,
        albumin=3.8,
        bilirubin=1.2,
        treatment="Surgical resection",
        gene_expression={
            "CA9": 3.5,
            "VEGFA": 4.2,
            "HK2": 4.0,
            "PKM2": 3.8,
            "LDHA": 4.1,
            "IDH1": 2.5,
            "GLS": 3.2,
            "SDHA": 2.8,
            "TP53": 1.5,
            "CTNNB1": 3.0,
        }
    )

    print("Sample Patient Data:")
    print(f"  Age: {patient.age}, Gender: {patient.gender}")
    print(f"  Stage: {patient.stage} ({patient.bclc_stage})")
    print(f"  Grade: {patient.grade}")
    print(f"  AFP: {patient.afp_level} ng/mL")
    print(f"  Gene Expression: {len(patient.gene_expression)} genes")
    print()

    # Run assessment
    workflow = HCCPrognosisWorkflow()
    result = workflow.assess(patient)

    if result["success"]:
        print("Assessment completed successfully!\n")
        report = result["state"].final_report
        print(format_report_text(report))
    else:
        print(f"Assessment failed: {result['error']}")

    return result


def run_demo():
    """Run demo - uses LLM if available, otherwise falls back to mock."""
    import os

    # Check if API key is available
    api_key = os.environ.get("CLAUDE_API_KEY") or None

    # If no API key in env, try config
    if not api_key:
        try:
            from config.config import config
            api_key = config.api.api_key if hasattr(config, 'api') else None
        except:
            pass

    # If no API key, skip LLM test and run mock
    if not api_key or api_key == "sk-308KwjH0x1DHdKBm9hxJf25bYWWPdrMI1JXTfvKew5Ki1ERC":
        print("Note: Running in mock mode (no valid API key detected)\n")
        return run_demo_with_mock()

    # Check if we can use LLM (with timeout)
    try:
        from src.utils.llm_client import get_llm_client
        import threading

        result = {"response": None, "error": None}

        def test_llm():
            try:
                llm = get_llm_client()
                result["response"] = llm.generate("Say OK", max_tokens=10)
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=test_llm)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)  # 10 second timeout

        if thread.is_alive():
            # Thread still running, LLM not responding
            print("Note: LLM connection timeout, running in mock mode\n")
            return run_demo_with_mock()

        if result["error"]:
            print(f"Note: LLM not available ({type(result['error']).__name__}), running in mock mode\n")
            return run_demo_with_mock()

        if result["response"] and len(result["response"].strip()) > 0:
            return run_demo_with_llm()
        else:
            print("Note: LLM returned empty response, running in mock mode\n")

    except Exception as e:
        print(f"Note: LLM not available ({type(e).__name__}), running in mock mode\n")

    # Fall back to mock mode
    return run_demo_with_mock()


def run_patient_assessment(patient_id: str, workflow: HCCPrognosisWorkflow):
    """Run assessment for a specific patient."""
    print(f"\nFetching patient data for: {patient_id}")

    tcga = get_tcga_loader()
    patient = tcga.get_patient(patient_id)

    if not patient:
        print(f"Patient {patient_id} not found in dataset")
        return None

    print("Patient Data Loaded:")
    print(f"  Age: {patient.age}, Gender: {patient.gender}")
    print(f"  Stage: {patient.stage} ({patient.bclc_stage})")
    print(f"  Grade: {patient.grade}")
    if patient.afp_level:
        print(f"  AFP: {patient.afp_level} ng/mL")
    if patient.gene_expression:
        print(f"  Gene Expression: {len(patient.gene_expression)} genes")
    print()

    # Run assessment
    result = workflow.assess(patient)

    if result["success"]:
        print("Assessment completed successfully!\n")
        report = result["state"].final_report
        report_text = format_report_text(report)
        print(report_text)
        return report_text
    else:
        print(f"Assessment failed: {result['error']}")
        return None


def list_patients():
    """List available patient IDs."""
    tcga = get_tcga_loader()
    patients = tcga.get_cohort(n=20)

    print("\nAvailable Patient IDs (first 20):")
    print("-" * 40)
    for p in patients:
        stage_info = f"{p.stage or 'N/A'} ({p.bclc_stage or 'N/A'})"
        print(f"  {p.patient_id}: Stage {stage_info}, Age {p.age}")
    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(
        api_key=args.api_key,
        api_base=args.api_base
    )

    # Check for API key
    api_key = args.api_key or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        print("Warning: No API key provided. Set --api-key or CLAUDE_API_KEY env var.")
        print("Running in limited mode.\n")

    # Initialize LLM client if API key available
    if api_key:
        api_base = args.api_base or os.environ.get("CLAUDE_API_BASE", "https://rsxermu666.cn/v1")
        init_llm_client(api_key, api_base)
        print(f"LLM client initialized (model: {config.api.model_name})")
    else:
        print("Note: Running without LLM access")

    # Execute requested action
    if args.list_patients:
        list_patients()

    elif args.demo:
        run_demo()

    elif args.patient_id:
        if args.skip_llm:
            print("Note: --skip-llm mode not supported for patient assessment")
            print("Please provide valid API credentials or use --demo mode")
        else:
            workflow = HCCPrognosisWorkflow()
            report_text = run_patient_assessment(args.patient_id, workflow)

            if report_text and args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"\nReport saved to: {args.output}")

    else:
        # Default: run demo
        print("No specific action requested. Running demo...\n")
        run_demo()


if __name__ == "__main__":
    main()
