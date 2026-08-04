"""
Final workflow test - verify full execution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.state.schema import AgentState, PatientData
from src.workflow import get_workflow
from src.utils.llm_client import init_llm_client
from config.config import load_config

print("Initializing...", flush=True)
config = load_config()
if config.api.api_key:
    init_llm_client(config.api.api_key, config.api.api_base)

patient = PatientData(
    patient_id="TCGA-TEST-001",
    age=62,
    gender="M",
    stage="T2N0M0",
    grade="G2",
    bclc_stage="A",
    afp_level=350.0,
    gene_expression={"CA9": 3.5, "VEGFA": 4.2, "HK2": 4.0, "PKM2": 3.8, "LDHA": 4.1}
)

print(f"Running assessment for patient {patient.patient_id}...", flush=True)
workflow = get_workflow()

import time
start = time.time()

# Run with invoke
result_dict = workflow.invoke(AgentState(patient_data=patient))
result = AgentState(**result_dict)

elapsed = time.time() - start
print(f"\nWorkflow completed in {elapsed:.1f}s", flush=True)

# Show results
print("\n=== RESULTS ===", flush=True)
if result.risk_assessment:
    print(f"Risk Level: {result.risk_assessment.risk_level.value}", flush=True)
    print(f"Confidence: {result.risk_assessment.confidence_score:.0%}", flush=True)
    if result.risk_assessment.estimated_survival_months:
        print(f"Est. Survival: {result.risk_assessment.estimated_survival_months:.1f} months", flush=True)

if result.final_report:
    print(f"\nExecutive Summary:\n{result.final_report.executive_summary[:500]}...", flush=True)

print(f"\nCompleted tasks: {result.completed_tasks}", flush=True)
print("SUCCESS!", flush=True)
