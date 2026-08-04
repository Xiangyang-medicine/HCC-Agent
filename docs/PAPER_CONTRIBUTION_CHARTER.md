# ACM TIST Submission - Paper Contribution Charter

**Version**: 1.3
**Date**: 2026-07-24
**Status**: ACTIVE - Phase 3A closed; Phase 3B and Phase 4 empirical work pending

---

## 1. Paper Positioning

### 1.1 Submission Track
- **Journal**: ACM Transactions on Intelligent Systems and Technology (TIST)
- **Special Issue**: LLM-Driven Agentic AI for Intelligent Systems
- **Classification**: Technology + High-Stakes Application

This is NOT a survival analysis paper with LLM features. This is an LLM Multi-Agent Systems paper where the prognostic engine serves as a concrete, high-stakes instantiation of the architecture.

### 1.2 Core Theme

> **"A provenance-constrained, evidence-grounded and self-correcting LLM multi-agent system for reproducible biomedical prognostic research"**

The paper demonstrates that LLM agents can reliably assist in complex prognostic analysis while maintaining reproducibility guarantees through:
1. Strict separation between deterministic (model predictions) and stochastic (LLM reasoning) components
2. Evidence contracts that mandate source attribution for all claims
3. Self-correction mechanisms for handling prediction failures

### 1.3 Positioning Statement

**Target Problem**: Biomedical researchers conducting prognostic studies face information overload from genomic data, clinical literature, and historical cases. Traditional single-model approaches lack transparency and fail to explain predictions in terms of established evidence.

**Proposed Solution**: A closed-loop, LangGraph-based multi-agent system in which specialized components validate inputs, invoke a frozen prognostic-model tool, retrieve evidence, verify claims, revise failed outputs, and produce a traceable technical report. The prognostic engine provides quantitative predictions; the LLM layer provides observable planning, tool orchestration, evidence synthesis, verification, and safe abstention.

**What We Are NOT Claiming**:
- This is NOT a clinical decision support system
- We do NOT claim predictive validity for treatment selection
- We do NOT assert the system can replace clinician judgment
- We do NOT claim external validity until Phase 3B validation completes

---

## 2. Research Questions

### RQ1: System Architecture
**Can a multi-agent LLM architecture maintain reproducibility guarantees in complex biomedical reasoning tasks?**

- **Sub-RQ1.1**: Does the LLM/deterministic separation prevent LLM hallucinations from corrupting quantitative predictions?
- **Sub-RQ1.2**: Can fault injection demonstrate that the recovery mechanism catches prediction failures?
- **Sub-RQ1.3**: Do evidence contracts ensure that all claims are traceable to source material?

**Assessment Method**: Fault injection benchmark with automated recovery testing

### RQ2: Prognostic Tool Performance
**Can the integrated system achieve competitive prognostic discrimination compared to state-of-the-art survival models?**

- **Sub-RQ2.1**: Does M4 (combined Random Survival Forest) improve internal-validation performance over M1 (clinical-only Cox)?
- **Sub-RQ2.2**: Are the performance gains statistically significant after proper nested CV?
- **Sub-RQ2.3**: Does the system maintain performance on external validation cohorts?

**Assessment Method**: Nested 5×5×5 CV with IPCW-weighted metrics; Phase 3B external validation

### RQ3: Agentic Reliability
**Does the closed-loop multi-agent system improve verified task completion over a strong single-agent tool-using baseline?**

- **Sub-RQ3.1**: Does the system select and sequence tools more accurately?
- **Sub-RQ3.2**: Does the verifier-and-revision loop reduce unsupported claims and citation errors?
- **Sub-RQ3.3**: Does the system detect and recover from injected faults or abstain safely?

**Assessment Method**: Frozen, objective benchmark with deterministic gold action specifications, annotated claim-passage pairs, strong baselines, ablations, and patient-clustered inference

---

## 3. Core Innovations

### Innovation 1: Provenance-Constrained Architecture
The strict separation between deterministic prognostic engine (scikit-survival models) and stochastic LLM reasoning layer ensures that:
- Quantitative predictions are always traceable to trained models
- LLM outputs cannot retroactively modify prediction artifacts
- Failure modes are isolated and recoverable

**Implementation**: LangGraph state machine with typed channels separating "model_output" from "llm_reasoning"

### Innovation 2: Evidence Contracts
Every LLM-generated claim must be grounded in:
1. Explicit citation to source documents (PubMed, TCGA clinical data)
2. Confidence level assignment
3. Contradiction flagging when sources conflict

**Implementation**: Evidence extraction prompts with structured output validation

### Innovation 3: Self-Correction Mechanism
The proposed system includes:
1. Schema, provenance, numerical-fidelity, and evidence checks
2. Conditional revision within a fixed retry budget
3. Safe abstention when the quantitative model tool is unavailable or invalid
4. Explicit error reporting to users

**Implementation requirement**: A verifier node must be able to return control to planning or synthesis. Training-pipeline integrity checks alone do not constitute agent self-correction.

### Innovation 4: Fault Injection Benchmark
Prespecified evaluation methodology that:
1. Introduces controlled failures (missing data, model degradation)
2. Measures recovery success rate
3. Quantifies graceful degradation

**Implementation status**: Pending under Phase 4 protocol v3.0

### Application Component: Prespecified Metabolic Feature Panel for HCC
A 15-gene metabolic feature panel is evaluated as part of the prognostic models:
- Glycolysis: HK2, PKM, LDHA, LDHB, GPI, PFKL
- Glutamine: GLS, GLUD1
- Lipogenesis: FASN, SCD
- Hypoxia: CA9, VEGFA, HIF1A
- Oncogenic: MYC, CTNNB1

**Evidence**: TCGA-LIHC internal nested CV (n=363). This is not yet an externally validated signature and is not claimed as a standalone innovation.

---

## 4. Forbidden Claims

The following claims are **explicitly prohibited** until validated:

| Forbidden Claim | Reason | Required Validation |
|----------------|--------|---------------------|
| "The system can guide treatment decisions" | Regulatory, ethical concerns | FDA clearance or clinical trial |
| "M4 is clinically superior to M1" | Internal comparison was not significant after multiplicity adjustment | External validation and appropriate inferential evidence |
| "The model predicts survival accurately" | Survival models are prognostic, not predictive | Survival analysis terminology |
| "These results apply to all HCC patients" | Single-cohort derivation | Multi-cohort external validation |
| "The LLM understands patient prognosis" | Anthropomorphism concern | System attribution in all claims |

---

## 5. Permitted Claims

The following claims are **permitted** based on current evidence:

| Claim | Evidence Level | Framing |
|-------|---------------|---------|
| "M4 had the highest mean internal-validation discrimination in TCGA-LIHC" | Completed nested CV | Descriptive; M4 vs M1 was not significant after adjustment |
| "M4 is the provisional primary candidate for external validation" | Phase 3A selection rule and sensitivity analyses | Candidate-selection claim only |
| "The architecture requires claim-level provenance and safe abstention" | Frozen Phase 4 protocol | Design requirement, not an achieved performance result |

---

## 6. Metric Reporting Requirements

### 6.1 Internal Validation (Phase 3A)
Report per SAP v1.1:
- Harrell C-index (mean ± std across 25 folds)
- Uno C-index (IPCW-weighted)
- Time-dependent AUC at 12/36/60 months
- Integrated Brier Score
- Bootstrap comparison p-values (corrected for multiple testing)

**Publication Gate**: Only publish Phase 3A results AFTER integrity gate passes

### 6.2 External Validation (Phase 3B)
Required for any clinical relevance claim:
- ICGC-LIRI-JP cohort (n≈200)
- GEO GSE14520 cohort (n≈240)
- Performance degradation reporting

### 6.3 Objective Agent Benchmark (Phase 4)
Required for agent-system claims:
- verified case-level task success;
- tool selection and argument accuracy;
- claim support and citation correctness;
- verifier/revision ablations;
- fault detection, recovery, and safe abstention;
- test-retest reliability, latency, tokens, tool calls, and cost.

---

## 7. Status Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | COMPLETED | Data acquisition, preprocessing |
| Phase 2 | COMPLETED | TCGA-LIHC dataset verified |
| Phase 3A | **METHOD_CLOSURE_COMPLETED** | v6 comparisons and `AUDIT_REPORT_V5.json`; 112 passed, 0 failed |
| Phase 3B | **PROTOCOL_V2_FROZEN_DATA_ACQUISITION_PENDING** | M4 external-validation amendment frozen before external analysis |
| Phase 4 | **LIVE_DEVELOPMENT_COMPLETED_FORMAL_BENCHMARK_PENDING** | 20-case B0-B4 development benchmark complete; no formal performance claim yet |

## 7b. Phase 3A Final Status: **METHOD_CLOSURE_COMPLETED**

**Status as of 2026-07-24**: Formal 5x5 nested CV, sensitivity analyses, corrected patient-level comparisons, and final audit are complete.

> **Environment note**: The supported project environment is Python 3.12 in `.venv312`. The final suite completed with 112 passed, 5 skipped, 0 failed. The system Python 3.13 environment is not supported because of the known scikit-survival/scikit-learn incompatibility.

**Completed Issues** (verified by source code review):
1. [x] Remove performance-based integrity gates (C-index >= 0.50, AUC >= 0.50, IBS <= 0.50)
2. [x] Fix Uno C exception handling (no Harrell fallback, return NaN with logging)
3. [x] Fix tau handling (pre-define rules, record actual tau per fold)
4. [x] Fix AUC exception handling (record failures, mark NOT_ESTIMABLE)
5. [x] Delete manual Coxnet survival probability formula, use predict_survival_function
6. [x] Fix Brier/IBS: pass S(t) directly, not 1-S(t)
7. [x] Fix calibration: use KM with pre-defined bins, no fake slope
8. [x] Fix nested CV: preprocessing only on inner-training, not outer-training
9. [x] Fix Coxnet alpha path: dynamic generation, no zero-coefficient models
10. [x] Fix M1: no secret age-only fallback
11. [x] Fix M4/M5: real RSF and DeepSurv with proper inner splits
12. [x] Rewrite bootstrap: preserve patient multiplicity (verified correct)
13. [x] Create unit tests before any training (18 tests designed)
14. [x] Rewrite Phase 4 protocol v3: objective agent benchmark with baselines, ablations, and deterministic gold specifications

**Completed Issues** (verified by unit tests):
1. [x] Remove performance-based integrity gates (C-index >= 0.50, AUC >= 0.50, IBS <= 0.50)
2. [x] Fix Uno C exception handling (no Harrell fallback, return NaN with logging)
3. [x] Fix tau handling (pre-define rules, record actual tau per fold)
4. [x] Fix AUC exception handling (record failures, mark NOT_ESTIMABLE)
5. [x] Delete manual Coxnet survival probability formula, use predict_survival_function
6. [x] Fix Brier/IBS: pass S(t) directly, not 1-S(t)
7. [x] Fix calibration: use KM with pre-defined bins, no fake slope
8. [x] Fix nested CV: preprocessing only on inner-training, not outer-training
9. [x] Fix Coxnet alpha path: dynamic generation, no zero-coefficient models
10. [x] Fix M1: no secret age-only fallback
11. [x] Fix M4/M5: real RSF and DeepSurv with proper inner splits
12. [x] Rewrite bootstrap: preserve patient multiplicity (verified correct)
13. [x] Create unit tests before any training (18 tests passing)
14. [x] Freeze Phase 4 protocol v3 before implementation

**Next Steps**:
- Acquire and verify an external RNA-seq cohort under Phase 3B amendment v2
- Freeze the Phase 4 evidence corpus and live LLM configuration
- Run only the 20-case Phase 4 development set before freezing the formal benchmark

---

## 8. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-15 | Initial charter creation |
| 1.1 | 2026-07-15 | Phase 3A reset complete; Phase 3B/4 protocols created |
| 1.2 | 2026-07-15 | Phase 3A ALL ISSUES RESOLVED; Phase 4 rewritten with objective benchmarks |
| 1.3 | 2026-07-24 | Phase 3A v6 statistical closure; Phase 4 protocol v3 frozen; old rule-based Agent evaluators excluded |
| 1.4 | 2026-07-27 | Phase 4 canonical architecture smoke passed; Phase 3B M4 amendment v2 frozen |

## 9. Reference Documents

| Document | Location |
|----------|----------|
| Phase 3A Statistical Analysis Plan | docs/PHASE_3A_SAP.md |
| Phase 3B Validation Protocol | docs/PHASE_3B_VALIDATION_PROTOCOL.md |
| Phase 4 Agent Evaluation Protocol | docs/PHASE_4_AGENT_EVALUATION_PROTOCOL.md |
| TCGA-LIHC Data Verification | data/processed/gdc/20260713/VERIFICATION_REPORT.md |

---

*This charter is binding for all paper drafting. Any deviation requires explicit review and approval.*
