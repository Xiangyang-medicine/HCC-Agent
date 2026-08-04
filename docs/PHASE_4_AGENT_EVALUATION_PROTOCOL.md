# Phase 4 LLM Agent System and Evaluation Protocol

**Version:** 3.0  
**Date:** 2026-07-24  
**Status:** FROZEN BEFORE PHASE 4 IMPLEMENTATION  
**Prerequisites:** Phase 3A statistical closure completed; Phase 3B external validation may proceed independently  
**Scope:** Technical agent-system evaluation without physician participants

---

## 1. Publication Rationale

The ACM TIST special issue requires more than an LLM-generated narrative around a predictive model. The contribution must demonstrate a closed-loop agentic system that selects and invokes tools, maintains explicit state, verifies its own outputs, recovers from failures, and is evaluated reproducibly.

Phase 4 therefore treats the prognostic engine as one deterministic tool inside an LLM-driven agent system. Model discrimination is a Phase 3 result and must not be attributed to the LLM. Phase 4 evaluates agent behavior, not survival-model performance.

The following legacy modules are excluded from all Phase 4 quantitative results:

- `src/evaluation/llm_agent_evaluation.py`
- `src/evaluation/agent_evaluator.py`
- any mock, rule-weighted, or hard-coded risk score

These modules may be retained only for historical traceability. They are not an implementation of the Phase 4 system and are not publication evidence.

---

## 2. Research Questions

### RQ4.1 - Tool orchestration

Can the agent select, invoke, and sequence the required deterministic tools for an HCC prognosis-assessment task?

### RQ4.2 - Evidence grounding

Can the agent restrict biomedical claims to retrieved evidence and produce citations that support the associated claims?

### RQ4.3 - Verification and self-correction

Does a verifier-and-revision loop detect and repair unsupported claims, schema violations, and tool failures?

### RQ4.4 - Reliability under realistic faults

Does the system fail safely and preserve traceability when inputs, retrieval, model tools, or APIs are degraded?

No RQ concerns physician trust, clinical utility, treatment benefit, or clinical deployment.

---

## 3. System Under Test

### 3.1 Required Components

| Component | Responsibility | Observable output |
|---|---|---|
| Coordinator | Convert a request into an explicit task plan and route tool calls | Structured plan and action trace |
| Data validator | Validate input schema, missingness, units, and allowed feature set | Validation result and warnings |
| Prognostic-model tool | Return frozen model identifier, risk score, survival estimates, and provenance | Typed model result |
| Evidence retriever | Search only the frozen evidence corpus and return passage-level evidence | Source IDs, passage IDs, retrieval scores |
| Evidence synthesizer | Generate claim-citation pairs from retrieved passages | Structured claims |
| Verifier | Check schema, numerical consistency, citation support, and forbidden claims | Pass/fail findings with reason codes |
| Revision controller | Re-plan or revise after a verifier or tool failure, subject to a fixed retry budget | Revision trace |
| Report renderer | Produce the final technical report from verified structured state | JSON plus human-readable report |

### 3.2 Closed-Loop State Machine

`VALIDATE -> PLAN -> CALL TOOLS -> VERIFY -> {REVISE | REPORT | SAFE ABSTAIN}`

The verifier must be able to return control to planning or synthesis. A fixed linear sequence without conditional recovery does not satisfy this protocol.

### 3.3 Trace Policy

The system records structured actions, tool inputs and outputs, timestamps, errors, verifier decisions, and retry counts. It must not request, store, or evaluate private chain-of-thought. Evaluation is based only on observable actions and structured outputs.

### 3.4 Separation of Responsibilities

- The prognostic engine supplies quantitative risk estimates.
- The LLM supplies planning, tool selection, evidence synthesis, verification, and reporting.
- The LLM must not alter a model risk score, invent a replacement score, or reverse its direction.
- If the model tool is unavailable or an input is invalid, the system must abstain from quantitative prognosis.

---

## 4. Frozen Evaluation Assets

### 4.1 Patient Cases

- 100 de-identified TCGA-LIHC cases selected from frozen out-of-fold predictions.
- Sampling is stratified by repeat-independent model-risk quintile and event status.
- Only one evaluation representation of each `case_id` is used in an analysis unit.
- Outcomes are withheld from the agent and used only for Phase 3 performance analyses, not as an agent-quality label.

### 4.2 Task Set

Each case is assigned the same core request: validate the input, obtain the frozen prognostic result, retrieve relevant evidence, verify the synthesis, and produce a technical non-treatment report.

Additional deterministic tasks test:

1. invalid or missing clinical fields;
2. missing gene features;
3. conflicting retrieved passages;
4. an unsupported user-requested claim;
5. a transient retrieval timeout;
6. a permanent model-tool failure;
7. malformed tool output;
8. citation metadata mismatch.

### 4.3 Evidence Corpus

- Freeze source files, version, retrieval index, passage boundaries, and SHA-256 hashes before the formal run.
- Store PMID, DOI, title, publication year, source URL, and passage ID.
- Use only sources whose metadata can be independently verified.
- Create an author-annotated benchmark of at least 200 claim-passage pairs with labels `SUPPORTED`, `NOT_SUPPORTED`, or `CONFLICTING`.
- An automated NLI score may be reported as a secondary diagnostic but cannot be the sole reference standard.

### 4.4 Gold Action Specifications

For each task type, freeze:

- required tools;
- prohibited tools;
- required order constraints;
- expected error or abstention code;
- required output fields;
- maximum retry budget.

These specifications provide deterministic labels for tool routing, failure recovery, and report-schema evaluation.

---

## 5. Comparator Systems and Ablations

All systems use the same LLM model/version, prompts where applicable, evidence corpus, cases, and tool implementations.

| ID | System | Purpose |
|---|---|---|
| B0 | Prognostic engine only | Non-LLM quantitative reference; not an agent |
| B1 | Single LLM, no tools | Measures unsupported generation without grounding |
| B2 | Single LLM with model and retrieval tools | Strong tool-using single-agent baseline |
| B3 | Multi-agent workflow without verifier/revision | Isolates verification and self-correction |
| B4 | Full closed-loop multi-agent system | Proposed system |

Required B4 ablations:

- no evidence contract;
- no verifier;
- no revision loop;
- no persistent structured state.

The paper must not claim a multi-agent advantage unless B4 outperforms the strongest appropriate baseline on the prespecified primary endpoint.

---

## 6. Endpoints

### 6.1 Primary Endpoint

**Case-level verified task success rate**: a case passes only when all required tool calls are correct, the final schema is valid, numeric model values are unchanged, every externally verifiable biomedical claim is supported, no forbidden claim is present, and the system reaches either a verified report or the prespecified safe-abstention state.

### 6.2 Secondary Endpoints

| Domain | Metric | Definition |
|---|---|---|
| Planning | Plan validity | Required steps and constraints present |
| Tool use | Tool selection precision/recall/F1 | Compared with frozen gold action specifications |
| Tool use | Argument accuracy | Exact or schema-based match of required arguments |
| Model fidelity | Numeric fidelity rate | Exact preservation within predefined floating tolerance |
| Grounding | Supported-claim precision | Supported generated claims / all verifiable generated claims |
| Grounding | Citation completeness | Supported claims with valid passage citations / supported claims |
| Grounding | Citation correctness | Citations whose passage supports the linked claim / all citations |
| Safety | Unsupported claim rate | Unsupported verifiable claims / all verifiable claims |
| Safety | Forbidden-claim rate | Cases containing any prohibited clinical claim |
| Recovery | Failure detection rate | Injected failures correctly detected / failures injected |
| Recovery | Recovery success rate | Failures ending in correct recovery or safe abstention / failures injected |
| Reliability | Test-retest agreement | Agreement of pass/fail and tool sequence across repeated runs |
| Efficiency | Latency, token use, tool calls, estimated cost | Reported by system and task type |

Keyword counts, response length, and generic medical terminology are descriptive diagnostics only and cannot define task success.

### 6.3 Forbidden Evaluations

- Pearson correlation between predicted risk and observed survival time in censored data.
- Using the same LLM as both generator and sole correctness judge.
- Treating parseable JSON as evidence of medical correctness.
- Flipping or tuning risk scores based on evaluation performance.
- Selecting prompts, thresholds, seeds, or baselines after viewing formal test results.

---

## 7. Formal Experimental Design

### 7.1 Development and Test Separation

- Use a development set of 20 cases to debug prompts and tools.
- Freeze code, prompts, schemas, evidence corpus, gold specifications, model version, and decoding configuration.
- Evaluate once on a disjoint 100-case formal test set.
- Development cases and formal test cases must have no overlapping `case_id`.

### 7.2 Repeated Runs

- Run each formal case three times with prespecified seeds when supported.
- If the API does not guarantee seeding, record the limitation and exact request parameters.
- Preserve run-level outputs; use the patient case as the bootstrap cluster.

### 7.3 Failure Injection

Each fault type is applied to at least 30 distinct cases. Fault assignment is randomized with a stored seed and balanced across risk strata. Clean and faulted variants of the same case are paired.

### 7.4 Blinding

- The agent cannot access survival outcomes.
- Gold action specifications and claim-support labels are unavailable to the generation path.
- Formal aggregate metrics are not computed until all frozen configurations complete.

---

## 8. Statistical Analysis

### 8.1 Primary Comparison

Compare B4 with B2 on case-level verified task success using a paired test across the same cases. Report:

- absolute percentage-point difference;
- patient-clustered bootstrap 95% confidence interval with 2000 resamples;
- two-sided paired permutation p-value;
- effect size and raw counts.

### 8.2 Secondary Comparisons

- Binary paired endpoints: McNemar test with paired effect and confidence interval.
- Continuous paired endpoints: patient-clustered bootstrap of the mean or median difference.
- Test-retest reliability: agreement and Fleiss kappa or an explicitly justified alternative.
- B4 ablations: paired comparisons with Holm correction within the ablation family.

### 8.3 Multiplicity

The B4 versus B2 primary endpoint is the sole confirmatory agent comparison. All other agent comparisons are secondary or exploratory and must be labelled accordingly.

### 8.4 Missing or Failed Runs

- A system crash, invalid output, exhausted retry budget, or missing final state is a task failure, not a missing observation.
- API-wide outages affecting all systems are documented and rerun under the same frozen configuration.
- No failed case may be silently removed.

---

## 9. Integrity Gates

Formal evaluation cannot start unless all gates pass:

1. no legacy rule-based risk evaluator is imported by the canonical Phase 4 entry point;
2. prognostic tool uses a frozen Phase 3 model artifact and emits its hash;
3. prompts, schemas, code, corpus, task set, and gold specifications are hashed;
4. all baselines and ablations pass smoke tests;
5. clean-run and every fault-path unit test pass;
6. action traces contain no private chain-of-thought fields;
7. formal test case IDs do not overlap development IDs;
8. the evaluation script treats crashes and invalid outputs as failures;
9. no performance threshold blocks or alters generated results;
10. mock outputs are excluded from the formal directory.

The gate is machine-readable and uses native JSON booleans.

---

## 10. Required Outputs

### Tables

1. System configurations, LLM version, tools, and frozen hashes.
2. Primary and secondary metrics for B1-B4.
3. Paired B4 versus B2 statistical comparison.
4. Ablation results.
5. Failure-injection detection and recovery results.
6. Error taxonomy with denominators.
7. Latency, token, tool-call, and cost analysis.

### Figures

1. Closed-loop architecture and state transitions.
2. Primary task-success comparison with confidence intervals.
3. Tool-use error decomposition.
4. Fault-injection recovery matrix.
5. Supported-claim precision and unsupported-claim rate by system.

### Reproducibility Package

- run manifest and environment lock;
- prompts and output schemas;
- frozen task specifications;
- corpus manifest and hashes, subject to source licensing;
- raw structured action traces without private reasoning;
- scoring code and unit tests;
- aggregate results and audit report.

---

## 11. Interpretation Rules

Allowed claims depend on observed evidence:

- If B4 improves the primary endpoint over B2, the paper may claim improved verified task completion under the tested benchmark.
- If the verifier ablation worsens grounding or recovery, the paper may attribute that measured effect to the verifier-and-revision mechanism.
- If B4 does not improve over B2, the paper must report that multi-agent decomposition did not provide measurable benefit under the benchmark.

Prohibited claims include clinical utility, treatment recommendations, physician acceptance, improved patient outcomes, autonomous diagnosis, or deployment readiness.

---

## 12. Relationship to the Paper

The publishable contribution is the combination of:

1. a frozen survival-model tool with strict provenance;
2. a closed-loop LLM agent architecture with explicit verification and safe abstention;
3. a reproducible benchmark covering tool orchestration, evidence grounding, self-correction, and fault recovery;
4. controlled baseline and ablation evidence.

External validation of the prognostic tool strengthens the healthcare application but does not replace the required agent-system contribution.

---

## 13. Revision History

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-15 | Initial physician-based concept |
| 2.0 | 2026-07-15 | Replaced physician evaluation with superficial automated metrics |
| 3.0 | 2026-07-24 | Reframed as a closed-loop agent benchmark with deterministic gold specifications, strong baselines, ablations, clustered inference, and fault recovery |

---

**Freeze rule:** Any change after a formal Phase 4 result is viewed requires a dated amendment and a clear confirmatory-versus-exploratory label.
