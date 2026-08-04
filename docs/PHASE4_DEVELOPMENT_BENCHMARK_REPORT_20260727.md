# Phase 4 live development benchmark report

**Date:** 2026-07-27  
**Status:** `DEVELOPMENT_COMPLETED; FORMAL_TEST_NOT_RUN`  
**Interpretation boundary:** These results are for prompt, parser, tool-contract, and retry-rule development only. They must not be reported as confirmatory paper results or used in the main Figure 2.

## 1. Frozen development inputs

- 20 development cases sampled from the frozen Phase 3A OOF file, disjoint from 100 reserved formal cases.
- Agent access to survival outcomes: false.
- 12 PubMed records retrieved from NCBI.
- 48 immutable sentence-level evidence passages.
- 96 deterministic development claim-passage pairs: 48 exact supported pairs and 48 cross-passage unsupported pairs.
- Requested and API-returned model: `claude-opus-4-8`.
- Temperature: 0.
- All systems received the same case request, prognostic tool, corpus, deterministic retrieval rule, and model version.

## 2. Development results

| System | Successful cases | Rate | Mean LLM calls | Mean latency | Total tokens |
|---|---:|---:|---:|---:|---:|
| B0 engine only | 0/20 | 0% | 0.0 | 0.004 s | 0 |
| B1 single LLM, no tools | 0/20 | 0% | 1.0 | 14.49 s | 46,374 |
| B2 tool-using single agent | 17/20 | 85% | 1.9 | 24.78 s | 98,086 |
| B3 multi-agent, no verifier/revision | 13/20 | 65% | 1.8 | 23.03 s | 91,459 |
| B4 full closed loop | 20/20 | 100% | 2.1 | 31.01 s | 110,457 |

The endpoint required the exact tool plan and arguments, both required tool calls in order, valid report structure, exact preservation and rendering of frozen model values, at least one extractively supported claim, correct citations, and no forbidden claim.

These percentages are descriptive development diagnostics. No p-value, confidence interval, superiority claim, or paper conclusion is permitted from this 20-case set.

## 3. Failure and recovery audit

- B2 failures: two invalid plans and one missing/invalid evidence-claim output.
- B3 failures: four invalid plans and three missing/invalid evidence-claim outputs.
- B4 planning revisions: one; recovered successfully.
- B4 synthesis revisions: two; both recovered successfully.
- B4 safe abstentions on clean development cases: zero.
- API errors: zero across 136 live model calls.
- Provider-returned model identifier matched the requested model in all 136 calls.
- Credentials serialized to outputs: false.

The B4 recovery behavior is observable in action traces. Empty or invalid plans are not replaced with a hidden hard-coded plan. B4 may perform one explicit replan; B2 and B3 do not receive that recovery mechanism.

## 4. Test status

- Canonical Phase 4 plus prognostic-engine supported suite: 138 passed, 5 skipped, 56 warnings.
- The repository also contains superseded legacy tests that depend on mock workflows, missing `langgraph`, and superseded Phase 3A v2 files. They are not part of the canonical publication pipeline and remain excluded rather than being used as publication evidence.

## 5. Formal-run scale estimate

Scaling the observed development usage to 100 formal cases, three repeated runs, and B0-B4 clean runs gives approximately:

- 2,040 live LLM calls;
- 5,195,640 total tokens;
- about two hours at the observed latency with four concurrent workers, subject to provider throttling.

Including the four required ablations and eight fault types (30 cases per fault, three systems, three repeats) increases the projected total to approximately 8,700 calls and 22.6 million tokens. This is an engineering estimate, not a guaranteed billable total.

This estimate is not a price quotation because the proxy's billing rates are not available in the API response. Formal execution requires explicit approval of this external-token expenditure.

## 6. Remaining formal gates

1. Freeze a formal evidence corpus and deterministic support benchmark separate from development assets.
2. Record a dated pre-result protocol amendment for the extractive claim-support contract.
3. Implement and test every prespecified fault type and the paired clean/fault scorer.
4. Hash code, prompts, schemas, tasks, corpus, model configuration, and environment.
5. Run the 100-case, three-repeat formal benchmark exactly once.
6. Only then generate the numerical main-text Figure 2.

## 7. Authoritative development outputs

- `experiments/phase4/development_20cases_v1/run_records.jsonl`
- `experiments/phase4/development_20cases_v1/case_level_metrics.csv`
- `experiments/phase4/development_20cases_v1/metrics_by_system.csv`
- `experiments/phase4/development_20cases_v1/DEVELOPMENT_GATE.json`
- `experiments/phase4/development_20cases_v1/run_manifest.json`
