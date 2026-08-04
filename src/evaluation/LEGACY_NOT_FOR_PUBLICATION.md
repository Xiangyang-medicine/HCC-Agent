# Legacy Evaluation Modules - Not for Publication

The files `llm_agent_evaluation.py` and `agent_evaluator.py` are retained only for historical traceability.

They must not be imported by the canonical Phase 4 pipeline or used to generate paper results because they contain rule-weighted or mock risk scoring that is not an LLM-driven prognostic model. Quantitative prognosis in Phase 4 must come exclusively from a frozen Phase 3 prognostic-model tool.

The authoritative Phase 4 requirements are in:

`docs/PHASE_4_AGENT_EVALUATION_PROTOCOL.md` (version 3.0 or later).
