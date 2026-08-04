"""Prespecified B4 ablations for the formal Phase 4 benchmark."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import json

from .llm_policy import StructuredLLMPlanningPolicy, StructuredLLMSynthesisPolicy, parse_json_object
from .orchestrator import ClosedLoopAgent, SystemVariant
from .schema import Claim, DraftReport, RunState
from .verifier import ReportVerifier


class AblationVariant(str, Enum):
    NO_EVIDENCE_CONTRACT = "B4_NO_EVIDENCE_CONTRACT"
    NO_VERIFIER = "B4_NO_VERIFIER"
    NO_REVISION_LOOP = "B4_NO_REVISION_LOOP"
    NO_PERSISTENT_STRUCTURED_STATE = "B4_NO_PERSISTENT_STRUCTURED_STATE"


class StatelessLLMSynthesisPolicy:
    """Ablation that asks the LLM to copy model values without a typed channel."""

    SYSTEM_PROMPT = """Return JSON only with keys `model_id`, `risk_score`,
`survival_probability_36m`, `claims`, and `narrative`. Every claim must copy an
exact supplied passage sentence and cite its passage_id. Copy numerical values
exactly. Do not make treatment recommendations."""

    def __init__(self, completion):
        self.completion = completion

    def create_draft(self, state: RunState, revision: bool) -> DraftReport:
        model = state.model_result
        evidence = state.evidence_result
        payload = {
            "revision": revision,
            "model": None if model is None else {
                "model_id": model.model_id,
                "risk_score": model.risk_score,
                "survival_probability_36m": model.survival_probabilities.get("36m"),
            },
            "passages": [] if evidence is None else [
                {"passage_id": p.passage_id, "text": p.text}
                for p in evidence.passages
            ],
        }
        try:
            parsed = parse_json_object(
                self.completion(self.SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
            )
            copied_model = None if model is None else replace(
                model,
                model_id=str(parsed.get("model_id", "")),
                risk_score=float(parsed.get("risk_score")),
                survival_probabilities={
                    **model.survival_probabilities,
                    "36m": float(parsed.get("survival_probability_36m")),
                },
            )
            claims = tuple(
                Claim(
                    text=str(item["text"]),
                    citation_ids=tuple(str(value) for value in item.get("citation_ids", [])),
                )
                for item in parsed.get("claims", [])
                if isinstance(item, dict) and "text" in item
            )
            narrative = str(parsed.get("narrative", ""))
        except Exception as exc:
            copied_model = None
            claims = ()
            narrative = f"Stateless synthesis failed: {type(exc).__name__}."
        if copied_model is None:
            report_text = narrative
        else:
            report_text = (
                f"Frozen model={copied_model.model_id}; risk_score={copied_model.risk_score:.12g}; "
                f"36-month survival probability={copied_model.survival_probabilities.get('36m', float('nan')):.12g}. "
                f"{narrative}"
            )
        return DraftReport(
            model_result=copied_model,
            claims=claims,
            report_text=report_text,
            status="STATELESS_ABLATION",
        )


def build_ablation_agent(
    variant: AblationVariant,
    prognostic_tool,
    evidence_tool,
    completion,
) -> ClosedLoopAgent:
    planner = StructuredLLMPlanningPolicy(completion)
    synthesis = StructuredLLMSynthesisPolicy(completion)
    verifier = ReportVerifier()
    system_variant = SystemVariant.B4_FULL_CLOSED_LOOP
    kwargs = {}

    if variant == AblationVariant.NO_EVIDENCE_CONTRACT:
        verifier = ReportVerifier(enforce_extractive_support=False)
    elif variant == AblationVariant.NO_VERIFIER:
        system_variant = SystemVariant.B3_MULTI_AGENT_NO_VERIFIER
    elif variant == AblationVariant.NO_REVISION_LOOP:
        kwargs = {
            "max_revisions": 0,
            "enable_plan_revision": False,
            "enable_evidence_retry": False,
            "enable_synthesis_revision": False,
        }
    elif variant == AblationVariant.NO_PERSISTENT_STRUCTURED_STATE:
        synthesis = StatelessLLMSynthesisPolicy(completion)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported ablation: {variant}")

    return ClosedLoopAgent(
        prognostic_tool=prognostic_tool,
        evidence_tool=evidence_tool,
        planning_policy=planner,
        synthesis_policy=synthesis,
        verifier=verifier,
        variant=system_variant,
        **kwargs,
    )
