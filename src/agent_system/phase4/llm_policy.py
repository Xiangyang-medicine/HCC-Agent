"""Structured LLM adapter for the canonical Phase 4 system.

The adapter is optional and is not used by offline tests.  It accepts an
injected callable so the benchmark can freeze a provider, model version,
decoding parameters, and request trace without coupling the core agent to one
vendor.  LLM output cannot overwrite a deterministic model result.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from typing import Callable, Protocol

from .schema import ActionPlan, Claim, DraftReport, RunState, TaskRequest


class JSONCompletionCallable(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


def parse_json_object(raw: str) -> dict:
    """Parse a JSON object even when a provider wraps it in prose/fences.

    The original response is not modified or re-prompted.  We accept the first
    syntactically valid JSON object and reject arrays or scalar values.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No valid JSON object found", raw, 0)


class OpenAICompatibleJSONCallable:
    """Minimal OpenAI-compatible transport, configured only at run time."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.call_records: list[dict[str, object]] = []
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local optional dependency
            raise RuntimeError("Install the OpenAI-compatible client before live Phase 4 runs.") from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleJSONCallable":
        api_key = os.environ.get("PHASE4_LLM_API_KEY")
        model = os.environ.get("PHASE4_LLM_MODEL")
        base_url = os.environ.get("PHASE4_LLM_BASE_URL")
        if not api_key or not model:
            raise RuntimeError(
                "PHASE4_LLM_API_KEY and PHASE4_LLM_MODEL are required for live Phase 4 development runs."
            )
        return cls(api_key=api_key, base_url=base_url, model=model)

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            self.call_records.append({
                "status": "ERROR",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_type": type(exc).__name__,
            })
            raise
        usage = getattr(response, "usage", None)
        self.call_records.append({
            "status": "SUCCESS",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "model_returned": getattr(response, "model", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "request_id": getattr(response, "_request_id", None),
            "response_sha256": hashlib.sha256(
                (response.choices[0].message.content or "{}").encode("utf-8")
            ).hexdigest(),
        })
        return response.choices[0].message.content or "{}"

    def close(self) -> None:
        """Release the underlying HTTP connection pool deterministically."""
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleJSONCallable":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class StructuredLLMSynthesisPolicy:
    """Convert JSON-only LLM evidence synthesis into a verifiable draft."""

    SYSTEM_PROMPT = """You are the synthesis component of a research-only agent system.
Return JSON only with keys `claims` and `narrative`.
Each claim is an object with exactly `text` and `citation_ids`.
Use only supplied passage IDs. Every claim text must be copied exactly from one
supplied passage as a contiguous sentence; do not paraphrase it. When one or
more passages are supplied, return between one and three such claims. If a
revision is requested after `NO_EVIDENCE_CLAIMS`, add at least one exact claim
from the supplied passages. Do not make treatment recommendations. Do not
state or modify numerical model predictions: those are rendered separately by
a deterministic component. If evidence is insufficient, return an empty claim
list and explain the limitation in `narrative`."""

    def __init__(self, completion: JSONCompletionCallable):
        self.completion = completion

    @staticmethod
    def _canonical_model_section(state: RunState) -> str:
        model = state.model_result
        if model is None or model.risk_score is None:
            return "Model result unavailable."
        return (
            "Technical research report only. "
            f"Frozen model={model.model_id}; risk_score={model.risk_score:.12g}; "
            f"36-month survival probability={model.survival_probabilities.get('36m', float('nan')):.12g}. "
            "No treatment advice is included."
        )

    def create_draft(self, state: RunState, revision: bool) -> DraftReport:
        evidence = state.evidence_result
        passages = [] if evidence is None else [
            {
                "passage_id": passage.passage_id,
                "source_id": passage.source_id,
                "text": passage.text,
            }
            for passage in evidence.passages
        ]
        payload = {
            "task": state.request.query,
            "revision": revision,
            "available_passages": passages,
            "prior_verification_findings": []
            if state.verification is None
            else [finding.code for finding in state.verification.findings],
        }
        try:
            raw = self.completion(self.SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
            parsed = parse_json_object(raw)
            claim_objects = parsed.get("claims", [])
            if not isinstance(claim_objects, list):
                raise ValueError("claims must be a list")
            claims = tuple(
                Claim(
                    text=str(item["text"]),
                    citation_ids=tuple(str(value) for value in item.get("citation_ids", [])),
                )
                for item in claim_objects
                if isinstance(item, dict) and "text" in item
            )
            narrative = str(parsed.get("narrative", ""))
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            claims = ()
            narrative = f"Structured synthesis unavailable: {type(exc).__name__}."

        return DraftReport(
            model_result=state.model_result,
            claims=claims,
            report_text=f"{self._canonical_model_section(state)} {narrative}".strip(),
        )


class StructuredLLMPlanningPolicy:
    """Observable JSON planner; no private reasoning is requested or stored."""

    SYSTEM_PROMPT = """You are the planning component of a research-only agent system.
Return JSON only with keys `steps` and `arguments`.
The task requires exactly these tool steps in this order:
1. prognostic_tool
2. evidence_tool
Arguments must contain the supplied case_id, repeat, and requested_model exactly.
Do not add explanation, hidden reasoning, or additional tools."""

    def __init__(self, completion: JSONCompletionCallable):
        self.completion = completion

    def _create_plan(self, request: TaskRequest, revision_reason: str | None = None) -> ActionPlan:
        payload = {
            "task_id": request.task_id,
            "case_id": request.case_id,
            "repeat": request.repeat,
            "requested_model": request.requested_model,
            "request": request.query,
            "revision_reason": revision_reason,
        }
        try:
            raw = self.completion(self.SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
            parsed = parse_json_object(raw)
            raw_steps = parsed.get("steps", [])
            arguments = parsed.get("arguments", {})
            if not isinstance(raw_steps, list) or not isinstance(arguments, dict):
                raise ValueError("Planner fields have invalid types.")
            steps: list[str] = []
            step_arguments: list[dict] = []
            for item in raw_steps:
                if isinstance(item, str):
                    steps.append(item)
                elif isinstance(item, dict) and "tool" in item:
                    steps.append(str(item["tool"]))
                    if isinstance(item.get("arguments"), dict):
                        step_arguments.append(item["arguments"])
                else:
                    raise ValueError("Planner step must be a tool name or tool object.")
            # Some OpenAI-compatible providers attach the same arguments to
            # each step instead of returning the requested top-level field.
            # Normalize only when every observed argument dictionary agrees.
            if not arguments and step_arguments:
                first = step_arguments[0]
                if all(candidate == first for candidate in step_arguments):
                    arguments = first
            return ActionPlan(
                steps=tuple(steps),
                arguments={str(key): value for key, value in arguments.items()},
                status="LLM_PROPOSED",
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return ActionPlan(
                steps=(),
                arguments={},
                status=f"MALFORMED_{type(exc).__name__}",
            )

    def create_plan(self, request: TaskRequest) -> ActionPlan:
        return self._create_plan(request)

    def revise_plan(self, request: TaskRequest, finding: str) -> ActionPlan:
        return self._create_plan(request, revision_reason=finding)
