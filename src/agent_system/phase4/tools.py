"""Deterministic tools used by the Phase 4 agent system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from .schema import (
    EvidencePassage,
    EvidenceResult,
    ModelResult,
    TaskRequest,
    ToolStatus,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PrognosticTool(Protocol):
    def predict(self, request: TaskRequest) -> ModelResult: ...


class EvidenceTool(Protocol):
    def retrieve(self, request: TaskRequest) -> EvidenceResult: ...


class FrozenOOFPrognosticTool:
    """Read a frozen out-of-fold Phase 3A prediction without model fitting.

    This is intentionally evaluation-only.  OOF values are appropriate for
    Phase 4 benchmark cases but cannot be represented as a deployable patient
    model or used for new patients.
    """

    PROVENANCE = "PHASE3A_FROZEN_OOF_EVALUATION_ONLY"

    def __init__(self, oof_path: Path):
        self.oof_path = Path(oof_path)
        self.source_sha256 = sha256_file(self.oof_path)
        self._predictions = pd.read_csv(self.oof_path)
        required = {
            "case_id",
            "repeat",
            "model",
            "risk_score",
            "survival_probability_12m",
            "survival_probability_36m",
            "survival_probability_60m",
        }
        missing = required - set(self._predictions.columns)
        if missing:
            raise ValueError(f"OOF file missing required columns: {sorted(missing)}")

    def predict(self, request: TaskRequest) -> ModelResult:
        rows = self._predictions.loc[
            (self._predictions["case_id"] == request.case_id)
            & (self._predictions["repeat"] == request.repeat)
            & (self._predictions["model"] == request.requested_model)
        ]
        if len(rows) != 1:
            return ModelResult(
                status=ToolStatus.NOT_FOUND,
                case_id=request.case_id,
                repeat=request.repeat,
                model_id=request.requested_model,
                risk_score=None,
                survival_probabilities={},
                source_sha256=self.source_sha256,
                provenance=self.PROVENANCE,
                message=f"Expected one frozen OOF row, found {len(rows)}.",
            )
        row = rows.iloc[0]
        return ModelResult(
            status=ToolStatus.SUCCESS,
            case_id=request.case_id,
            repeat=int(row["repeat"]),
            model_id=str(row["model"]),
            risk_score=float(row["risk_score"]),
            survival_probabilities={
                "12m": float(row["survival_probability_12m"]),
                "36m": float(row["survival_probability_36m"]),
                "60m": float(row["survival_probability_60m"]),
            },
            source_sha256=self.source_sha256,
            provenance=self.PROVENANCE,
        )


class StaticEvidenceTool:
    """A frozen, passage-level corpus for deterministic smoke tests.

    Formal evaluation must replace this fixture with a versioned corpus
    manifest and author-annotated claim-passage benchmark.
    """

    def __init__(self, passages: list[EvidencePassage]):
        self._passages = tuple(passages)
        payload = "\n".join(
            f"{p.source_id}|{p.passage_id}|{p.text}" for p in self._passages
        ).encode("utf-8")
        self.corpus_sha256 = hashlib.sha256(payload).hexdigest()

    def retrieve(self, request: TaskRequest) -> EvidenceResult:
        if not self._passages:
            return EvidenceResult(
                status=ToolStatus.UNAVAILABLE,
                passages=(),
                corpus_sha256=self.corpus_sha256,
                message="No evidence passages are available.",
            )
        return EvidenceResult(
            status=ToolStatus.SUCCESS,
            passages=self._passages,
            corpus_sha256=self.corpus_sha256,
        )


class FrozenEvidenceCorpusTool:
    """Load a versioned JSONL corpus and retrieve a deterministic passage set.

    Retrieval is intentionally outcome-blind and stable across comparator
    systems.  The same ``case_id`` and frozen corpus always yield the same
    passages, preventing a comparator from receiving easier evidence.
    """

    def __init__(self, corpus_path: Path, top_k: int = 3):
        self.corpus_path = Path(corpus_path)
        self.corpus_sha256 = sha256_file(self.corpus_path)
        rows = []
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                try:
                    rows.append(EvidencePassage(
                        source_id=str(payload["source_id"]),
                        passage_id=str(payload["passage_id"]),
                        text=str(payload["text"]),
                        metadata={str(k): str(v) for k, v in payload["metadata"].items()},
                    ))
                except KeyError as exc:
                    raise ValueError(
                        f"Corpus row {line_number} is missing {exc.args[0]!r}."
                    ) from exc
        if not rows:
            raise ValueError("Frozen evidence corpus contains no passages.")
        if top_k < 1 or top_k > len(rows):
            raise ValueError("top_k must be between 1 and the corpus size.")
        passage_ids = [row.passage_id for row in rows]
        if len(passage_ids) != len(set(passage_ids)):
            raise ValueError("Frozen evidence corpus contains duplicate passage IDs.")
        self._passages = tuple(rows)
        self.top_k = top_k

    def retrieve(self, request: TaskRequest) -> EvidenceResult:
        seed = hashlib.sha256(
            f"{request.case_id}|{request.repeat}|{self.corpus_sha256}".encode("utf-8")
        ).digest()
        start = int.from_bytes(seed[:8], "big") % len(self._passages)
        selected = tuple(
            self._passages[(start + offset) % len(self._passages)]
            for offset in range(self.top_k)
        )
        return EvidenceResult(
            status=ToolStatus.SUCCESS,
            passages=selected,
            corpus_sha256=self.corpus_sha256,
        )


class FailingEvidenceTool:
    """Controlled timeout/unavailability fixture for fault-injection tests."""

    def __init__(self, message: str = "Simulated retrieval timeout"):
        self.message = message

    def retrieve(self, request: TaskRequest) -> EvidenceResult:
        return EvidenceResult(
            status=ToolStatus.UNAVAILABLE,
            passages=(),
            corpus_sha256="FAULT_INJECTION",
            message=self.message,
        )
