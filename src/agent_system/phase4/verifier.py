"""Deterministic verifier for observable Phase 4 outputs."""

from __future__ import annotations

import math

from .schema import DraftReport, EvidenceResult, ModelResult, VerificationFinding, VerificationResult


FORBIDDEN_PHRASES = (
    "treatment recommendation",
    "should receive",
    "clinically deployable",
    "replace clinician",
    "autonomous diagnosis",
)


class ReportVerifier:
    """Verify provenance, numeric fidelity, citations, and safety boundaries."""

    def __init__(
        self,
        enforce_extractive_support: bool = True,
        enforce_metadata_consistency: bool = True,
        enforce_conflict_detection: bool = True,
    ):
        self.enforce_extractive_support = enforce_extractive_support
        self.enforce_metadata_consistency = enforce_metadata_consistency
        self.enforce_conflict_detection = enforce_conflict_detection

    def verify(
        self,
        draft: DraftReport,
        model_result: ModelResult | None,
        evidence_result: EvidenceResult | None,
    ) -> VerificationResult:
        findings: list[VerificationFinding] = []

        if model_result is None or model_result.risk_score is None:
            findings.append(VerificationFinding("MODEL_RESULT_MISSING", "Model result is missing."))
        elif not math.isfinite(model_result.risk_score):
            findings.append(VerificationFinding("MODEL_RESULT_NONFINITE", "Model risk score is not finite."))

        if draft.model_result != model_result:
            findings.append(
                VerificationFinding(
                    "MODEL_FIDELITY_FAILURE",
                    "The draft does not preserve the exact tool-returned model result.",
                )
            )

        if model_result is not None and model_result.risk_score is not None:
            expected_risk = f"{model_result.risk_score:.12g}"
            expected_36m = f"{model_result.survival_probabilities.get('36m', float('nan')):.12g}"
            if model_result.model_id not in draft.report_text:
                findings.append(
                    VerificationFinding("MODEL_ID_NOT_RENDERED", "Model identifier is absent from the report.")
                )
            if expected_risk not in draft.report_text or expected_36m not in draft.report_text:
                findings.append(
                    VerificationFinding(
                        "MODEL_VALUE_NOT_RENDERED",
                        "The report does not render the exact deterministic model values.",
                    )
                )

        valid_citations = set()
        passage_by_id = {}
        if evidence_result is not None:
            valid_citations = {passage.passage_id for passage in evidence_result.passages}
            passage_by_id = {passage.passage_id: passage.text for passage in evidence_result.passages}
            conflict_groups: dict[str, set[str]] = {}
            for passage in evidence_result.passages:
                pmid = passage.metadata.get("pmid", "")
                source_url = passage.metadata.get("source_url", "")
                if self.enforce_metadata_consistency and pmid and pmid not in source_url:
                    findings.append(
                        VerificationFinding(
                            "EVIDENCE_METADATA_MISMATCH",
                            f"Passage {passage.passage_id} has PMID/source URL mismatch.",
                        )
                    )
                conflict_group = passage.metadata.get("conflict_group", "")
                stance = passage.metadata.get("stance", "")
                if conflict_group and stance:
                    conflict_groups.setdefault(conflict_group, set()).add(stance.upper())
            for conflict_group, stances in conflict_groups.items():
                if self.enforce_conflict_detection and {"SUPPORTS", "REFUTES"}.issubset(stances):
                    findings.append(
                        VerificationFinding(
                            "CONFLICTING_EVIDENCE_UNRESOLVED",
                            f"Retrieved evidence conflict {conflict_group!r} is unresolved.",
                        )
                    )

        for claim in draft.claims:
            if claim.kind == "biomedical_context" and not claim.citation_ids:
                findings.append(
                    VerificationFinding("CLAIM_MISSING_CITATION", f"Claim has no citation: {claim.text}")
                )
            for citation_id in claim.citation_ids:
                if citation_id not in valid_citations:
                    findings.append(
                        VerificationFinding(
                            "INVALID_CITATION",
                            f"Citation {citation_id!r} is not in the retrieved evidence.",
                        )
                    )
            if self.enforce_extractive_support and claim.citation_ids and not any(
                claim.text.strip() in passage_by_id.get(citation_id, "")
                for citation_id in claim.citation_ids
            ):
                findings.append(
                    VerificationFinding(
                        "CLAIM_NOT_EXTRACTIVELY_SUPPORTED",
                        "Claim text is not an exact sentence from any cited passage.",
                    )
                )

        if evidence_result is not None and evidence_result.passages and not draft.claims:
            findings.append(
                VerificationFinding(
                    "NO_EVIDENCE_CLAIMS",
                    "Retrieved evidence exists but the draft contains no evidence-grounded claim.",
                )
            )

        report_lower = draft.report_text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in report_lower:
                findings.append(
                    VerificationFinding("FORBIDDEN_CLAIM", f"Forbidden phrase found: {phrase}"))

        return VerificationResult(passed=not findings, findings=tuple(findings))
