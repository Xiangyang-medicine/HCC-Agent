"""Prepare the disjoint, 200-pair formal evidence benchmark.

No LLM is used. Supported labels are exact sentence identity; unsupported
labels are deterministic cross-source mismatches. This script must run before
any reserved formal-case evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_phase4_development_assets import (
    NCBI_EFETCH,
    PMIDS as DEVELOPMENT_PMIDS,
    parse_records,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "phase4_evidence"
READINESS_DIR = ROOT / "experiments" / "phase4" / "readiness"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
QUERY = (
    '("hepatocellular carcinoma"[Title/Abstract]) AND '
    '(metabolism[Title/Abstract] OR metabolic[Title/Abstract] OR glycolysis[Title/Abstract]) AND '
    '(prognosis[Title/Abstract] OR prognostic[Title/Abstract] OR survival[Title/Abstract])'
)


def request_bytes(url: str, params: dict[str, str]) -> bytes:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "ACM-Phase4-Formal-Evidence/1.0 (research benchmark)"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def search_pmids() -> list[str]:
    payload = request_bytes(NCBI_ESEARCH, {
        "db": "pubmed",
        "term": QUERY,
        "retmode": "json",
        "retmax": "100",
        "sort": "relevance",
    })
    parsed = json.loads(payload)
    ids = [str(value) for value in parsed["esearchresult"]["idlist"]]
    return [pmid for pmid in ids if pmid not in set(DEVELOPMENT_PMIDS)]


def fetch_records(pmids: list[str]) -> bytes:
    return request_bytes(NCBI_EFETCH, {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })


def select_exact_100(documents: list[dict], passages: list[dict]) -> tuple[list[dict], list[dict]]:
    selected_passages = passages[:100]
    if len(selected_passages) != 100:
        raise RuntimeError(f"Expected at least 100 eligible passages, found {len(passages)}.")
    selected_sources = {row["source_id"] for row in selected_passages}
    selected_documents = [row for row in documents if row["source_id"] in selected_sources]
    if {row["pmid"] for row in selected_documents} & set(DEVELOPMENT_PMIDS):
        raise RuntimeError("Formal and development PMID sets overlap.")
    return selected_documents, selected_passages


def mismatched_passage(passages: list[dict], index: int) -> dict:
    current = passages[index]
    for offset in range(1, len(passages)):
        candidate = passages[(index + 37 * offset) % len(passages)]
        if (
            candidate["source_id"] != current["source_id"]
            and candidate["text"] not in current["text"]
        ):
            return candidate
    raise RuntimeError("Unable to construct a cross-source mismatch.")


def build_annotations(passages: list[dict]) -> list[dict]:
    rows = []
    for index, passage in enumerate(passages):
        rows.append({
            "annotation_id": f"FORMAL_SUP_{index + 1:04d}",
            "claim": passage["text"],
            "passage_id": passage["passage_id"],
            "label": "SUPPORTED",
            "rationale": "Exact contiguous sentence identity.",
            "annotator": "DETERMINISTIC_EXTRACTIVE_RULE_V1",
            "annotation_version": "formal-v1",
        })
        mismatch = mismatched_passage(passages, index)
        rows.append({
            "annotation_id": f"FORMAL_NSUP_{index + 1:04d}",
            "claim": mismatch["text"],
            "passage_id": passage["passage_id"],
            "label": "NOT_SUPPORTED",
            "rationale": "Claim originates from a different PMID and is absent from the paired passage.",
            "annotator": "DETERMINISTIC_CROSS_SOURCE_MISMATCH_V1",
            "annotation_version": "formal-v1",
        })
    if len(rows) != 200:
        raise RuntimeError("Formal annotation set must contain exactly 200 pairs.")
    return rows


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_pmids = search_pmids()
    if len(candidate_pmids) < 30:
        raise RuntimeError("PubMed search returned too few sources after development exclusion.")
    raw_xml = fetch_records(candidate_pmids[:60])
    documents, passages = parse_records(raw_xml)
    documents, passages = select_exact_100(documents, passages)
    annotations = build_annotations(passages)

    raw_path = EVIDENCE_DIR / "formal_pubmed_records.xml"
    raw_path.write_bytes(raw_xml)
    corpus_path = EVIDENCE_DIR / "formal_passages.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in passages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    annotation_path = EVIDENCE_DIR / "formal_claim_passage_annotations.json"
    write_json(annotation_path, annotations)

    development_manifest = json.loads(
        (EVIDENCE_DIR / "development_corpus_manifest.json").read_text(encoding="utf-8")
    )
    development_pmids = set(development_manifest["pmids_requested"])
    formal_pmids = {row["pmid"] for row in documents}
    forbidden_tokens = {"survival_months", "event", "vital_status", "risk_score"}
    corpus_text = corpus_path.read_text(encoding="utf-8").lower()
    forbidden_field_keys = sorted(
        token
        for token in forbidden_tokens
        if re.search(rf'"{re.escape(token)}"\s*:', corpus_text)
    )
    gate = {
        "status": "FORMAL_EVIDENCE_READY_NOT_YET_USED",
        "success": bool(
            len(passages) == 100
            and len(annotations) == 200
            and not (development_pmids & formal_pmids)
            and not forbidden_field_keys
        ),
        "formal_result_seen": False,
        "passage_count": len(passages),
        "annotation_count": len(annotations),
        "supported_count": sum(row["label"] == "SUPPORTED" for row in annotations),
        "not_supported_count": sum(row["label"] == "NOT_SUPPORTED" for row in annotations),
        "development_formal_pmid_overlap": len(development_pmids & formal_pmids),
        "outcome_field_tokens_found": forbidden_field_keys,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "raw_xml_sha256": sha256_bytes(raw_xml),
        "protocol_amendment": "PHASE4_PROTOCOL_AMENDMENT_V3_1_EXTRACTIVE_EVIDENCE_20260727.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(READINESS_DIR / "PHASE4_FORMAL_EVIDENCE_GATE.json", gate)
    write_json(EVIDENCE_DIR / "formal_corpus_manifest.json", {
        "status": gate["status"],
        "query": QUERY,
        "search_endpoint": NCBI_ESEARCH,
        "fetch_endpoint": NCBI_EFETCH,
        "retrieval_utc": gate["generated_utc"],
        "document_count": len(documents),
        "passage_count": len(passages),
        "annotation_count": len(annotations),
        "documents": documents,
        "corpus_sha256": gate["corpus_sha256"],
        "annotations_sha256": gate["annotations_sha256"],
        "formal_use_permitted_after_full_readiness_gate": True,
    })
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
