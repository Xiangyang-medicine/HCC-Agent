"""Prepare traceable Phase 4 development evidence and benchmark cases.

The development corpus uses bounded PubMed abstract sentences retrieved from
NCBI.  Exact source sentences are the only supported claim form, making the
development scorer deterministic and avoiding an LLM-as-judge design.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "phase4_evidence"
TASK_DIR = ROOT / "data" / "phase4_benchmark"
OOF_PATH = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"

PMIDS = [
    "33869278", "35677150", "35406630", "36919714", "33188160",
    "30463528", "37006288", "35685867", "33013158", "31847435",
    "23824744", "35651292",
]
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEED = 20260727


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def text_of(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def fetch_pubmed_xml() -> bytes:
    query = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(PMIDS),
        "retmode": "xml",
    })
    request = urllib.request.Request(
        f"{NCBI_EFETCH}?{query}",
        headers={"User-Agent": "ACM-Phase4-Benchmark/1.0 (research corpus acquisition)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [
        sentence.strip()
        for sentence in candidates
        if 70 <= len(sentence.strip()) <= 480
    ]


def parse_records(xml_payload: bytes) -> tuple[list[dict], list[dict]]:
    root = ET.fromstring(xml_payload)
    documents: list[dict] = []
    passages: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = text_of(article.find(".//PMID"))
        title = text_of(article.find(".//ArticleTitle"))
        abstract_parts = [text_of(node) for node in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part)
        if not pmid or not abstract:
            continue
        year = text_of(article.find(".//PubDate/Year"))
        if not year:
            medline_date = text_of(article.find(".//PubDate/MedlineDate"))
            match = re.search(r"\b(19|20)\d{2}\b", medline_date)
            year = match.group(0) if match else "unknown"
        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = text_of(article_id)
                break
        source_id = f"PMID_{pmid}"
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        documents.append({
            "source_id": source_id,
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "publication_year": year,
            "source_url": source_url,
            "abstract_sha256": sha256_bytes(abstract.encode("utf-8")),
        })
        for index, sentence in enumerate(split_sentences(abstract)[:4], start=1):
            passages.append({
                "source_id": source_id,
                "passage_id": f"{source_id}_P{index:02d}",
                "text": sentence,
                "metadata": {
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "publication_year": year,
                    "source_url": source_url,
                },
            })
    if len(documents) < 8 or len(passages) < 24:
        raise RuntimeError(
            f"Insufficient PubMed development corpus: {len(documents)} documents, "
            f"{len(passages)} passages."
        )
    return documents, passages


def build_development_annotations(passages: list[dict]) -> list[dict]:
    annotations: list[dict] = []
    for index, passage in enumerate(passages):
        annotations.append({
            "annotation_id": f"DEV_SUP_{index + 1:04d}",
            "claim": passage["text"],
            "passage_id": passage["passage_id"],
            "label": "SUPPORTED",
            "rationale": "Exact contiguous sentence from the cited frozen passage.",
            "annotator": "DETERMINISTIC_EXTRACTIVE_RULE_V1",
            "annotation_version": "development-v1",
        })
        other = passages[(index + 7) % len(passages)]
        annotations.append({
            "annotation_id": f"DEV_NSUP_{index + 1:04d}",
            "claim": other["text"],
            "passage_id": passage["passage_id"],
            "label": "NOT_SUPPORTED",
            "rationale": "Exact sentence originates from a different passage and is not present in the cited passage.",
            "annotator": "DETERMINISTIC_MISMATCH_RULE_V1",
            "annotation_version": "development-v1",
        })
    return annotations


def stratified_case_manifests() -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(OOF_PATH)
    m4 = frame.loc[frame["model"] == "M4_combined_rsf"].copy()
    summary = (
        m4.groupby("case_id", as_index=False)
        .agg(risk_score=("risk_score", "mean"), event=("event", "first"))
    )
    summary["risk_quintile"] = pd.qcut(
        summary["risk_score"], 5, labels=False, duplicates="drop"
    ).astype(int) + 1
    rng = np.random.default_rng(SEED)
    development_ids: list[str] = []
    formal_ids: list[str] = []
    for (_, _), group in summary.groupby(["risk_quintile", "event"], sort=True):
        ids = group["case_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        development_ids.extend(ids[:2].tolist())
        formal_ids.extend(ids[2:12].tolist())
    # Sparse strata can make the exact totals drift; top up deterministically.
    all_ids = summary["case_id"].tolist()
    remaining = [case_id for case_id in all_ids if case_id not in development_ids + formal_ids]
    rng.shuffle(remaining)
    development_ids = development_ids[:20]
    while len(development_ids) < 20:
        development_ids.append(remaining.pop())
    formal_ids = formal_ids[:100]
    while len(formal_ids) < 100:
        formal_ids.append(remaining.pop())
    if set(development_ids) & set(formal_ids):
        raise RuntimeError("Development and formal case IDs overlap.")

    lookup = summary.set_index("case_id")
    def render(ids: list[str], split: str) -> list[dict]:
        rows = []
        for index, case_id in enumerate(ids, start=1):
            row = lookup.loc[case_id]
            rows.append({
                "task_id": f"{split.upper()}_{index:03d}",
                "case_id": case_id,
                "oof_repeat": 1,
                "risk_quintile_sampling_stratum": int(row["risk_quintile"]),
                "event_sampling_stratum": int(row["event"]),
                "agent_outcome_access": False,
            })
        return rows
    return render(development_ids, "development"), render(formal_ids, "formal")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    raw_xml = fetch_pubmed_xml()
    documents, passages = parse_records(raw_xml)
    annotations = build_development_annotations(passages)
    development_cases, formal_cases = stratified_case_manifests()

    raw_path = EVIDENCE_DIR / "development_pubmed_records.xml"
    raw_path.write_bytes(raw_xml)
    corpus_path = EVIDENCE_DIR / "development_passages.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
    write_json(EVIDENCE_DIR / "development_claim_passage_annotations.json", annotations)
    write_json(EVIDENCE_DIR / "development_corpus_manifest.json", {
        "status": "DEVELOPMENT_CORPUS_READY_NOT_FORMAL",
        "purpose": "Phase 4 development only; exact-sentence support contract.",
        "retrieval_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_endpoint": NCBI_EFETCH,
        "pmids_requested": PMIDS,
        "raw_xml_sha256": sha256_bytes(raw_xml),
        "corpus_sha256": sha256_bytes(corpus_path.read_bytes()),
        "document_count": len(documents),
        "passage_count": len(passages),
        "annotation_count": len(annotations),
        "documents": documents,
        "formal_use_permitted": False,
    })
    write_json(TASK_DIR / "development_cases.json", development_cases)
    write_json(TASK_DIR / "formal_cases_reserved_blinded.json", formal_cases)
    write_json(TASK_DIR / "case_split_manifest.json", {
        "seed": SEED,
        "development_n": len(development_cases),
        "formal_reserved_n": len(formal_cases),
        "overlap_n": len(
            {row["case_id"] for row in development_cases}
            & {row["case_id"] for row in formal_cases}
        ),
        "outcomes_available_to_agent": False,
        "oof_source_sha256": sha256_bytes(OOF_PATH.read_bytes()),
    })
    print(json.dumps({
        "documents": len(documents),
        "passages": len(passages),
        "annotations": len(annotations),
        "development_cases": len(development_cases),
        "formal_reserved_cases": len(formal_cases),
    }, indent=2))


if __name__ == "__main__":
    main()
