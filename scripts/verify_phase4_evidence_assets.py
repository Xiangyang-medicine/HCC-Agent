#!/usr/bin/env python3
"""Validate Phase 4 evidence-asset structure without claiming corpus readiness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "phase4_evidence"
OUTPUT = ROOT / "experiments" / "phase4" / "readiness" / "PHASE4_EVIDENCE_ASSETS_GATE.json"


def _loads(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    required = [
        ASSET_DIR / "evidence_schema.json",
        ASSET_DIR / "claim_passage_annotation_schema.json",
        ASSET_DIR / "development_corpus_manifest.json",
        ASSET_DIR / "development_claim_passage_annotations.json",
    ]
    valid_json = True
    for path in required:
        try:
            _loads(path)
        except (OSError, json.JSONDecodeError):
            valid_json = False
    manifest = _loads(ASSET_DIR / "development_corpus_manifest.json")
    annotations = _loads(ASSET_DIR / "development_claim_passage_annotations.json")
    structural_gates = {
        "required_assets_exist": all(path.is_file() for path in required),
        "required_assets_valid_json": valid_json,
        "development_manifest_explicitly_nonformal": manifest.get("status") == "PENDING_SOURCE_ACQUISITION",
        "annotations_are_list": isinstance(annotations, list),
    }
    result = {
        "status": "PHASE4_EVIDENCE_STRUCTURE_READY_CORPUS_PENDING" if all(structural_gates.values()) else "PHASE4_EVIDENCE_STRUCTURE_FAILED",
        "success": bool(all(structural_gates.values())),
        "formal_corpus_ready": False,
        "gates": {key: bool(value) for key, value in structural_gates.items()},
        "document_count": len(manifest.get("documents", [])),
        "annotation_count": len(annotations),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
