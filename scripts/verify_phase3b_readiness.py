#!/usr/bin/env python3
"""Report Phase 3B gates without fabricating readiness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "phase3b" / "readiness" / "PHASE3B_READINESS_GATE.json"


def main() -> int:
    amendment = ROOT / "docs" / "PHASE_3B_PROTOCOL_AMENDMENT_V2.md"
    gse_audit = ROOT / "experiments" / "phase3b" / "gse14520" / "SOURCE_AUDIT.json"
    artifact_verification = ROOT / "experiments" / "phase3b" / "derivation" / "ARTIFACT_VERIFICATION.json"
    primary_rnaseq_manifest = ROOT / "data" / "external" / "primary_rnaseq" / "SOURCE_MANIFEST.json"
    gates = {
        "protocol_amendment_frozen": amendment.is_file(),
        "gse14520_exploratory_source_audited": gse_audit.is_file(),
        "m4_external_validation_artifact_verified": artifact_verification.is_file()
        and json.loads(artifact_verification.read_text(encoding="utf-8")).get("success") is True,
        "primary_independent_rnaseq_manifest_present": primary_rnaseq_manifest.is_file(),
    }
    ready = all(gates.values())
    result = {
        "status": "PHASE3B_READY_FOR_SCORING" if ready else "PHASE3B_DATA_AND_ARTIFACT_PENDING",
        "success": bool(ready),
        "gates": {key: bool(value) for key, value in gates.items()},
        "blocked_by": [key for key, value in gates.items() if not value],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
