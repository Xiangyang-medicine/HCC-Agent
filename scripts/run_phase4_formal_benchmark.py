"""Fail-closed formal Phase 4 runner for clean, ablation, and fault runs.

This script cannot start without the explicit cost-authorization flag and a
passing machine-readable readiness gate. Credentials remain environment-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_system.phase4.ablations import AblationVariant, build_ablation_agent
from src.agent_system.phase4.benchmark import run_comparator, score_clean_run, serializable_state
from src.agent_system.phase4.faults import (
    ConflictingEvidenceTool,
    FaultType,
    MetadataMismatchEvidenceTool,
    MissingFeaturePrognosticTool,
    NonfiniteModelOutputTool,
    PermanentFailingPrognosticTool,
    TransientEvidenceTool,
    invalid_request_fields,
    score_fault_run,
    unsupported_claim_request,
)
from src.agent_system.phase4.llm_policy import OpenAICompatibleJSONCallable
from src.agent_system.phase4.orchestrator import SystemVariant
from src.agent_system.phase4.schema import TaskRequest
from src.agent_system.phase4.tools import FrozenEvidenceCorpusTool, FrozenOOFPrognosticTool


CASES = ROOT / "data" / "phase4_benchmark" / "formal_cases_reserved_blinded.json"
CORPUS = ROOT / "data" / "phase4_evidence" / "formal_passages.jsonl"
OOF = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
OUTPUT = ROOT / "experiments" / "phase4" / "formal"
GATE = ROOT / "experiments" / "phase4" / "readiness" / "PHASE4_FORMAL_READINESS_GATE.json"
AUTHORIZATION_PHRASE = "I_AUTHORIZE_FORMAL_LLM_COST"
CLEAN_PROGNOSTIC = None
CLEAN_EVIDENCE = None

CLEAN_SYSTEMS = [
    SystemVariant.B0_ENGINE_ONLY,
    SystemVariant.B1_SINGLE_LLM_NO_TOOLS,
    SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
    SystemVariant.B3_MULTI_AGENT_NO_VERIFIER,
    SystemVariant.B4_FULL_CLOSED_LOOP,
]
FAULT_SYSTEMS = [
    SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
    SystemVariant.B3_MULTI_AGENT_NO_VERIFIER,
    SystemVariant.B4_FULL_CLOSED_LOOP,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--mode", choices=["clean", "ablations", "faults", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run after strict validation of its JSONL checkpoint.",
    )
    return parser.parse_args()


def authorize_and_verify(resume: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "verify_phase4_formal_readiness.py"),
        "--external-cost-authorized",
    ]
    if resume:
        command.append("--resume")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("Readiness verification command failed.")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("formal_run_permitted"):
        failed = [key for key, value in gate.get("checks", {}).items() if not value]
        raise RuntimeError(f"Formal run is blocked by readiness checks: {failed}")


def base_request(case: dict) -> TaskRequest:
    return TaskRequest(
        task_id=str(case["task_id"]),
        case_id=str(case["case_id"]),
        repeat=int(case["oof_repeat"]),
    )


def apply_fault(fault: FaultType, request, prognostic, evidence):
    if fault == FaultType.INVALID_REQUEST_FIELDS:
        request = invalid_request_fields(request)
    elif fault == FaultType.TRANSIENT_RETRIEVAL_TIMEOUT:
        evidence = TransientEvidenceTool(evidence)
    elif fault == FaultType.PERMANENT_MODEL_FAILURE:
        prognostic = PermanentFailingPrognosticTool()
    elif fault == FaultType.MISSING_GENE_FEATURES:
        prognostic = MissingFeaturePrognosticTool()
    elif fault == FaultType.MALFORMED_MODEL_OUTPUT:
        prognostic = NonfiniteModelOutputTool(prognostic)
    elif fault == FaultType.CITATION_METADATA_MISMATCH:
        evidence = MetadataMismatchEvidenceTool(evidence)
    elif fault == FaultType.CONFLICTING_EVIDENCE:
        evidence = ConflictingEvidenceTool(evidence)
    elif fault == FaultType.UNSUPPORTED_REQUESTED_CLAIM:
        request = unsupported_claim_request(request)
    return request, prognostic, evidence


def execute_job(job: dict) -> dict:
    completion = OpenAICompatibleJSONCallable.from_environment()
    clean_prognostic = CLEAN_PROGNOSTIC
    clean_evidence = CLEAN_EVIDENCE
    if clean_prognostic is None or clean_evidence is None:
        raise RuntimeError("Formal deterministic tools were not initialized.")
    request = base_request(job["case"])
    started = time.perf_counter()
    error_type = None
    try:
        if job["kind"] == "clean":
            variant = job["variant"]
            state = run_comparator(request, variant, clean_prognostic, clean_evidence, completion)
            scores = score_clean_run(
                state, variant, request, clean_prognostic, clean_evidence
            )
            label = variant.name
        elif job["kind"] == "ablation":
            ablation = job["ablation"]
            state = build_ablation_agent(
                ablation, clean_prognostic, clean_evidence, completion
            ).run(request)
            scores = score_clean_run(
                state, SystemVariant.B4_FULL_CLOSED_LOOP,
                request, clean_prognostic, clean_evidence,
            )
            label = ablation.value
        else:
            variant = job["variant"]
            fault = job["fault"]
            fault_request, fault_prognostic, fault_evidence = apply_fault(
                fault, request, clean_prognostic, clean_evidence
            )
            state = run_comparator(
                fault_request, variant, fault_prognostic, fault_evidence, completion
            )
            scores = score_fault_run(
                state, fault, variant, fault_request, clean_prognostic, clean_evidence
            )
            label = variant.name
        state_payload = serializable_state(state)
    except Exception as exc:
        error_type = type(exc).__name__
        scores = {"verified_task_success": False, "recovery_or_safe_abstention": False}
        state_payload = {"status": "FAILED", "error_type": error_type}
        label = job.get("variant", job.get("ablation", "UNKNOWN"))
        label = label.name if isinstance(label, SystemVariant) else str(label)
    finally:
        completion.close()
    calls = completion.call_records
    record = {
        "task_id": job["case"]["task_id"],
        "case_id": job["case"]["case_id"],
        "formal_repeat": job["repeat"],
        "run_kind": job["kind"],
        "system": label,
        "fault_type": None if job.get("fault") is None else job["fault"].value,
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "llm_call_count": len(calls),
        "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in calls),
        "completion_tokens": sum(row.get("completion_tokens") or 0 for row in calls),
        "total_tokens": sum(row.get("total_tokens") or 0 for row in calls),
        "api_error_type": error_type,
        **scores,
        "state": state_payload,
        "llm_calls": calls,
    }
    record["record_id"] = record_id_from_record(record)
    return record


def build_jobs(cases: list[dict], mode: str) -> list[dict]:
    jobs = []
    if mode in {"clean", "all"}:
        jobs.extend(
            {"kind": "clean", "case": case, "repeat": repeat, "variant": variant}
            for repeat in range(1, 4)
            for case in cases
            for variant in CLEAN_SYSTEMS
        )
    if mode in {"ablations", "all"}:
        jobs.extend(
            {"kind": "ablation", "case": case, "repeat": repeat, "ablation": ablation}
            for repeat in range(1, 4)
            for case in cases
            for ablation in AblationVariant
        )
    if mode in {"faults", "all"}:
        for fault_index, fault in enumerate(FaultType):
            # Deterministic, shifted 30-case assignment per fault.
            selected = [cases[(fault_index * 11 + offset) % len(cases)] for offset in range(30)]
            jobs.extend(
                {"kind": "fault", "case": case, "repeat": repeat, "fault": fault, "variant": variant}
                for repeat in range(1, 4)
                for case in selected
                for variant in FAULT_SYSTEMS
            )
    return jobs


def job_system_label(job: dict) -> str:
    if job["kind"] == "ablation":
        return job["ablation"].value
    return job["variant"].name


def job_identity(job: dict) -> tuple[str, int, str, str, str | None]:
    return (
        str(job["case"]["task_id"]),
        int(job["repeat"]),
        str(job["kind"]),
        job_system_label(job),
        None if job.get("fault") is None else job["fault"].value,
    )


def record_identity(record: dict) -> tuple[str, int, str, str, str | None]:
    return (
        str(record["task_id"]),
        int(record["formal_repeat"]),
        str(record["run_kind"]),
        str(record["system"]),
        record.get("fault_type"),
    )


def identity_record_id(identity: tuple[str, int, str, str, str | None]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_id_from_record(record: dict) -> str:
    return identity_record_id(record_identity(record))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_resume_records(raw_path: Path, jobs: list[dict]) -> tuple[list[dict], dict]:
    if not raw_path.exists():
        raise RuntimeError(f"Resume checkpoint is missing: {raw_path}")
    checkpoint_sha256 = sha256_file(raw_path)
    expected = {job_identity(job) for job in jobs}
    if len(expected) != len(jobs):
        raise RuntimeError("Prespecified formal job manifest contains duplicate identities.")

    parsed = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed checkpoint JSON at line {line_number}.") from exc
        identity = record_identity(record)
        if identity not in expected:
            raise RuntimeError(f"Unexpected checkpoint identity at line {line_number}: {identity}")
        record["record_id"] = identity_record_id(identity)
        parsed.append(record)

    successful = [
        record for record in parsed
        if not str(record.get("api_error_type") or "").strip()
    ]
    successful_identities = [record_identity(record) for record in successful]
    if len(successful_identities) != len(set(successful_identities)):
        raise RuntimeError("Checkpoint contains duplicate successful job identities.")
    dropped_errors = len(parsed) - len(successful)
    audit = {
        "status": "RESUME_CHECKPOINT_VALIDATED",
        "checkpoint_sha256_before_sanitization": checkpoint_sha256,
        "checkpoint_records_total": len(parsed),
        "checkpoint_successful_records_retained": len(successful),
        "checkpoint_api_error_records_scheduled_for_rerun": dropped_errors,
        "expected_job_count": len(jobs),
        "remaining_job_count": len(jobs) - len(successful),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return successful, audit


def rewrite_checkpoint(raw_path: Path, records: list[dict]) -> None:
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(raw_path)


def main() -> None:
    global CLEAN_PROGNOSTIC, CLEAN_EVIDENCE
    args = parse_args()
    if args.authorization != AUTHORIZATION_PHRASE:
        raise RuntimeError("Exact external-cost authorization phrase is required.")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be within 1..8")
    authorize_and_verify(args.resume)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 100:
        raise RuntimeError("Formal case manifest must contain exactly 100 cases.")
    jobs = build_jobs(cases, args.mode)
    CLEAN_PROGNOSTIC = FrozenOOFPrognosticTool(OOF)
    CLEAN_EVIDENCE = FrozenEvidenceCorpusTool(CORPUS, top_k=3)
    raw_path = OUTPUT / f"{args.mode}_run_records.jsonl"
    if args.resume:
        if not OUTPUT.is_dir():
            raise RuntimeError("Resume requested but formal output directory does not exist.")
        records, resume_audit = read_resume_records(raw_path, jobs)
        rewrite_checkpoint(raw_path, records)
        (OUTPUT / f"{args.mode}_RESUME_AUDIT.json").write_text(
            json.dumps(resume_audit, indent=2) + "\n", encoding="utf-8"
        )
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        records = []

    completed = {record_identity(record) for record in records}
    pending_jobs = [job for job in jobs if job_identity(job) not in completed]
    file_mode = "a" if args.resume else "w"
    with raw_path.open(file_mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute_job, job) for job in pending_jobs]
            for index, future in enumerate(as_completed(futures), start=len(records) + 1):
                record = future.result()
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"{index}/{len(jobs)} {record['run_kind']} {record['task_id']} "
                    f"{record['system']}"
                )
    identities = [record_identity(record) for record in records]
    unique_count = len(set(identities))
    flat = pd.DataFrame([
        {key: value for key, value in record.items() if key not in {"state", "llm_calls"}}
        for record in records
    ]).sort_values(["run_kind", "formal_repeat", "task_id", "system"])
    flat.to_csv(OUTPUT / f"{args.mode}_case_level_metrics.csv", index=False)
    manifest = {
        "status": (
            f"FORMAL_{args.mode.upper()}_COMPLETED_UNANALYZED"
            if len(records) == len(jobs)
            and unique_count == len(jobs)
            and int(flat["api_error_type"].notna().sum()) == 0
            else f"FORMAL_{args.mode.upper()}_FAILED_COMPLETENESS"
        ),
        "mode": args.mode,
        "job_count": len(jobs),
        "record_count": len(records),
        "unique_record_count": unique_count,
        "api_error_count": int(flat["api_error_type"].notna().sum()),
        "resume_used": bool(args.resume),
        "credential_serialized": False,
        "source_hashes_sha256": {
            "formal_runner": sha256_file(Path(__file__)),
            "llm_policy": sha256_file(
                ROOT / "src" / "agent_system" / "phase4" / "llm_policy.py"
            ),
            "resume_protocol": sha256_file(
                ROOT / "docs" / "PHASE4_FORMAL_INTERRUPTION_AND_RESUME_PROTOCOL_20260727.md"
            ),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / f"{args.mode}_RUN_GATE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
