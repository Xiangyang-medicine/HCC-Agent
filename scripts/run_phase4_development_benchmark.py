"""Run the live 20-case Phase 4 development benchmark.

Outputs are explicitly development-only and cannot be reported as formal test
performance.  API credentials are read from environment variables and are
never serialized.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agent_system.phase4.benchmark import run_comparator, score_clean_run, serializable_state
from src.agent_system.phase4.llm_policy import OpenAICompatibleJSONCallable
from src.agent_system.phase4.orchestrator import SystemVariant
from src.agent_system.phase4.schema import TaskRequest
from src.agent_system.phase4.tools import FrozenEvidenceCorpusTool, FrozenOOFPrognosticTool, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "phase4_benchmark" / "development_cases.json"
CORPUS = ROOT / "data" / "phase4_evidence" / "development_passages.jsonl"
OOF = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "experiments" / "phase4" / "development"
ALL_SYSTEMS = [
    SystemVariant.B0_ENGINE_ONLY,
    SystemVariant.B1_SINGLE_LLM_NO_TOOLS,
    SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
    SystemVariant.B3_MULTI_AGENT_NO_VERIFIER,
    SystemVariant.B4_FULL_CLOSED_LOOP,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--systems",
        default=",".join(system.name for system in ALL_SYSTEMS),
        help="Comma-separated SystemVariant enum names.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.limit > 20:
        raise ValueError("Development limit must be within 1..20.")
    if args.runs < 1:
        raise ValueError("runs must be positive.")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be within 1..8.")
    selected = [SystemVariant[name.strip()] for name in args.systems.split(",")]
    cases = json.loads(CASES.read_text(encoding="utf-8"))[: args.limit]
    config_completion = OpenAICompatibleJSONCallable.from_environment()
    prognostic_tool = FrozenOOFPrognosticTool(OOF)
    evidence_tool = FrozenEvidenceCorpusTool(CORPUS, top_k=3)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (run_index, case, variant)
        for run_index in range(1, args.runs + 1)
        for case in cases
        for variant in selected
    ]

    def execute_job(job):
        run_index, case, variant = job
        request = TaskRequest(
            task_id=str(case["task_id"]),
            case_id=str(case["case_id"]),
            repeat=int(case["oof_repeat"]),
        )
        completion = OpenAICompatibleJSONCallable.from_environment()
        started = time.perf_counter()
        error_type = None
        try:
            state = run_comparator(
                request=request,
                variant=variant,
                prognostic_tool=prognostic_tool,
                evidence_tool=evidence_tool,
                completion=completion,
            )
            scores = score_clean_run(
                state=state,
                variant=variant,
                request=request,
                prognostic_tool=prognostic_tool,
                evidence_tool=evidence_tool,
            )
            state_payload = serializable_state(state)
        except Exception as exc:
            error_type = type(exc).__name__
            scores = {
                "verified_task_success": False,
                "plan_valid": False,
                "tool_order_exact": False,
                "schema_valid": False,
                "numeric_fidelity": False,
                "forbidden_claim_case": False,
                "safe_abstain": False,
                "revision_count": 0,
            }
            state_payload = {"status": "FAILED", "error_type": error_type}
        call_rows = completion.call_records
        return {
            "task_id": request.task_id,
            "case_id": request.case_id,
            "benchmark_run": run_index,
            "system": variant.name,
            "system_value": variant.value,
            "evaluation_split": "DEVELOPMENT_ONLY",
            "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "llm_call_count": len(call_rows),
            "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in call_rows),
            "completion_tokens": sum(row.get("completion_tokens") or 0 for row in call_rows),
            "total_tokens": sum(row.get("total_tokens") or 0 for row in call_rows),
            "api_error_type": error_type,
            **scores,
            "state": state_payload,
            "llm_calls": call_rows,
        }

    records: list[dict] = []
    raw_path = args.output_dir / "run_records.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        if args.workers == 1:
            iterator = map(execute_job, jobs)
            for record in iterator:
                records.append(record)
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_handle.flush()
                print(
                    f"{len(records):03d} {record['task_id']} {record['system']} "
                    f"success={record['verified_task_success']} calls={record['llm_call_count']}"
                )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(execute_job, job) for job in jobs]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw_handle.flush()
                    print(
                        f"{len(records):03d} {record['task_id']} {record['system']} "
                        f"success={record['verified_task_success']} calls={record['llm_call_count']}"
                    )

    flat_rows = [
        {key: value for key, value in record.items() if key not in {"state", "llm_calls"}}
        for record in records
    ]
    frame = pd.DataFrame(flat_rows)
    frame = frame.sort_values(["benchmark_run", "task_id", "system"]).reset_index(drop=True)
    frame.to_csv(args.output_dir / "case_level_metrics.csv", index=False)
    metric_columns = [
        "verified_task_success", "plan_valid", "tool_order_exact", "schema_valid",
        "numeric_fidelity", "supported_claim_precision", "citation_completeness",
        "citation_correctness", "unsupported_claim_rate", "forbidden_claim_case",
        "safe_abstain", "wall_latency_ms", "llm_call_count", "total_tokens",
    ]
    for column in metric_columns:
        if column not in frame.columns:
            frame[column] = float("nan")
    summary = (
        frame.groupby("system", sort=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = frame.groupby("system", sort=False).size().rename("n_runs").reset_index()
    summary = counts.merge(summary, on="system", how="left")
    summary.to_csv(args.output_dir / "metrics_by_system.csv", index=False)
    (args.output_dir / "metrics_by_system.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )

    expected = len(cases) * args.runs * len(selected)
    gate = {
        "status": "DEVELOPMENT_COMPLETED_NOT_FORMAL",
        "success": bool(len(records) == expected and frame["api_error_type"].isna().all()),
        "formal_benchmark_ready": False,
        "case_count": len(cases),
        "run_repeats": args.runs,
        "systems": [system.name for system in selected],
        "expected_records": expected,
        "observed_records": len(records),
        "api_error_count": int(frame["api_error_type"].notna().sum()),
        "credential_serialized": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "DEVELOPMENT_GATE.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": gate["status"],
        "model": config_completion.model,
        "temperature": config_completion.temperature,
        "max_tokens": config_completion.max_tokens,
        "base_url_sha256": sha256_text(config_completion.base_url or "default"),
        "workers": args.workers,
        "oof_sha256": sha256_file(OOF),
        "corpus_sha256": sha256_file(CORPUS),
        "cases_sha256": sha256_file(CASES),
        "credential_fields_recorded": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
