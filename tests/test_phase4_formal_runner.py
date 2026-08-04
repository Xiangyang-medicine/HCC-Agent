import json
from types import SimpleNamespace

import pytest

from src.agent_system.phase4.llm_policy import OpenAICompatibleJSONCallable
from scripts.run_phase4_formal_benchmark import (
    build_jobs,
    job_identity,
    read_resume_records,
    record_id_from_record,
)


def test_formal_all_mode_job_counts_are_prespecified():
    cases = [
        {"task_id": f"FORMAL_{index:03d}", "case_id": f"CASE_{index:03d}", "oof_repeat": 1}
        for index in range(1, 101)
    ]
    jobs = build_jobs(cases, "all")
    clean = [job for job in jobs if job["kind"] == "clean"]
    ablations = [job for job in jobs if job["kind"] == "ablation"]
    faults = [job for job in jobs if job["kind"] == "fault"]
    assert len(clean) == 100 * 3 * 5
    assert len(ablations) == 100 * 3 * 4
    assert len(faults) == 8 * 30 * 3 * 3
    assert len(jobs) == 4860


def test_each_fault_uses_30_distinct_cases_per_repeat_and_system():
    cases = [
        {"task_id": f"FORMAL_{index:03d}", "case_id": f"CASE_{index:03d}", "oof_repeat": 1}
        for index in range(1, 101)
    ]
    jobs = [job for job in build_jobs(cases, "faults") if job["repeat"] == 1]
    keys = {(job["fault"].value, job["variant"].name) for job in jobs}
    for fault, system in keys:
        subset = [
            job for job in jobs
            if job["fault"].value == fault and job["variant"].name == system
        ]
        assert len(subset) == 30
        assert len({job["case"]["case_id"] for job in subset}) == 30


def _record_for_job(job, api_error_type=None):
    if job["kind"] == "ablation":
        system = job["ablation"].value
    else:
        system = job["variant"].name
    record = {
        "task_id": job["case"]["task_id"],
        "case_id": job["case"]["case_id"],
        "formal_repeat": job["repeat"],
        "run_kind": job["kind"],
        "system": system,
        "fault_type": None if job.get("fault") is None else job["fault"].value,
        "api_error_type": api_error_type,
    }
    record["record_id"] = record_id_from_record(record)
    return record


def test_resume_retains_success_and_reruns_api_errors(tmp_path):
    cases = [
        {"task_id": "FORMAL_001", "case_id": "CASE_001", "oof_repeat": 1},
        {"task_id": "FORMAL_002", "case_id": "CASE_002", "oof_repeat": 1},
    ]
    jobs = build_jobs(cases, "clean")
    checkpoint = tmp_path / "all_run_records.jsonl"
    records = [
        _record_for_job(jobs[0]),
        _record_for_job(jobs[1], api_error_type="APITimeoutError"),
    ]
    checkpoint.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    retained, audit = read_resume_records(checkpoint, jobs)

    assert len(retained) == 1
    assert job_identity(jobs[0]) == (
        retained[0]["task_id"],
        retained[0]["formal_repeat"],
        retained[0]["run_kind"],
        retained[0]["system"],
        retained[0]["fault_type"],
    )
    assert audit["checkpoint_api_error_records_scheduled_for_rerun"] == 1
    assert audit["remaining_job_count"] == len(jobs) - 1


def test_resume_rejects_duplicate_success_records(tmp_path):
    cases = [{"task_id": "FORMAL_001", "case_id": "CASE_001", "oof_repeat": 1}]
    jobs = build_jobs(cases, "clean")
    record = _record_for_job(jobs[0])
    checkpoint = tmp_path / "all_run_records.jsonl"
    checkpoint.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate successful"):
        read_resume_records(checkpoint, jobs)


def test_resume_rejects_unexpected_identity(tmp_path):
    cases = [{"task_id": "FORMAL_001", "case_id": "CASE_001", "oof_repeat": 1}]
    jobs = build_jobs(cases, "clean")
    record = _record_for_job(jobs[0])
    record["task_id"] = "NOT_IN_MANIFEST"
    checkpoint = tmp_path / "all_run_records.jsonl"
    checkpoint.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unexpected checkpoint identity"):
        read_resume_records(checkpoint, jobs)


def test_live_callable_reuses_one_client_and_closes_pool(monkeypatch):
    created = []

    class FakeCompletions:
        def __init__(self, owner):
            self.owner = owner

        def create(self, **kwargs):
            self.owner.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                ),
                model="frozen-test-model",
                _request_id="request-test",
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0
            self.closed = False
            self.chat = SimpleNamespace(completions=FakeCompletions(self))
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    completion = OpenAICompatibleJSONCallable(
        api_key="test-only",
        base_url="https://example.invalid/v1",
        model="frozen-test-model",
    )

    assert completion("system", "user") == '{"ok": true}'
    assert completion("system", "user") == '{"ok": true}'
    completion.close()

    assert len(created) == 1
    assert created[0].calls == 2
    assert created[0].closed is True
    assert created[0].kwargs["timeout"] == 120.0
    assert created[0].kwargs["max_retries"] == 1
