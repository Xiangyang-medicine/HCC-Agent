# Phase 4 Formal Interruption and Resume Protocol

Status: infrastructure-only amendment approved by the user on 2026-07-27.

## Trigger

The first formal execution stopped making progress after 170 of 4,860 job
records. The checkpoint had not changed for more than 30 minutes. The process
remained alive, seven records contained `APITimeoutError`, and the Python
process accumulated many established and `CLOSE_WAIT` connections.

## Controlled interruption

- Both formal-run Python processes were terminated at
  2026-07-27T14:16:51+08:00.
- The incomplete JSONL contained 170 records.
- Its SHA-256 was
  `E04CBFC7B0A7802738AA3DC0C03979B0F8E3A7A7938B163B582955BB5EB3D953`.
- An immutable working copy was placed under
  `experiments/phase4/archive/formal_interrupted_20260727_141651/`.
- The interrupted checkpoint is not a formal result and must never be used for
  comparative inference.

## Permitted infrastructure changes

Only the following operational changes are allowed:

1. Explicitly close each job's HTTP client and connection pool.
2. Set the request timeout to 120 seconds and transport retries to one.
3. Assign deterministic identities to all 4,860 prespecified jobs.
4. Resume only after validating every checkpoint identity against the frozen
   job manifest.
5. Retain successful records and rerun only records carrying a transport/API
   error.
6. Reject duplicate or unexpected successful records.
7. Flush and synchronize every completed record before considering it
   checkpointed.
8. Require exactly 4,860 unique records and zero API errors for the completion
   gate to pass.

No formal case, model, prompt, system variant, fault assignment, ablation,
scoring rule, endpoint model setting, or statistical comparison may be changed.
No comparative performance result from the interrupted checkpoint may be
inspected or used to tune the system.

## Second operational interruption

At 2026-07-27T15:50:09+08:00, the resumed four-worker execution was stopped
after the transport-error count increased from 53 to 180; 57 of the most
recent 100 completed jobs carried an API error. At interruption, the
checkpoint contained 1,308 records: 1,128 transport-successful records and 180
API-error records. Its SHA-256 was
`7A220A1A644A9E247DAE937677029DE4DE75CAB00670F67615D162188DAD0908`.
The snapshot was archived under
`experiments/phase4/archive/formal_concurrency_reduction_20260727_155009/`.

The worker count was reduced from four to two as an infrastructure-only load
control. Successful job identities remain checkpointed; API-error identities
are rerun. The formal cases, systems, prompts, tools, model, decoding
parameters, metrics, and statistical analysis remain unchanged.

## Third operational interruption and independent background resume

The two-worker process later disappeared without producing a completion gate.
At 2026-07-27T16:48:16+08:00, the checkpoint contained 1,386 records:
1,236 transport-successful records and 150 API-error records. Its SHA-256 was
`23A727013280E653A389F69F8ADD745EBF04ABDC5F1621474B17B779730A5B55`.
The snapshot was archived under
`experiments/phase4/archive/formal_process_lost_20260727_164816/`.

The next resume uses one worker and an on-demand Windows Scheduled Task so the
process is independent of the Codex execution host. The API credential is
stored only as Windows DPAPI ciphertext decryptable by the current Windows
user. The task launcher does not print or serialize the plaintext credential.
The reduction to one worker is an infrastructure-only response to repeated
connection errors and does not alter any scientific input, prompt, output
scoring rule, comparator, or analysis.

## Migration to teamr1 remote server

The remote host `teamr1.iwangshu.com` was reachable only through the user's
direct network after disabling the local FlyingBird tunnel. The minimal formal
bundle and checkpoint were transferred to the remote host and verified by
SHA-256 before extraction. No TCGA raw files or local credential store were
included.

The remote host uses Linux and Python 3.10.12. Two Windows-specific interpreter
paths in the formal runner and readiness verifier were replaced with
`sys.executable`. This is a platform-compatibility change only. The remote
isolated environment freezes numpy 2.2.6, pandas 2.2.3, openai 2.48.0, and
pytest 8.3.5. Scientific assets, cases, prompts, decoding parameters, systems,
fault assignments, scoring rules, and endpoints remain unchanged.

## Required audit outputs

The resumed run must generate:

- `all_RESUME_AUDIT.json`
- `all_run_records.jsonl`
- `all_case_level_metrics.csv`
- `all_RUN_GATE.json`

The final gate must record the resume flag, exact record counts, API error
count, and SHA-256 hashes of the runner and LLM transport source.
