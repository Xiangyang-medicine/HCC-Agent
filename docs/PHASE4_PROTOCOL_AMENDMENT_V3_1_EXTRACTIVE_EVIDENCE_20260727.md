# Phase 4 Protocol Amendment v3.1: Deterministic Extractive Evidence Contract

**Date:** 2026-07-27  
**Timing:** Written after the development benchmark and before any reserved formal-case run  
**Status:** `PRE-FORMAL_AMENDMENT`  
**Changes confirmatory endpoint:** No; it operationalizes claim support without an LLM judge or physician panel.

## Rationale

Protocol v3.0 required at least 200 author-annotated claim-passage pairs. The development run showed that a stricter and more reproducible contract is feasible: every scored biomedical claim must be an exact contiguous sentence from a retrieved frozen passage. This makes claim support and citation linkage deterministic and eliminates the need to use the generating LLM, another LLM, or a physician as the sole correctness judge.

This amendment is frozen before formal evidence acquisition is evaluated with the reserved cases. It does not use or respond to any formal performance result.

## Formal evidence benchmark

1. Development and formal PubMed source identifiers must be disjoint.
2. The formal corpus contains exactly 100 immutable sentence-level passages with source URL, PMID, DOI where available, title, year, and SHA-256 provenance.
3. The support benchmark contains exactly 200 deterministic pairs:
   - 100 `SUPPORTED` pairs: the claim is the exact passage sentence;
   - 100 `NOT_SUPPORTED` pairs: the claim is an exact sentence from a different source and is not contained in the paired passage.
4. `CONFLICTING` is evaluated as a separate, prespecified fault-injection condition rather than being mixed into the primary extractive-support reference set.
5. The formal generator may produce one to three claims. Every claim must be copied exactly from a retrieved passage and cite that passage ID.
6. Parseable JSON, valid-looking citations, or semantic similarity alone cannot establish support.

## Scoring

A claim is supported only if all conditions hold:

- at least one citation ID is present;
- every cited ID exists in the retrieved passage set;
- the complete normalized claim text is an exact contiguous substring of at least one cited passage;
- no prohibited clinical claim is present.

The primary case-level verified task-success endpoint remains unchanged. A case still requires correct planning, correct tool order and arguments, exact model-value fidelity, valid schema, supported claims, and a verified report or prespecified safe-abstention state.

## Interpretation

This benchmark evaluates provenance-constrained extractive evidence use, not open-ended clinical explanation quality. It supports claims about tool orchestration, numerical fidelity, evidence traceability, verification, revision, and safe failure under the frozen task distribution. It does not support claims of physician acceptance, clinical utility, diagnosis, treatment guidance, or general biomedical reasoning ability.

