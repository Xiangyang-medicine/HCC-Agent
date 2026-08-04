# Phase 3B microarray reporting decision — GPL571 small stratum

**Date:** 2026-07-27  
**Status:** reporting deviation recorded after canonical analysis  
**Applies to:** GSE14520 GPL571, 21 tumour cases and 11 OS events

## Decision

GSE14520 GPL571 is excluded from the **main-text microarray results table and headline summary** because its 21 cases and 11 events provide insufficient precision for a stable external performance estimate. The main-text secondary microarray evidence therefore comprises GSE14520 GPL3921 (221 cases, 85 events) and GSE116174 GPL570 (64 cases, 27 events).

## Transparency requirement

This decision was made after the canonical analysis had produced the GPL571 result. It must therefore **not** be represented as a prospectively specified exclusion or deleted from the analytical record. The stratum, source audit, score file, complete result JSON, and its performance estimate remain in the supplement, and the manuscript must state that a small GPL571 stratum was analysed descriptively but was not included in the main summary because of insufficient sample size.

## Prohibited actions

- Do not delete or overwrite `GSE14520_GPL571_M2T_EVALUATION.json`.
- Do not pool GPL571 patients with GPL3921 patients.
- Do not describe the two retained main-text strata as the only analysed external data.
- Do not use the GPL571 result to select a new model, transformation, probe rule, or cut-point.

## Main-text wording

“The main cross-platform analysis included a 221-patient GSE14520 GPL3921 stratum and a 64-patient GSE116174 GPL570 cohort. A separate 21-patient GPL571 stratum was analysed descriptively and is reported in the Supplement because its size precluded a stable main-text estimate.”
