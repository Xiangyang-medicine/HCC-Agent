# External Data Acquisition Status - 2026-07-27

**Status:** `PRIMARY_RNASEQ_COHORT_PENDING_SOURCE_AND_FIELD_ACCESS`

## Completed Local Work

- Phase 3B amendment v2 is frozen before external analysis.
- The TCGA-only M4 external-validation artifact and its integrity verification are complete.
- Local GSE14520 source files have been audited and are labelled exploratory cross-platform only.

## Network status and source screening

The initial workstation connection problem was resolved on 2026-07-27.  Read-only requests now succeed for GDC, GEO, cBioPortal, NCBI E-utilities, HCCDB, NODE, and UCSC Xena public endpoints.  This removes the *network* blocker, but not the much more important source/field-compatibility gate.

Completed source checks are recorded in `PHASE3B_EXTERNAL_COHORT_SCREENING_20260727.md`:

1. The retired ICGC LIRI-JP URLs now redirect to an HTML landing page.  The HCCDB mirror has expression and vital status but no OS/follow-up duration.
2. NODE project OEP000321 confirms a scientifically relevant 159-patient HBV-HCC RNA-seq cohort, but its raw transcriptomic files are access-restricted and no eligible public processed expression-plus-OS package has been acquired.
3. Public RNA-seq GEO candidates GSE144269 and GSE242315 provide expression but not the matched survival and clinical fields required for formal M4 validation.
4. GSE14520 and GSE76427 retain value as independent microarray survival cohorts, but only for a separately frozen, clearly labelled exploratory gene-only transport analysis.

## Non-Substitutable Requirement

GSE14520 cannot serve as the confirmatory cohort because it is an Affymetrix microarray dataset. It may later be analysed only as the separate exploratory analysis specified in Phase 3B amendment v2.

## Resolution Path

Obtain an independent RNA-seq HCC cohort from an official or otherwise documented source and place the original files under `data/external/primary_rnaseq/raw/`. Create `SOURCE_MANIFEST.json` before preprocessing. The Phase 3B readiness gate will remain false until that manifest is present and the source/feature/outcome gates are implemented and passed.

No external validation claim is permitted while this document remains active.
