# Phase 3B external-cohort screening record (2026-07-27)

## Purpose and decision rule

This record distinguishes a cohort that is merely relevant to HCC from one that can validly evaluate the frozen Phase 3A candidate, M4 (combined random survival forest).  A cohort may be used for a survival-performance analysis only when all of the following are traceable and available before scoring:

1. independent patient cohort and primary-tumour selection;
2. expression values for all 15 locked metabolic genes, with a documented feature mapping;
3. overall-survival duration and event indicator, matched to expression at patient level;
4. age, stage, and grade (or a pre-specified, documented handling of a missing predictor);
5. an expression-scale transport procedure fixed without external outcomes; and
6. source URLs, download date, checksums, and an auditable sample flow.

The M4 artifact must not be modified, retuned, recalibrated using outcomes, or scored on a partially compatible cohort.  A cohort that fails a criterion is retained as a screened candidate, not a negative validation result.

## Screened cohorts

| Candidate | Modality and independent population | Evidence/source checked | Current decision | Reason |
|---|---|---|---|---|
| ICGC LIRI-JP / HCCDB18 | Japanese HCC RNA-seq; HCCDB mirrors normalized expression and patient annotations | HCCDB download page and `HCCDB18_mRNA_level3.zip`, `HCCDB18.patient.zip` | **Not ready for formal M4 survival validation** | The accessible HCCDB patient table has age, sex, viral status, TNM-related fields and vital status, but no OS/follow-up duration. The retired ICGC endpoint now redirects to an HTML landing page. Therefore C-index/AUC/IBS cannot be calculated. |
| CHCC-HBV / NODE OEP000321 | 159 paired HBV-HCC cases; RNA-seq (HiSeq X Ten) | NODE public metadata API, project OEP000321; Cell 2019, DOI `10.1016/j.cell.2019.08.052` | **Access-restricted candidate** | NODE confirms 316 transcriptomic samples (648 FASTQ files), but marks all FASTQ files restricted. No public processed matrix and matched OS table have been acquired. Do not bypass access controls. |
| GSE144269 (Mongolian HCC) | 70 paired HCC tumour/non-tumour samples; RNA-seq | NCBI GEO series matrix and counts/voom supplementary-file listing | **Not a prognosis-validation cohort from public record** | Public GEO data provide tumour labels and expression but no OS duration/event or the required clinical predictors in the series annotation. |
| GSE14520 (NCI/FULCI HCC) | Chinese HCC; Affymetrix microarray | Local raw archive and GEO annotation audit at `experiments/phase3b/gse14520/SOURCE_AUDIT.json` | **Exploratory gene-only transport candidate** | It has independently sourced expression and survival data, but is a microarray platform and is not a legitimate direct input to the frozen RNA-seq-plus-clinical M4 artifact. Legacy signatures/weights remain excluded. |
| GSE76427 | HCC microarray cohort with OS fields | GEO accession and HCCDB17 patient-table mirror | **Exploratory gene-only transport candidate** | OS event/time fields are present, but platform is microarray and the available annotation does not establish complete M4 predictor compatibility. |
| GSE54236 | Italian HCC microarray cohort | GEO accession and HCCDB12 patient-table mirror | **Not ready** | Accessible HCCDB clinical mirror exposes sex and tumour doubling time, not a matched OS duration/event table suitable for formal survival metrics. |

## Consequences for the paper

1. **Do not report external M4 performance yet.**  The Phase 3A internal result remains the only completed M4 performance evaluation.
2. The primary external-validation target remains an independently acquired bulk RNA-seq cohort with OS time/event and complete M4 predictors.  LIRI-JP can be reconsidered only if the original clinical duration table is obtained from an authorized, reproducible source.
3. GSE14520 and GSE76427 are useful, but only after a separately frozen, gene-only cross-platform transport model and a protocol amendment are approved.  Such an analysis would be labelled *secondary/exploratory*; it cannot be presented as external validation of M4.
4. CHCC-HBV is scientifically attractive, but access restrictions are a data-access issue, not an invitation to use an unverified mirror.  A reproducible controlled-access application or an openly released processed matrix with linked outcomes is required.

## Next operational gate

The Phase 3B readiness gate remains `PHASE3B_DATA_AND_ARTIFACT_PENDING`.  It can change to ready only after one candidate meets all six criteria in the decision rule above.  Until then, only source audits and no-outcome feature compatibility checks are permitted.

## Source links

- HCCDB download page: <http://lifeome.net/database/hccdb/download.html>
- GEO GSE144269: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144269>
- GEO GSE14520: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520>
- GEO GSE76427: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76427>
- GEO GSE54236: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54236>
- NODE OEP000321: <https://www.biosino.org/node/project/detail/OEP000321>
