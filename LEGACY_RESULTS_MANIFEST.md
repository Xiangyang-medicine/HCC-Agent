# Legacy Results Manifest

**Date**: 2026-07-13
**Project**: HCC Prognosis Multi-Agent LLM System
**Status**: ALL QUANTITATIVE RESULTS MARKED AS `legacy_unvalidated`

---

## Preamble

All files in this manifest contain results derived from **methodologically invalid** approaches identified in `RESEARCH_AUDIT.md`. These files MUST NOT be used in any publication, presentation, or formal report. They are preserved for audit trail purposes only.

### Critical Issues Summary

| Issue | Severity | Impact |
|-------|----------|--------|
| "LLM Agent" is rule-based scoring | CRITICAL | All LLM Agent performance claims are false |
| External validation uses pre-existing signatures | CRITICAL | Cannot attribute results to this team's method |
| Weights optimized on validation sets | CRITICAL | Data leakage, inflated metrics |
| Incorrect survival analysis metrics | HIGH | C-index, AUC, Brier Score values unreliable |
| Mock data without proper labeling | MEDIUM | Transparency issue |

---

## File Inventory

### A. Publication Figures (PDF)

| File | Generation Script | Data Source | Issues | Classification |
|------|-------------------|-------------|--------|----------------|
| `publication_figures/figures/tcga_figure1_model_comparison.pdf` | `generate_figures.py` | `evaluation_metrics_*.json` | Mock LLM (`use_mock=True`), incorrect metrics | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure2_risk_factors.pdf` | `generate_figures.py` | `risk_factors.csv` | Cox coefficients from invalid model | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure3_km_by_risk.pdf` | `generate_figures.py` | `agent_predictions_*.csv` | Mock agent predictions | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure4_km_combined.pdf` | `generate_figures.py` | `agent_predictions_*.csv` | Mock agent predictions | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure5_calibration.pdf` | `generate_figures.py` | `agent_predictions_*.csv` | Incorrect Brier Score implementation | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure6_data_overview.pdf` | `generate_figures.py` | `data_summary.json` | Uses real TCGA data (may retain) | `legacy_unvalidated` |
| `publication_figures/figures/tcga_figure7_prediction_heatmap.pdf` | `generate_figures.py` | `agent_predictions_*.csv` | Mock agent predictions | `legacy_unvalidated` |
| `GSE116174/figure_km_all_models.pdf` | `generate_figures_v2.py` | `all_model_risk_scores.csv` | Weighted ensemble mislabeled as LLM Agent | `legacy_unvalidated` |
| `GSE116174/figure_model_comparison.pdf` | `generate_figures_v2.py` | `model_comparison_results.csv` | Weights optimized on validation set | `legacy_unvalidated` |
| `GSE116174/figure_roc_curves.pdf` | `generate_figures_v2.py` | `all_model_risk_scores.csv` | Based on rule-based scoring | `legacy_unvalidated` |
| `GSE116174/figure_score_correlation.pdf` | `generate_figures_v2.py` | `all_model_risk_scores.csv` | Based on rule-based scoring | `legacy_unvalidated` |
| `GSE14520/figure_km_all_models.pdf` | `generate_figures.py` | `all_model_risk_scores.csv` | Uses pre-existing Metastasis Signature | `legacy_unvalidated` |
| `GSE14520/figure_model_comparison.pdf` | `generate_figures.py` | `model_comparison_results.csv` | Derived from external signature | `legacy_unvalidated` |
| `GSE14520/figure_roc_curves.pdf` | `generate_figures.py` | `all_model_risk_scores.csv` | Based on external signature | `legacy_unvalidated` |
| `GSE14520/figure_score_correlation.pdf` | `generate_figures.py` | `all_model_risk_scores.csv` | Based on external signature | `legacy_unvalidated` |

**Sub-figures in `publication_figures/figures/`**: Additional copies of above figures stored for publication package.

---

### B. Evaluation Metrics (JSON)

| File | Contents | Issues | Classification |
|------|----------|--------|----------------|
| `experiments/evaluation_metrics_20260709_113012.json` | C-index, AUC, Brier Score for all models | Mock agent + incorrect metrics | `legacy_unvalidated` |
| `experiments/evaluation_metrics_20260709_113012_*.json` | Timestamped copies | Same | `legacy_unvalidated` |
| `experiments/data_summary.json` | Dataset characteristics | Uses real TCGA data | `legacy_unvalidated` |

---

### C. Prediction Results (CSV)

| File | Contents | Issues | Classification |
|------|----------|--------|----------------|
| `experiments/agent_predictions_20260709_113012.csv` | LLM Agent predictions | Rule-based scoring, not LLM | `legacy_unvalidated` |
| `experiments/test_predictions_20260709_113012.csv` | Baseline model predictions | Partially valid (non-LLM models) | `legacy_unvalidated` |
| `experiments/model_comparison_20260709_113012.csv` | Model comparison | Includes invalid LLM Agent | `legacy_unvalidated` |
| `experiments/risk_factors.csv` | Cox model coefficients | From invalid evaluation | `legacy_unvalidated` |
| `GSE116174/all_model_risk_scores.csv` | Risk scores for all models | Weighted ensemble mislabeled | `legacy_unvalidated` |
| `GSE116174/model_comparison_results.csv` | External validation metrics | Weights optimized on validation | `legacy_unvalidated` |
| `GSE14520/all_model_risk_scores.csv` | Risk scores for all models | Uses pre-existing signature | `legacy_unvalidated` |
| `GSE14520/model_comparison_results.csv` | External validation metrics | Uses external signature | `legacy_unvalidated` |

---

### D. Supplementary Tables (CSV)

| File | Contents | Issues | Classification |
|------|----------|--------|----------------|
| `publication_figures/supplementary_tables/SuppTable1_Model_Performance.csv` | Complete model metrics | Includes invalid LLM Agent | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable2_Risk_Factors.csv` | Cox coefficients and HR | From invalid evaluation | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable3_Dataset_Characteristics.csv` | Dataset demographics | Uses real TCGA data (may retain) | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable4_External_Validation.csv` | External validation results | Invalid methods | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable5_LLM_Agent_Predictions.csv` | Individual predictions | Rule-based scoring | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable6_Gene_Expression_Statistics.csv` | Gene expression stats | Uses real TCGA data (may retain) | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable7_Model_Predictions.csv` | Cross-model predictions | Partially valid | `legacy_unvalidated` |
| `publication_figures/supplementary_tables/SuppTable8_KM_Statistics.csv` | KM statistics | Based on invalid predictions | `legacy_unvalidated` |

---

### E. Documentation Files

| File | Contents | Issues | Classification |
|------|----------|--------|----------------|
| `publication_figures/SOURCE_DATA_MAPPING.md` | Figure-to-data mapping | Documents invalid figures | `legacy_unvalidated` |

---

## Classification Key

| Label | Meaning |
|-------|---------|
| `legacy_unvalidated` | Do NOT use in publications. Requires complete rework. |
| `legacy_unvalidated_may_retain` | Contains real data but derived from invalid analysis. Re-process with new methods. |
| `valid_preserve` | Real data that can be retained and re-analyzed with proper methods. |

---

## Files Potentially Retainable (Real Data Only)

| File | Reason for Retention |
|------|---------------------|
| `data/tcga_lihc_validated.parquet` | Likely real TCGA-LIHC data (371 patients) |
| `data/tcga_lihc_realistic.parquet` | Realistic training data (requires verification) |
| `publication_figures/supplementary_tables/SuppTable3_Dataset_Characteristics.csv` | TCGA demographics (re-run with verification) |
| `publication_figures/supplementary_tables/SuppTable6_Gene_Expression_Statistics.csv` | TCGA gene expression (re-run with verification) |

---

## Forbidden Claims (Based on Legacy Files)

The following claims MUST NOT be made in any publication:

1. "LLM Agent achieves C-index of X" - The "LLM Agent" was not using any LLM
2. "LLM Agent achieves AUC of X" - Rule-based scoring was used
3. "LLM Agent external validation C-index = 0.76" - Weighted ensemble was used
4. "GSE14520 validated with LLM Agent" - Pre-existing Metastasis Signature was used
5. "Metabolic genes predict HCC prognosis" - Univariate p-values all > 0.05
6. "LLM Agent integrates literature evidence" - No evidence of real literature integration

---

## Required Actions Before Any Publication

1. Mark ALL current figures as `legacy_unvalidated`
2. Remove or rename all files containing "LLM Agent" performance claims
3. Re-run all survival analysis with proper implementations (lifelines, scikit-survival)
4. Clearly separate Prognostic Engine metrics from LLM Agent evaluation
5. Document which data sources are real vs mock

---

*Manifest generated: 2026-07-13*
*Total files catalogued: 37 PDFs, 104+ CSVs, 43+ JSONs, 1 DOCX*
*Files requiring rework: ALL quantitative performance claims*
