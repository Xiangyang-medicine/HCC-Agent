# ACM TIST Publication Figures - Source Data Mapping

## Folder Structure
```
publication_figures/
├── figures/              # All publication figures (PDF)
├── source_data/          # Raw data used for figure generation
└── supplementary_tables/ # Supplementary tables (CSV)
```

---

## Figure to Source Data Mapping

### TCGA-LIHC Internal Validation Figures

| Figure | File | Source Data | Description |
|--------|------|-------------|-------------|
| **Figure 1** | `tcga_figure1_model_comparison.pdf` | `evaluation_metrics_20260709_113012.json` | Model performance comparison (C-index, AUC, Brier) |
| **Figure 2** | `tcga_figure2_risk_factors.pdf` | `risk_factors.csv` | Risk factor importance (Cox coefficients) |
| **Figure 3** | `tcga_figure3_km_by_risk.pdf` | `agent_predictions_20260709_113012.csv` + TCGA-LIHC parquet | KM curves by LLM Agent risk groups |
| **Figure 4** | `tcga_figure4_km_combined.pdf` | `agent_predictions_20260709_113012.csv` + TCGA-LIHC parquet | Combined KM (High vs Low risk) |
| **Figure 5** | `tcga_figure5_calibration.pdf` | `test_predictions_20260709_113012.csv` + `agent_predictions_20260709_113012.csv` | Calibration curves (3-year outcome) |
| **Figure 6** | `tcga_figure6_data_overview.pdf` | `data_summary.json` + `risk_factors.csv` | Dataset overview and distributions |
| **Figure 7** | `tcga_figure7_prediction_heatmap.pdf` | `test_predictions_20260709_113012.csv` + `agent_predictions_20260709_113012.csv` | Risk prediction heatmaps |

### External Validation Figures (GSE116174)

| Figure | File | Source Data | Description |
|--------|------|-------------|-------------|
| **Figure 8** | `figure_km_all_models.pdf` | `all_model_risk_scores.csv` + clinical data | KM curves for all models |
| **Figure 9** | `figure_model_comparison.pdf` | `model_comparison_results.csv` | Model discrimination and HR |
| **Figure 10** | `figure_roc_curves.pdf` | `all_model_risk_scores.csv` | ROC curves |
| **Figure 11** | `figure_score_correlation.pdf` | `all_model_risk_scores.csv` | Risk score correlation heatmap |

### External Validation Figures (GSE14520)

| Figure | File | Source Data | Description |
|--------|------|-------------|-------------|
| **Figure 12** | `figure_km_all_models.pdf` | `all_model_risk_scores.csv` + clinical data | KM curves for all models |
| **Figure 13** | `figure_model_comparison.pdf` | `model_comparison_results.csv` | Model discrimination and HR |
| **Figure 14** | `figure_roc_curves.pdf` | `all_model_risk_scores.csv` | ROC curves |
| **Figure 15** | `figure_score_correlation.pdf` | `all_model_risk_scores.csv` | Risk score correlation heatmap |

### Combined Supplementary Figures

| Figure | File | Source Data | Description |
|--------|------|-------------|-------------|
| **Figure S1** | `figure_calibration_curves.pdf` | Both external datasets | Calibration curves comparison |
| **Figure S2** | `figure_time_dependent_cindex.pdf` | Both external datasets | Time-dependent C-index |
| **Figure S3** | `figure_decision_curve.pdf` | Both external datasets | Decision curve analysis |
| **Figure S4** | `figure_combined_validation.pdf` | Both external datasets | Side-by-side validation |
| **Figure S5** | `figure_precision_recall.pdf` | Both external datasets | Precision-recall curves |

---

## Source Data Files

### TCGA-LIHC Internal Validation
| File | Description | Columns |
|------|-------------|---------|
| `evaluation_metrics_20260709_113012.json` | Performance metrics for all models | c_index, AUC, calibration, brier |
| `model_comparison_20260709_113012.csv` | Model comparison summary | Model, C-index, AUC, etc. |
| `risk_factors.csv` | Cox model coefficients | feature, coefficient, abs_coef |
| `test_predictions_20260709_113012.csv` | Test set predictions | Simple LR, Cox PH, DeepSurv, time, event |
| `agent_predictions_20260709_113012.csv` | LLM Agent predictions | patient_id, risk_level, risk_score, etc. |
| `data_summary.json` | Dataset characteristics | n_patients, n_train, n_test, features |

### External Validation (GSE116174)
| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Model performance metrics |
| `all_model_risk_scores.csv` | Risk scores for all samples |

### External Validation (GSE14520)
| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Model performance metrics |
| `all_model_risk_scores.csv` | Risk scores for all samples |

---

## Supplementary Tables

| Table | File | Content |
|-------|------|---------|
| **Table S1** | `SuppTable1_Model_Performance.csv` | Complete model performance metrics |
| **Table S2** | `SuppTable2_Risk_Factors.csv` | Cox model coefficients and HR |
| **Table S3** | `SuppTable3_Dataset_Characteristics.csv` | Dataset demographics and features |
| **Table S4** | `SuppTable4_External_Validation.csv` | External validation results |
| **Table S5** | `SuppTable5_LLM_Agent_Predictions.csv` | Individual patient predictions |
| **Table S6** | `SuppTable6_Gene_Expression_Statistics.csv` | Metabolic gene expression stats |
| **Table S7** | `SuppTable7_Model_Predictions.csv` | Cross-model prediction comparison |
| **Table S8** | `SuppTable8_KM_Statistics.csv` | Kaplan-Meier statistics by risk group |

---

## Key Results Summary

### Internal Validation (TCGA-LIHC, N=371)
| Model | C-index | AUC (1yr/3yr/5yr) | Brier Score |
|-------|---------|-------------------|-------------|
| LLM Agent | 0.659 | 0.849/0.675/0.695 | 0.507 |
| DeepSurv | 0.667 | 0.699/0.720/0.693 | 0.547 |
| Cox PH | 0.668 | 0.493/0.695/0.702 | 0.360 |
| Simple LR | 0.597 | 0.205/0.685/0.586 | 0.387 |

### External Validation
| Dataset | N | Model | C-index | HR (95% CI) |
|---------|---|-------|---------|-------------|
| GSE116174 | 64 | LLM Agent | ~0.76 | ~5.98 |
| GSE14520 | 247 | Metabolic Sig | ~0.60 | varies |

---

*Generated: 2026-07-11*
