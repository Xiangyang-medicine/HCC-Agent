# Methodology Redesign Documentation

**Date**: 2026-07-13
**Project**: HCC Prognosis Multi-Agent LLM System
**Target**: ACM TIST Special Issue (LLM-Driven Agentic AI)

---

## 1. Background

The original implementation was found to have critical methodological flaws (see `RESEARCH_AUDIT.md`). This document defines the new architecture that addresses those issues.

---

## 2. Two-Tier Architecture

The system is divided into two distinct layers with clear responsibility boundaries:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM Multi-Agent Layer                             │
│   (Case checking, task planning, literature retrieval, reasoning,  │
│    report generation, uncertainty expression, conflict detection)   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ calls for risk scores
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Prognostic Engine                              │
│   (Trained survival model, outputs C-index, AUC, Brier Score,       │
│    calibration metrics, survival probabilities)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Responsibilities

### 3.1 Prognostic Engine

**Responsibility**: Quantitative survival prediction

| Function | Description |
|----------|-------------|
| Survival Model Training | Cox PH, DeepSurv, or similar trained on clinical + gene expression |
| Risk Score Generation | Quantitative risk scores (0-1 or log-hazard ratio) |
| Survival Probability | P(T > t) for specified time points |
| Performance Metrics | C-index (Harrell's/Uno's), time-dependent AUC, Brier Score, calibration |

**DOES NOT:**
- Make claims about LLM capabilities
- Perform literature retrieval
- Generate narrative reports

### 3.2 LLM Multi-Agent Layer

**Responsibility**: Qualitative analysis, verification, and explanation

| Agent | Function |
|-------|----------|
| Coordinator | Intent analysis, task planning, workflow orchestration |
| Feature Extraction | Validates clinical data completeness, checks for data quality issues |
| Literature Agent | PubMed retrieval for evidence, citation verification |
| Reasoning Agent | Interprets Prognostic Engine output, checks for contradictions |
| Report Generator | Synthesizes findings into human-readable report |

**DOES NOT:**
- Compute survival risk scores directly (delegates to Prognostic Engine)
- Claim C-index/AUC/Brier Score as its own metrics
- Use hardcoded gene weights as "LLM reasoning"

---

## 4. Forbidden Information Flows

The following are **NOT permitted** in the new implementation:

### 4.1 Naming Prohibition
- ❌ "LLM Agent achieves C-index = 0.76"
- ❌ "Our LLM calculates survival probability"
- ❌ "LLM Agent score" as a risk metric
- ✅ "Prognostic Engine C-index = 0.76" (if validated)
- ✅ "LLM Agent retrieves and synthesizes literature evidence"

### 4.2 Methodological Prohibition
- ❌ Hardcoded gene weights masquerading as "LLM reasoning"
- ❌ Rule-based scoring named "LLM Agent score"
- ❌ Weights optimized on validation set
- ✅ Trained survival model with proper train/validation/test split
- ✅ LLM used only for qualitative tasks (retrieval, reasoning, reporting)

### 4.3 Metric Attribution
| Metric | Must Be Attributed To |
|--------|----------------------|
| C-index | Prognostic Engine |
| Time-dependent AUC | Prognostic Engine |
| Brier Score | Prognostic Engine |
| Calibration | Prognostic Engine |
| Tool call accuracy | LLM Multi-Agent Layer |
| Hallucination rate | LLM Multi-Agent Layer |
| Citation authenticity | LLM Multi-Agent Layer |
| Report quality | LLM Multi-Agent Layer |

---

## 5. Evaluation Framework

### 5.1 Prognostic Engine Metrics
Use established libraries: `lifelines`, `scikit-survival`, `pycox`

```python
# Harrell's C-index
from lifelines.utils import concordance_index
c_index = concordance_index(durations, -risk_scores, events)

# Uno's C-index (IPCW weighted)
from sksurv.metrics import concordance_index_censored
c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    event, duration, risk_score, tau=None
)

# Time-dependent AUC
from sksurv.metrics import cumulative_dynamic_auc
auc, mean_auc = cumulative_dynamic_auc(event_train, duration_train, risk_train,
                                        event_test, duration_test, risk_test, tau)
```

### 5.2 LLM Multi-Agent Layer Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Tool call accuracy | % of tool calls that successfully retrieve correct data | > 95% |
| Citation authenticity | % of citations that can be verified in PubMed | 100% |
| Hallucination rate | % of claims without supporting evidence | < 5% |
| Self-correction rate | % of errors caught and corrected | > 50% |
| Latency | Time from query to report | < 30s |
| Cost | API calls per evaluation | Track |

---

## 6. Data Handling

### 6.1 Fail-Closed Mechanism

All data loading must implement `--data-mode real` parameter:

```python
def load_data(mode: str = "real"):
    """
    mode: 'real' - Only use verified real datasets, fail if missing
           'any'  - Fall back to mock/synthetic if real unavailable
    """
    if mode == "real":
        df = load_verified_tcga()
        if df is None:
            raise DataSourceError("Real data unavailable. Cannot proceed in --data-mode real.")
    elif mode == "any":
        df = load_verified_tcga() or load_mock_data()
```

### 6.2 Data Provenance Requirements

For each dataset, document:
- Source (TCGA, GEO, ICGC, etc.)
- Download date
- Verification method
- Any preprocessing applied
- Known limitations

---

## 7. New Directory Structure

```
ACM/
├── config/
│   └── config.py           # Configuration
├── src/
│   ├── prognostic_engine/  # NEW: Survival prediction models
│   │   ├── models/         # Cox PH, DeepSurv, etc.
│   │   ├── metrics/        # C-index, AUC, Brier Score
│   │   └── data/           # Data preprocessing for survival
│   ├── agent_system/       # NEW: LLM Multi-Agent Layer
│   │   ├── agents/         # Coordinator, Literature, Reasoning
│   │   ├── tools/          # Tool definitions
│   │   └── evaluation/     # Agent evaluation metrics
│   ├── state/
│   │   └── schema.py       # State definitions
│   └── workflow.py         # LangGraph workflow
├── data/
│   ├── real/               # NEW: Verified real data only
│   │   ├── DATA_PROVENANCE.md
│   │   └── cohort_flow.csv
│   ├── external/           # External validation datasets
│   └── cache/              # KEGG, PubMed caches
├── scripts/
│   └── real_data/          # Data download and processing
├── docs/                   # This documentation
└── tests/
```

---

## 8. File Modification List

### Files to Archive (do not delete)
- `src/evaluation/llm_agent_evaluation.py` → Move to `src/legacy/`
- `GSE116174/generate_figures_v2.py` → Move to `src/legacy/`
- `GSE14520/generate_figures.py` → Move to `src/legacy/`

### Files to Create
- `src/prognostic_engine/models/` - Survival model implementations
- `src/prognostic_engine/metrics/` - Proper metric implementations
- `src/agent_system/evaluation/` - Agent evaluation metrics

### Files to Update
- `src/workflow.py` - Integrate new two-tier architecture
- `src/main.py` - Add `--data-mode real` parameter

---

## 9. Unresolved Issues

| Issue | Status | Blocker |
|-------|--------|---------|
| Real TCGA-LIHC data verification | PENDING | Need to verify `tcga_lihc_validated.parquet` source |
| External validation data | BLOCKED | Network issues prevent GEO downloads |
| Agent evaluation pipeline | PENDING | Define specific evaluation tasks |
| LLM API integration | PENDING | Decide on specific LLM model |

---

## 10. Next Steps

1. **Verify data sources** - Confirm TCGA-LIHC parquet files are real data
2. **Set up Prognostic Engine** - Implement survival model training pipeline
3. **Design agent evaluation** - Define specific evaluation tasks for LLM layer
4. **Create baseline comparisons** - Cox PH, DeepSurv implementations

---

*Document version: 1.0*
*Created: 2026-07-13*
*Replaces: Invalid methodology documented in RESEARCH_AUDIT.md*
