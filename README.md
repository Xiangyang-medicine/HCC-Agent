# HCC Prognosis Assessment Multi-Agent System

> **⚠️ DEPRECATION WARNING (2026-07-13) ⚠️**
>
> All quantitative results in this repository (C-index, AUC, Brier Score, Kaplan-Meier curves, model performance comparisons) are **INVALID** and marked as `legacy_unvalidated`.
>
> **Critical issues identified:**
> - The "LLM Agent" was NOT using any LLM - it was a hardcoded rule-based scoring system
> - External validation datasets used pre-existing signatures, not the team's method
> - Statistical metric implementations were fundamentally incorrect
> - Weights were optimized on validation sets (data leakage)
>
> **See `RESEARCH_AUDIT.md` and `LEGACY_RESULTS_MANIFEST.md` for details.**
>
> **A complete methodological rework is underway.** See `docs/METHODOLOGY_REDESIGN.md` for the new architecture.

A multi-agent LLM system for hepatocellular carcinoma (HCC) prognosis assessment using LangGraph.

## Overview

This system implements a multi-agent architecture for analyzing patient data and providing AI-driven prognosis assessments for liver cancer patients. It integrates:

- **Metabolic pathway analysis** from gene expression data
- **Literature evidence synthesis** from PubMed
- **Risk assessment** with explainable reasoning
- **Comprehensive reports** for clinical decision support

## Project Structure

```
ACM/
├── config/
│   └── config.py           # Configuration management
├── src/
│   ├── agents/
│   │   ├── coordinator.py   # Orchestrator agent
│   │   ├── feature_extraction.py  # Metabolic feature analysis
│   │   ├── literature.py    # Literature search agent
│   │   ├── reasoning.py     # Risk assessment agent
│   │   └── report.py        # Report generation
│   ├── state/
│   │   └── schema.py        # State definitions for LangGraph
│   ├── tools/
│   │   ├── tcga_loader.py   # TCGA data loader
│   │   ├── pubmed_tool.py   # PubMed search tool
│   │   └── kegg_analyzer.py # KEGG pathway analyzer
│   ├── utils/
│   │   └── llm_client.py    # LLM API client
│   ├── workflow.py          # LangGraph workflow
│   └── main.py              # Application entry point
├── data/                    # Data directory
└── tests/                   # Test files
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your API key:

```bash
export CLAUDE_API_KEY="your-api-key"
export CLAUDE_API_BASE="https://api.xycloud-ai.com/v1"  # Optional, for proxy
```

## Usage

### Run Demo
```bash
python -m src.main --demo
```

### List Available Patients
```bash
python -m src.main --list-patients
```

### Assess a Specific Patient
```bash
python -m src.main --patient-id TCGA-CC-A7WJ --api-key YOUR_KEY
```

### Save Report to File
```bash
python -m src.main --patient-id TCGA-CC-A7WJ --output report.txt
```

## System Architecture

```
┌─────────────────────────────────────────────┐
│              User Interface                 │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          Coordinator Agent                  │
│    (Intent analysis, task planning)         │
└─────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│Feature  │   │Literature│   │Reasoning│
│Extraction│   │  Agent  │   │  Agent  │
└─────────┘   └─────────┘   └─────────┘
    │               │               │
    └───────────────┼───────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│            Report Generation                │
└─────────────────────────────────────────────┘
```

## Agents

1. **Coordinator Agent**: Orchestrates the workflow, analyzes intent, plans tasks
2. **Feature Extraction Agent**: Analyzes metabolic pathways from gene expression
3. **Literature Agent**: Searches and synthesizes PubMed evidence
4. **Reasoning Agent**: Performs risk assessment with explanations
5. **Report Agent**: Generates comprehensive clinical reports

## Technology Stack

- **LangGraph**: Multi-agent workflow orchestration
- **Anthropic Claude API**: LLM backend
- **TCGA Data**: Patient cohort data
- **KEGG**: Metabolic pathway database
- **PubMed**: Literature search

## License

MIT License
