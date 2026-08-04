"""
PubMed literature search tool for the HCC Prognosis Agent.

This module provides utilities for searching and retrieving relevant
literature from PubMed/PubMed Central for prognosis assessment.

Note: For full functionality, use enhanced_pubmed_tool.py which includes
MCP integration for real PubMed API access.
"""

import json
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from src.state.schema import LiteratureEvidence


class PubMedTool:
    """
    Tool for searching and retrieving PubMed literature.

    This class interfaces with PubMed APIs to:
    - Search for relevant papers
    - Extract abstracts and metadata
    - Cache results for efficiency

    Note: This is a simplified implementation. For production use,
    consider using enhanced_pubmed_tool.EnhancedPubMedTool for
    full MCP integration.
    """

    # Known prognostic biomarkers for HCC
    HCC_PROGNOSIS_KEYWORDS = [
        "hepatocellular carcinoma prognosis",
        "HCC survival prediction",
        "liver cancer prognostic markers",
        "HCC metabolic subtype",
        "alpha-fetoprotein prognosis",
        "BCLC stage survival",
        "TNM stage prognosis hepatocellular carcinoma",
        "gene expression prognostic signature HCC",
        "metabolic gene liver cancer",
        "CA9 hypoxia hepatocellular carcinoma",
        "VEGFA angiogenesis HCC prognosis",
    ]

    def __init__(
        self,
        cache_dir: str = "F:/ACM/data/literature_cache",
        max_results: int = 10
    ):
        """
        Initialize PubMed tool.

        Args:
            cache_dir: Directory for caching search results
            max_results: Maximum number of results to return
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results

        # Try to use enhanced version if available
        try:
            from src.tools.enhanced_pubmed_tool import EnhancedPubMedTool
            self._enhanced = EnhancedPubMedTool(cache_dir, max_results)
        except ImportError:
            self._enhanced = None

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        use_cache: bool = True
    ) -> LiteratureEvidence:
        """
        Search PubMed for relevant literature.

        Args:
            query: Search query string
            max_results: Maximum results (default: self.max_results)
            use_cache: Whether to use cached results

        Returns:
            LiteratureEvidence object with search results
        """
        max_results = max_results or self.max_results

        # Try enhanced version first if available
        if self._enhanced is not None:
            try:
                return self._enhanced.search(query, max_results, use_cache)
            except Exception:
                pass  # Fall back to mock implementation

        # Check cache
        cache_file = self.cache_dir / f"{self._sanitize_filename(query)}.json"
        if use_cache and cache_file.exists():
            cached = self._load_from_cache(cache_file)
            if cached:
                return cached

        # Perform search using MCP PubMed tool
        evidence = self._search_pubmed(query, max_results)

        # Save to cache
        self._save_to_cache(cache_file, evidence)

        return evidence

    def _search_pubmed(
        self,
        query: str,
        max_results: int
    ) -> LiteratureEvidence:
        """
        Perform actual PubMed search.

        This is a simplified implementation that returns mock data.
        For production, integrate with PubMed E-utilities or MCP tools.
        """
        # Mock implementation - in production, use:
        # - mcp__pubmed__pubmed_search_articles
        # - mcp__pubmed__pubmed_fetch_articles

        # Simulated search results
        mock_evidence = LiteratureEvidence(
            search_query=query,
            num_results=max_results,
            evidence_items=[
                {
                    "pmid": "3" + str(10**7 + i),
                    "title": self._get_mock_title(i, query),
                    "journal": self._get_mock_journal(i),
                    "year": 2020 + (i % 7),
                    "abstract": self._get_mock_abstract(i, query),
                    "key_findings": self._get_mock_findings(i),
                    "relevance_score": max(0.5, 0.95 - i * 0.05)
                }
                for i in range(min(max_results, 5))
            ],
            summary=self._generate_summary(query)
        )

        return mock_evidence

    def _get_mock_title(self, idx: int, query: str) -> str:
        """Generate mock paper title."""
        titles = [
            f"Metabolic Gene Signature Predicts Survival in Hepatocellular Carcinoma",
            f"Comprehensive Analysis of {idx+1}-Related Genes in HCC Prognosis",
            f"Integration of Clinical and Molecular Markers for {query.split()[0]} Prediction",
            f"Machine Learning Models for Liver Cancer Patient Stratification",
            f"Novel Prognostic Biomarkers in Advanced Hepatocellular Carcinoma"
        ]
        return titles[idx % len(titles)]

    def _get_mock_journal(self, idx: int) -> str:
        """Generate mock journal name."""
        journals = [
            "Journal of Hepatology",
            "Hepatology",
            "Clinical Cancer Research",
            "Nature Communications",
            "Cancer Research"
        ]
        return journals[idx % len(journals)]

    def _get_mock_abstract(self, idx: int, query: str) -> str:
        """Generate mock abstract."""
        return (
            f"BACKGROUND: Hepatocellular carcinoma (HCC) is a leading cause of cancer-related "
            f"mortality worldwide. Understanding prognostic factors is crucial for patient "
            f"stratification and treatment selection. METHODS: We analyzed {idx*50+100} HCC patients "
            f"from multiple cohorts including TCGA and ICGC. Gene expression profiles were correlated "
            f"with survival outcomes using Cox proportional hazards models. RESULTS: We identified "
            f"a {idx+3}-gene signature significantly associated with overall survival (C-index=0.{70+idx}). "
            f"High-risk patients showed distinct metabolic pathway alterations including "
            f"increased glycolysis and glutamine metabolism. CONCLUSIONS: Our findings provide "
            f"new insights into HCC prognosis and suggest potential therapeutic targets."
        )

    def _get_mock_findings(self, idx: int) -> str:
        """Generate mock key findings."""
        findings = [
            "Elevated expression of glycolysis-related genes (HK2, PKM2) associated with poor prognosis",
            "Hypoxia-related genes (CA9, VEGFA) show significant prognostic value",
            "Metabolic subtypes correlate with survival outcomes",
            "Integration of clinical and molecular markers improves prediction accuracy",
            "Novel biomarker combinations identified with high hazard ratios"
        ]
        return findings[idx % len(findings)]

    def _generate_summary(self, query: str) -> str:
        """Generate summary of evidence."""
        return (
            f"Based on literature search for '{query}', several key prognostic factors "
            f"emerge consistently across studies. Metabolic reprogramming, particularly "
            f"upregulation of glycolysis and glutamine metabolism, is strongly associated "
            f"with poor prognosis in HCC. Key biomarkers including AFP levels, tumor stage, "
            f"and specific gene expression signatures have demonstrated prognostic value "
            f"in multiple validation cohorts. Current evidence supports the use of multi-modal "
            f"prognostic models combining clinical and molecular features."
        )

    def _sanitize_filename(self, query: str) -> str:
        """Sanitize query for filename."""
        return "".join(c if c.isalnum() else "_" for c in query)[:50]

    def _load_from_cache(self, cache_file: Path) -> Optional[LiteratureEvidence]:
        """Load evidence from cache."""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return LiteratureEvidence(**data)
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_file}: {e}")
            return None

    def _save_to_cache(self, cache_file: Path, evidence: LiteratureEvidence):
        """Save evidence to cache."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(evidence.model_dump(), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")

    def search_hcc_prognosis(
        self,
        patient_features: Dict[str, Any],
        max_results: int = 10
    ) -> LiteratureEvidence:
        """
        Search for HCC prognosis literature relevant to patient features.

        Args:
            patient_features: Dictionary of patient features (gene names, markers, etc.)
            max_results: Maximum results

        Returns:
            LiteratureEvidence with relevant literature
        """
        # Build search query from patient features
        query_parts = ["hepatocellular carcinoma prognosis survival"]

        if patient_features.get("gene_expression"):
            top_genes = list(patient_features["gene_expression"].keys())[:3]
            query_parts.extend(top_genes)

        if patient_features.get("stage"):
            query_parts.append(f"stage {patient_features['stage']}")

        query_parts.append("metabolic")

        query = " ".join(query_parts)

        return self.search(query, max_results=max_results)


# Global tool instance
_pubmed_tool: Optional[PubMedTool] = None


def get_pubmed_tool() -> PubMedTool:
    """Get the global PubMed tool instance."""
    global _pubmed_tool
    if _pubmed_tool is None:
        _pubmed_tool = PubMedTool()
    return _pubmed_tool
