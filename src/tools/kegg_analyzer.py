"""
KEGG pathway analysis tool for metabolic feature extraction.

This module provides utilities for analyzing metabolic pathways
using KEGG database information.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathwayResult:
    """Result of pathway analysis."""
    pathway_id: str
    pathway_name: str
    genes_in_pathway: List[str]
    enriched_genes: List[str]
    p_value: float
    adjusted_p_value: float
    effect_size: float  # e.g., NES (normalized enrichment score)
    regulation: str  # "up", "down", "mixed"


class KEGGAnalyzer:
    """
    Analyzer for KEGG metabolic pathways.

    This class provides:
    - Mapping of genes to KEGG pathways
    - Pathway enrichment analysis
    - Activity score calculation
    """

    # Pre-defined HCC-relevant metabolic pathways
    HCC_METABOLIC_PATHWAYS = {
        "hsa00010": {
            "name": "Glycolysis and Gluconeogenesis",
            "genes": ["HK1", "HK2", "HK3", "GPI", "PFKL", "PKM", "PKLR", "LDHA", "LDHB", "GCK", "PCK1", "PCK2"]
        },
        "hsa00020": {
            "name": "Citrate Cycle (TCA Cycle)",
            "genes": ["ACO1", "ACO2", "IDH1", "IDH2", "IDH3A", "IDH3B", "OGDH", "SUCLA2", "SUCLG1", "MDH1", "MDH2", "FH", "SDHA", "SDHB", "SDHC", "SDHD"]
        },
        "hsa00250": {
            "name": "Alanine, aspartate and glutamate metabolism",
            "genes": ["GPT", "GOT1", "GOT2", "GDH", "GLS", "GLS2", "ASNS", "ASS1", "ASL", "ARG1", "OTC"]
        },
        "hsa00190": {
            "name": "Oxidative phosphorylation",
            "genes": ["NDUFA1", "NDUFA2", "NDUFA4", "NDUFA5", "NDUFA6", "NDUFA7", "NDUFA8", "NDUFA9", "NDUFA10",
                     "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5", "NDUFB6", "NDUFB7", "NDUFB8", "NDUFB9", "NDUFB10",
                     "NDUFC1", "NDUFC2", "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS4", "NDUFS5", "NDUFS6", "NDUFS7", "NDUFS8",
                     "NDUFAF1", "NDUFAF2", "NDUFAF3", "NDUFAF4", "COX4I1", "COX4I2", "COX5A", "COX5B", "COX6A1", "COX6B1",
                     "COX7A1", "COX7A2", "COX7B", "COX8A", "ATP5A1", "ATP5B", "ATP5C1", "ATP5D", "ATP5E", "ATP5F1",
                     "ATP5G1", "ATP5G2", "ATP5G3", "ATP5H", "ATP5I", "ATP5J", "ATP5J2", "ATP5L", "ATP5O"]
        },
        "hsa00071": {
            "name": "Fatty acid metabolism",
            "genes": ["ACACA", "ACACB", "FASN", "ACSL1", "ACSL3", "ACSL4", "ACSL5", "ACSL6", "ACSL3", "SCD", "SCD2",
                     "ELOVL1", "ELOVL2", "ELOVL3", "ELOVL4", "ELOVL5", "ELOVL6", "ELOVL7", "HADH", "HADHA", "HADHB"]
        },
        "hsa00120": {
            "name": "Primary bile acid biosynthesis",
            "genes": ["CYP7A1", "CYP8B1", "CYP27A1", "AKR1D1", "AKR1C4", "HSD3B1", "HSD3B2", "CYP7B1", "CYP39A1"]
        },
        "hsa00480": {
            "name": "Glutathione metabolism",
            "genes": ["GCLC", "GCLM", "GSR", "GSS", "GSTM1", "GSTM2", "GSTA1", "GSTA2", "GSTA3", "GSTA4", "GSTA5",
                     "GSTP1", "GSTK1", "GSTT1", "GSTT2", "GSTZ1", "MGST1", "MGST2", "MGST3", "TXN", "TXNRD1", "TXNRD2"]
        },
        "hsa04210": {
            "name": "Apoptosis",
            "genes": ["BAX", "BAK1", "BCL2", "BCL2L1", "BCL2L11", "BID", "BAD", "PMAIP1", "BBC3", "APAF1", "CASP3",
                     "CASP7", "CASP8", "CASP9", "CASP10", "CYTC", "AIFM1", "ENDOG", "TP53"]
        },
        "hsa04115": {
            "name": "p53 signaling pathway",
            "genes": ["TP53", "CDK2", "CDK4", "CDK6", "CCNE1", "CCNE2", "CCND1", "CCND2", "CCND3", "RB1", "MDM2",
                     "MDM4", "BAX", "PUMA", "NOXA", "PTEN", "AKT1", "AKT2", "BAD", "CASP3"]
        },
        "hsa05200": {
            "name": "Pathways in cancer",
            "genes": ["RAF1", "BRAF", "KRAS", "HRAS", "NRAS", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "EGFR", "ERBB2",
                     "ERBB3", "PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3", "PTEN", "TP53", "RB1", "CDK4",
                     "CDK6", "CCND1", "MYC", "JUN", "FOS", "CTNNB1", "APC", "AXIN1", "AXIN2"]
        },
        "hsa04066": {
            "name": "HIF-1 signaling pathway",
            "genes": ["HIF1A", "EPAS1", "VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4", "PGK1", "ENO1", "LDHA",
                     "HK1", "HK2", "HK3", "SLC2A1", "SLC2A2", "SLC2A3", "PDGFRB", "KIT", "FLT1"]
        },
    }

    def __init__(
        self,
        cache_dir: str = "F:/ACM/data/kegg_cache"
    ):
        """
        Initialize KEGG analyzer.

        Args:
            cache_dir: Directory for caching pathway data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_pathways(
        self,
        gene_expression: Dict[str, float],
        top_n: int = 10
    ) -> List[PathwayResult]:
        """
        Analyze enriched pathways from gene expression data.

        Args:
            gene_expression: Dictionary of gene_id -> expression value
            top_n: Number of top pathways to return

        Returns:
            List of PathwayResult sorted by significance
        """
        results = []

        # Calculate pathway activities
        for pathway_id, pathway_info in self.HCC_METABOLIC_PATHWAYS.items():
            pathway_genes = set(pathway_info["genes"])
            expressed_genes = set(gene_expression.keys())

            # Find overlapping genes
            overlapping = pathway_genes & expressed_genes

            if len(overlapping) < 2:
                continue

            # Calculate metrics
            expressed_values = [gene_expression[g] for g in overlapping]
            mean_expr = sum(expressed_values) / len(expressed_values)

            # Calculate enrichment (simplified)
            avg_expr = sum(gene_expression.values()) / len(gene_expression) if gene_expression else 0
            fold_change = mean_expr / avg_expr if avg_expr > 0 else 1.0

            # Simplified p-value (in production, use proper statistical test)
            p_value = 0.05 if abs(fold_change - 1) > 0.3 else 0.5

            # Determine regulation
            if fold_change > 1.2:
                regulation = "up"
            elif fold_change < 0.8:
                regulation = "down"
            else:
                regulation = "mixed"

            result = PathwayResult(
                pathway_id=pathway_id,
                pathway_name=pathway_info["name"],
                genes_in_pathway=list(pathway_genes),
                enriched_genes=list(overlapping),
                p_value=p_value,
                adjusted_p_value=p_value * (len(results) + 1),  # Simple correction
                effect_size=fold_change,
                regulation=regulation
            )
            results.append(result)

        # Sort by p-value
        results.sort(key=lambda x: x.p_value)

        return results[:top_n]

    def calculate_pathway_scores(
        self,
        gene_expression: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate activity scores for all pathways.

        Args:
            gene_expression: Dictionary of gene_id -> expression value

        Returns:
            Dictionary of pathway_id -> activity score
        """
        scores = {}

        for pathway_id, pathway_info in self.HCC_METABOLIC_PATHWAYS.items():
            pathway_genes = pathway_info["genes"]
            overlapping = [g for g in pathway_genes if g in gene_expression]

            if overlapping:
                # Average expression as score
                scores[pathway_id] = sum(gene_expression[g] for g in overlapping) / len(overlapping)
            else:
                scores[pathway_id] = 0.0

        return scores

    def get_metabolic_subtype(
        self,
        gene_expression: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict metabolic subtype based on gene expression.

        Based on literature, HCC metabolic subtypes include:
        - Proliferation: High glycolysis, proliferation markers
        - Differentiation: Lower proliferation, hepatocyte-like
        - Hypoxic: Hypoxia response genes elevated

        Args:
            gene_expression: Dictionary of gene_id -> expression value

        Returns:
            Dictionary with predicted subtype and confidence
        """
        # Define subtype markers
        glycolysis_genes = ["HK2", "PKM2", "LDHA", "LDHB", "GPI"]
        hypoxia_genes = ["CA9", "VEGFA", "HIF1A", "EPAS1", "PDGFA"]
        tca_genes = ["IDH1", "IDH2", "SDHA", "FH", "MDH2"]
        glutamine_genes = ["GLS", "GLS2", "GLUD1"]

        def calc_score(genes):
            values = [gene_expression.get(g, 0) for g in genes]
            return sum(values) / len(values) if values else 0

        scores = {
            "glycolysis": calc_score(glycolysis_genes),
            "hypoxia": calc_score(hypoxia_genes),
            "tca": calc_score(tca_genes),
            "glutamine": calc_score(glutamine_genes)
        }

        # Determine subtype based on highest scores
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5

        subtype_map = {
            "glycolysis": "Proliferation/Metabolic",
            "hypoxia": "Hypoxic",
            "tca": "Oxidative",
            "glutamine": "Glutamine-addicted"
        }

        return {
            "predicted_subtype": subtype_map.get(max_type, "Mixed"),
            "subtype_confidence": float(confidence),
            "subtype_scores": scores,
            "reasoning": f"Based on gene expression patterns, this tumor shows {subtype_map.get(max_type, 'mixed')} characteristics"
        }

    def get_key_biomarkers(
        self,
        gene_expression: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Identify key metabolic biomarkers.

        Args:
            gene_expression: Dictionary of gene_id -> expression value

        Returns:
            List of key biomarkers with expression levels
        """
        # Known prognostic biomarkers in HCC
        known_markers = {
            "CA9": {"name": "Carbonic anhydrase IX", "direction": "poor", "pathway": "Hypoxia"},
            "VEGFA": {"name": "VEGF-A", "direction": "poor", "pathway": "Angiogenesis"},
            "CXCL12": {"name": "C-X-C motif chemokine 12", "direction": "context", "pathway": "Signaling"},
            "HK2": {"name": "Hexokinase 2", "direction": "poor", "pathway": "Glycolysis"},
            "PKM2": {"name": "Pyruvate kinase M2", "direction": "poor", "pathway": "Glycolysis"},
            "LDHA": {"name": "Lactate dehydrogenase A", "direction": "poor", "pathway": "Glycolysis"},
            "GLS": {"name": "Glutaminase", "direction": "poor", "pathway": "Glutamine"},
            "IDH1": {"name": "Isocitrate dehydrogenase 1", "direction": "context", "pathway": "TCA"},
            "SCD": {"name": "Stearoyl-CoA desaturase", "direction": "context", "pathway": "Lipid"},
            "FASN": {"name": "Fatty acid synthase", "direction": "poor", "pathway": "Lipid"},
        }

        avg_expr = sum(gene_expression.values()) / len(gene_expression) if gene_expression else 0
        biomarkers = []

        for gene_id, info in known_markers.items():
            if gene_id in gene_expression:
                expr = gene_expression[gene_id]
                fold_change = expr / avg_expr if avg_expr > 0 else 1.0

                # Determine if elevated/depressed
                if fold_change > 1.5:
                    status = "elevated"
                elif fold_change < 0.7:
                    status = "reduced"
                else:
                    status = "normal"

                biomarkers.append({
                    "gene": gene_id,
                    "name": info["name"],
                    "expression": float(expr),
                    "fold_change": float(fold_change),
                    "status": status,
                    "pathway": info["pathway"],
                    "prognostic_direction": info["direction"],
                    "clinical_significance": self._interpret_biomarker(gene_id, fold_change, info["direction"])
                })

        # Sort by absolute fold change
        biomarkers.sort(key=lambda x: abs(x["fold_change"] - 1), reverse=True)

        return biomarkers[:10]

    def _interpret_biomarker(
        self,
        gene: str,
        fold_change: float,
        prognostic_direction: str
    ) -> str:
        """Generate clinical interpretation for a biomarker."""
        if prognostic_direction == "poor":
            if fold_change > 1.5:
                return f"Elevated {gene} is associated with poor prognosis"
            elif fold_change < 0.7:
                return f"Reduced {gene} may indicate better prognosis"
        elif prognostic_direction == "good":
            if fold_change > 1.5:
                return f"Elevated {gene} may indicate better prognosis"
        else:
            return f"{gene} expression level has context-dependent prognostic value"

        return f"{gene} expression is within normal range"


# Global analyzer instance
_kegg_analyzer: Optional[KEGGAnalyzer] = None


def get_kegg_analyzer() -> KEGGAnalyzer:
    """Get the global KEGG analyzer instance."""
    global _kegg_analyzer
    if _kegg_analyzer is None:
        _kegg_analyzer = KEGGAnalyzer()
    return _kegg_analyzer
