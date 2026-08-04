#!/usr/bin/env python3
"""
Download External Validation Datasets from GEO

Downloads GSE116174 and GSE14520 for external validation.

Usage:
    python scripts/real_data/download_geo.py --dataset GSE116174
    python scripts/real_data/download_geo.py --dataset all

Requirements:
    GEOparse, pandas, openpyxl (for some datasets)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import GEOparse
except ImportError:
    print("Error: GEOparse not installed")
    print("Install with: pip install GEOparse")
    sys.exit(1)

# Dataset configurations
DATASETS = {
    'GSE116174': {
        'title': 'HCC-64-u133_plus_2',
        'platform': 'GPL570',  # HG-U133_Plus_2
        'n_samples': 64,
        'description': 'HCC patients with survival data',
        'clinical_file': 'GSE116174_HCC-64-u133_plus_2_clinical_data.xls',
        'metabolic_genes': ['HK2', 'LDHA', 'LDHB', 'GLS', 'GLUD1', 'CA9', 'VEGFA', 'HIF1A', 'MYC', 'FASN', 'SCD', 'GPI', 'PFKL']
    },
    'GSE14520': {
        'title': 'GPL3921',
        'platform': 'GPL3921',  # Affymetrix Human Genome U133A
        'n_samples': 247,
        'description': 'HCC patients with TNM staging and survival',
        'clinical_file': 'GSE14520_Extra_Supplement.txt',
        'metabolic_genes': ['HK2', 'PKM', 'LDHA', 'LDHB', 'GLS', 'GLUD1', 'CA9', 'VEGFA', 'HIF1A', 'MYC', 'FASN', 'SCD', 'GPI', 'PFKL']
    }
}

def download_gse116174(output_dir: Path) -> dict:
    """Download and process GSE116174."""
    print("\n" + "=" * 60)
    print("Downloading GSE116174")
    print("=" * 60)

    cache_dir = output_dir / "GSE116174_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download from GEO
        print("  Fetching from GEO...")
        geo = GEOparse.get_GEO(geo='GSE116174', destdir=str(cache_dir), silent=False)

        # Get expression data
        gpl = list(geo.gpls.values())[0]
        annotation = gpl.table[['ID', 'Gene Symbol']].copy()

        # Build expression matrix
        expression_data = {}
        for gsm_name, gsm in geo.gsms.items():
            expression_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']

        expr_df = pd.DataFrame(expression_data)
        expr_df = expr_df.join(annotation.set_index('ID'))
        gene_expr = expr_df.groupby('Gene Symbol').mean().T

        print(f"  Expression data: {gene_expr.shape[0]} samples, {gene_expr.shape[1]} genes")

        # Load clinical data
        clinical_path = cache_dir / 'GSE116174_HCC-64-u133_plus_2_clinical_data.xls'
        if clinical_path.exists():
            clinical = pd.read_excel(clinical_path)
            print(f"  Clinical data: {len(clinical)} samples")

            # Check available genes
            available_genes = [g for g in DATASETS['GSE116174']['metabolic_genes'] if g in gene_expr.columns]
            missing_genes = [g for g in DATASETS['GSE116174']['metabolic_genes'] if g not in gene_expr.columns]

            result = {
                'status': 'success',
                'dataset': 'GSE116174',
                'n_samples': len(clinical),
                'n_genes': len(gene_expr.columns),
                'available_genes': available_genes,
                'missing_genes': missing_genes,
                'cache_dir': str(cache_dir)
            }
        else:
            result = {
                'status': 'partial',
                'dataset': 'GSE116174',
                'message': 'Expression downloaded but clinical file not found',
                'cache_dir': str(cache_dir)
            }

    except Exception as e:
        result = {
            'status': 'error',
            'dataset': 'GSE116174',
            'error': str(e)
        }
        print(f"  Error: {e}")

    return result

def download_gse14520(output_dir: Path) -> dict:
    """Download and process GSE14520."""
    print("\n" + "=" * 60)
    print("Downloading GSE14520")
    print("=" * 60)

    cache_dir = output_dir / "GSE14520_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download from GEO
        print("  Fetching from GEO...")
        geo = GEOparse.get_GEO(geo='GSE14520', destdir=str(cache_dir), silent=False)

        # Get expression data
        gpl = list(geo.gpls.values())[0]
        annotation = gpl.table[['ID', 'Gene Symbol']].copy()

        # Build expression matrix
        expression_data = {}
        for gsm_name, gsm in geo.gsms.items():
            expression_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']

        expr_df = pd.DataFrame(expression_data)
        expr_df = expr_df.join(annotation.set_index('ID'))
        gene_expr = expr_df.groupby('Gene Symbol').mean().T

        print(f"  Expression data: {gene_expr.shape[0]} samples, {gene_expr.shape[1]} genes")

        # Load clinical data (usually in supplementary file)
        clinical_path = cache_dir / 'GSE14520_Extra_Supplement.txt'
        if clinical_path.exists():
            clinical = pd.read_csv(clinical_path, sep='\t')
            print(f"  Clinical data: {len(clinical)} samples")

            available_genes = [g for g in DATASETS['GSE14520']['metabolic_genes'] if g in gene_expr.columns]
            missing_genes = [g for g in DATASETS['GSE14520']['metabolic_genes'] if g not in gene_expr.columns]

            result = {
                'status': 'success',
                'dataset': 'GSE14520',
                'n_samples': len(clinical),
                'n_genes': len(gene_expr.columns),
                'available_genes': available_genes,
                'missing_genes': missing_genes,
                'cache_dir': str(cache_dir)
            }
        else:
            result = {
                'status': 'partial',
                'dataset': 'GSE14520',
                'message': 'Expression downloaded but clinical file not found',
                'cache_dir': str(cache_dir)
            }

    except Exception as e:
        result = {
            'status': 'error',
            'dataset': 'GSE14520',
            'error': str(e)
        }
        print(f"  Error: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Download GEO datasets for external validation")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['GSE116174', 'GSE14520', 'all'],
                        help='Dataset to download')
    parser.add_argument('--output-dir', type=str,
                        default='F:/ACM/data/external',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("GEO External Validation Dataset Download")
    print("=" * 60)

    results = []

    if args.dataset in ['GSE116174', 'all']:
        results.append(download_gse116174(output_dir))

    if args.dataset in ['GSE14520', 'all']:
        results.append(download_gse14520(output_dir))

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for r in results:
        status_icon = '✓' if r['status'] == 'success' else '⚠' if r['status'] == 'partial' else '✗'
        print(f"\n{status_icon} {r['dataset']}: {r['status']}")

        if r['status'] == 'success':
            print(f"  Samples: {r['n_samples']}")
            print(f"  Genes: {r['n_genes']}")
            print(f"  Available metabolic genes: {len(r['available_genes'])}")
            if r['missing_genes']:
                print(f"  Missing genes: {r['missing_genes']}")

        if 'error' in r:
            print(f"  Error: {r['error']}")

    # Save results
    import json
    results_path = output_dir / "download_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
