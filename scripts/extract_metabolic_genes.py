#!/usr/bin/env python3
"""
Extract 15 metabolic genes from downloaded TCGA-LIHC expression files.
"""
import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Constants
RAW_DIR = Path("data/raw/gdc/20260713")
EXPR_DIR = RAW_DIR / "raw_expression"
OUTPUT_DIR = Path("data/processed/gdc/20260713")
MANIFEST_PATH = RAW_DIR / "gdc_manifest_primary_tumor.tsv"
CASES_PATH = RAW_DIR / "cases_complete_response.json"

# 15 Metabolic Genes with their gene symbols
METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",  # Glycolysis
    "GLS", "GLUD1",  # Glutamine
    "FASN", "SCD",  # Lipogenesis
    "CA9", "VEGFA", "HIF1A",  # Hypoxia
    "MYC", "CTNNB1"  # Oncogenic
]

def load_manifest():
    """Load manifest file."""
    downloads = []
    with open(MANIFEST_PATH, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                downloads.append({
                    'file_id': parts[0],  # UUID
                    'filename': parts[1],  # Full filename
                    'md5': parts[2],
                    'size': parts[3]
                })
    return downloads

def load_cases():
    """Load cases data."""
    with open(CASES_PATH, 'r') as f:
        return json.load(f)

def parse_ensembl_gene_id(gene_id):
    """Extract gene symbol from Ensembl ID like ENSG00000067225.18."""
    if '.' in gene_id:
        return gene_id.split('.')[0]
    return gene_id

def extract_genes_from_file(filepath, manifest_md5):
    """Extract 15 metabolic genes from a single expression file."""
    results = {
        'filename': filepath.name,
        'file_id': filepath.stem.split('.')[0],
        'status': 'success',
        'gene_counts': {},
        'error': None
    }

    try:
        # Read file
        with open(filepath, 'r') as f:
            content = f.read()

        # Verify MD5
        actual_md5 = hashlib.md5(content.encode()).hexdigest()
        if actual_md5 != manifest_md5:
            results['status'] = 'md5_mismatch'
            results['error'] = f"Expected {manifest_md5}, got {actual_md5}"
            return results

        # Parse TSV
        lines = content.strip().split('\n')

        # Skip comment lines starting with #
        data_lines = [l for l in lines if not l.startswith('#')]
        if not data_lines:
            results['status'] = 'no_data'
            results['error'] = "File contains only comments"
            return results

        header = data_lines[0].split('\t')

        # Find gene_name and unstranded columns (counts)
        gene_name_col = None
        counts_col = None
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower == 'gene_name':
                gene_name_col = i
            if col_lower == 'unstranded':
                counts_col = i

        if gene_name_col is None or counts_col is None:
            results['status'] = 'missing_columns'
            results['error'] = f"gene_name_col={gene_name_col}, counts_col={counts_col}"
            return results

        # Extract metabolic genes
        found_genes = set()
        for line in data_lines[1:]:
            parts = line.split('\t')
            if len(parts) <= max(gene_name_col, counts_col):
                continue

            gene_symbol = parts[gene_name_col].strip()

            if gene_symbol in METABOLIC_GENES:
                try:
                    count = float(parts[counts_col])
                    results['gene_counts'][gene_symbol] = count
                    found_genes.add(gene_symbol)
                except ValueError:
                    pass

        # Check if all 15 genes found
        missing = set(METABOLIC_GENES) - found_genes
        if missing:
            results['status'] = 'incomplete'
            results['missing_genes'] = list(missing)

        return results

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        return results

def build_sample_to_file_mapping():
    """Build mapping from sample_id to expression file."""
    # This would require parsing the filename structure
    # TCGA filenames typically follow: UUID.rna_seq.augmented_star_gene_counts.tsv
    # We need to match by file_id from manifest
    manifest = load_manifest()
    return {m['file_id']: m for m in manifest}

def main():
    print("=" * 60)
    print("TCGA-LIHC Gene Extraction")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest and cases
    manifest = load_manifest()
    cases_data = load_cases()

    print(f"\nManifest files: {len(manifest)}")
    print(f"Metabolic genes: {len(METABOLIC_GENES)}")

    # Get all downloaded files
    downloaded_files = list(EXPR_DIR.glob("*.tsv"))
    print(f"Downloaded files: {len(downloaded_files)}")

    # Create manifest lookup by filename
    manifest_lookup = {m['filename']: m for m in manifest}

    # Process files
    print("\n" + "=" * 60)
    print("Extracting genes from expression files...")
    print("=" * 60)

    all_results = []
    success_count = 0
    incomplete_count = 0
    error_count = 0

    for i, filepath in enumerate(downloaded_files):
        if (i + 1) % 50 == 0:
            print(f"Processing [{i+1}/{len(downloaded_files)}]...")

        # Match by filename
        if filepath.name not in manifest_lookup:
            continue

        manifest_entry = manifest_lookup[filepath.name]
        result = extract_genes_from_file(filepath, manifest_entry['md5'])
        result['file_id'] = manifest_entry['file_id']
        all_results.append(result)

        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'incomplete':
            incomplete_count += 1
        else:
            error_count += 1

    print(f"\nExtraction Summary:")
    print(f"  Success: {success_count}")
    print(f"  Incomplete (missing some genes): {incomplete_count}")
    print(f"  Errors: {error_count}")

    # Save extraction results
    extraction_results_path = OUTPUT_DIR / "extraction_results.json"
    with open(extraction_results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_files': len(all_results),
            'success': success_count,
            'incomplete': incomplete_count,
            'errors': error_count,
            'metabolic_genes': METABOLIC_GENES,
            'results': all_results
        }, f, indent=2)
    print(f"\nExtraction results saved to: {extraction_results_path}")

    # Build expression matrix
    print("\n" + "=" * 60)
    print("Building expression matrix...")
    print("=" * 60)

    # Create expression matrix
    expression_data = {}
    for result in all_results:
        if result['status'] in ['success', 'incomplete']:
            expression_data[result['file_id']] = result['gene_counts']

    expr_df = pd.DataFrame.from_dict(expression_data, orient='index')
    expr_df.index.name = 'file_id'

    # Save counts matrix
    counts_path = OUTPUT_DIR / "tcga_lihc_expression_counts.parquet"
    expr_df.to_parquet(counts_path)
    print(f"Expression counts saved to: {counts_path}")
    print(f"Matrix shape: {expr_df.shape}")

    # Summary statistics
    print(f"\nGene detection summary:")
    for gene in METABOLIC_GENES:
        if gene in expr_df.columns:
            detected = expr_df[gene].notna().sum()
            print(f"  {gene}: {detected}/{len(expr_df)} samples ({detected/len(expr_df)*100:.1f}%)")

    # Create gene mapping
    print("\n" + "=" * 60)
    print("Creating gene mapping...")
    print("=" * 60)

    # Parse first file to get Ensembl IDs
    sample_file = list(EXPR_DIR.glob("*.tsv"))[0]
    gene_mapping = {}

    with open(sample_file, 'r') as f:
        header = f.readline().split('\t')
        for line in f:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            raw_gene_id = parts[0]
            ensembl_id = raw_gene_id.split('.')[0] if '.' in raw_gene_id else raw_gene_id
            if ensembl_id in METABOLIC_GENES:
                gene_mapping[ensembl_id] = {
                    'ensembl_id': ensembl_id,
                    'gene_symbol': ensembl_id,
                    'raw_id': raw_gene_id
                }

    gene_mapping_df = pd.DataFrame.from_dict(gene_mapping, orient='index')
    gene_mapping_path = OUTPUT_DIR / "gene_mapping.csv"
    gene_mapping_df.to_csv(gene_mapping_path)
    print(f"Gene mapping saved to: {gene_mapping_path}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
