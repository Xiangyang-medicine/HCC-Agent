#!/usr/bin/env python3
"""Create auditable, unscored microarray inputs for Phase 3B amendment v3.

The script deliberately does not import a model, calculate a risk score, or
open any legacy risk-score/figure/result file. It can be rerun before model
derivation because all its gene operations are outcome-blind.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prognostic_engine.microarray_transport import (
    collapse_probes_to_genes,
    read_geo_platform_annotation,
    read_geo_series_matrix,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "external" / "source_downloads" / "20260727"
OUTPUT = ROOT / "experiments" / "phase3b" / "microarray_transport"
CANONICAL = ROOT / "data" / "external" / "canonical_microarray"
GSE116_CLINICAL = ROOT / "GSE116174" / "GSE116174_HCC-64-u133_plus_2_clinical_data.xls"
GSE14520_CLINICAL = ROOT / "GSE14520" / "GSE14520_Extra_Supplement.txt"


def _hash_record(path: Path, url: str) -> dict:
    return {"path": str(path.relative_to(ROOT)), "source_url": url, "sha256": sha256_file(path)}


def _matrix_sample_map(metadata: dict[str, list[str]], matrix: pd.DataFrame) -> pd.DataFrame:
    gsm = [str(column) for column in matrix.columns[1:]]
    source = metadata.get("source_name_ch1", [])
    accession = metadata.get("geo_accession", [])
    if len(gsm) != len(source) or len(gsm) != len(accession):
        raise ValueError("GEO sample metadata cardinality does not match expression columns.")
    result = pd.DataFrame({"gsm": accession, "source_sample_id": source})
    if result["gsm"].duplicated().any() or result["source_sample_id"].duplicated().any():
        raise ValueError("GEO sample metadata contains duplicate GSM or source sample IDs.")
    if set(gsm) != set(result["gsm"]):
        raise ValueError("GEO expression-column IDs do not match GEO accession metadata.")
    return result


def _audit_gse116174() -> dict:
    matrix_path = DOWNLOADS / "GSE116174_series_matrix.txt.gz"
    annotation_path = DOWNLOADS / "GPL570.annot.gz"
    matrix, metadata = read_geo_series_matrix(matrix_path)
    genes = collapse_probes_to_genes(matrix, read_geo_platform_annotation(annotation_path))
    sample_map = _matrix_sample_map(metadata, matrix)
    clinical = pd.read_excel(GSE116_CLINICAL, sheet_name="mRNA")
    clinical = clinical.rename(
        columns={
            "sampleID": "source_sample_id",
            "Follow_up time(month)": "survival_months",
            "Event_death": "event",
            "Age": "age",
            "clinstage": "clinical_stage",
        }
    )
    required = {"source_sample_id", "survival_months", "event", "age", "clinical_stage"}
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(f"GSE116174 clinical sheet missing fields: {missing}")
    clinical = clinical.loc[:, list(required)].copy()
    clinical["source_sample_id"] = clinical["source_sample_id"].astype(str)
    if clinical["source_sample_id"].duplicated().any():
        raise ValueError("GSE116174 clinical mRNA sheet has duplicate sample IDs.")
    merged = sample_map.merge(clinical, on="source_sample_id", how="left", validate="one_to_one")
    complete = merged.dropna(subset=["survival_months", "event"]).copy()
    complete["case_id"] = complete["gsm"]
    complete["cohort"] = "GSE116174"
    complete["platform"] = "GPL570"
    # The expression extraction used all official tumour arrays without reading outcome.
    retained_gsm = complete["gsm"].tolist()
    canonical_gene = genes.loc[:, retained_gsm].copy()
    clinical_output = complete.loc[:, ["case_id", "cohort", "platform", "source_sample_id", "survival_months", "event", "age", "clinical_stage"]]
    CANONICAL.mkdir(parents=True, exist_ok=True)
    gene_path = CANONICAL / "GSE116174_GPL570_15gene_unscored.csv"
    clinical_path = CANONICAL / "GSE116174_GPL570_clinical_os.csv"
    canonical_gene.to_csv(gene_path, index_label="gene")
    clinical_output.to_csv(clinical_path, index=False)
    return {
        "status": "CANONICAL_INPUT_READY_NO_SCORE_GENERATED",
        "source_files": [
            _hash_record(matrix_path, "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116174/matrix/GSE116174_series_matrix.txt.gz"),
            _hash_record(annotation_path, "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"),
            _hash_record(GSE116_CLINICAL, "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116174/suppl/GSE116174_HCC-64-u133_plus_2_clinical_data.xls.gz"),
        ],
        "sample_flow": {
            "official_expression_arrays": int(matrix.shape[1] - 1),
            "unique_expression_to_clinical_matches": int(merged["survival_months"].notna().sum()),
            "complete_os_cases": int(len(complete)),
            "events": int(complete["event"].astype(int).sum()),
            "unmatched_expression_cases": int(merged["survival_months"].isna().sum()),
        },
        "feature_mapping": {"platform": "GPL570", "all_15_genes_available": True, "collapse_rule": "median of uniquely target-mapped probes"},
        "canonical_outputs": {
            "gene_matrix": str(gene_path.relative_to(ROOT)),
            "gene_matrix_sha256": sha256_file(gene_path),
            "clinical_os": str(clinical_path.relative_to(ROOT)),
            "clinical_os_sha256": sha256_file(clinical_path),
        },
    }


def _audit_gse14520_platform(matrix_name: str, platform: str, clinical: pd.DataFrame) -> dict:
    matrix_path = DOWNLOADS / matrix_name
    matrix, metadata = read_geo_series_matrix(matrix_path)
    sample_map = _matrix_sample_map(metadata, matrix)
    merged = sample_map.merge(clinical, left_on="gsm", right_on="Affy_GSM", how="left", validate="one_to_one")
    tumour = merged.loc[merged["Tissue Type"].eq("Tumor")].copy()
    complete = tumour.dropna(subset=["Survival status", "Survival months"])
    annotation_name = f"{platform}.annot.gz"
    annotation_path = DOWNLOADS / annotation_name
    result = {
        "platform": platform,
        "matrix": _hash_record(matrix_path, f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/{matrix_name}"),
        "sample_flow": {
            "official_expression_arrays": int(matrix.shape[1] - 1),
            "clinical_matches": int(merged["Affy_GSM"].notna().sum()),
            "tumour_matches": int(len(tumour)),
            "complete_os_cases": int(len(complete)),
            "events": int(complete["Survival status"].astype(int).sum()),
        },
        "annotation_gate": {
            "required_path": str(annotation_path.relative_to(ROOT)),
            "present": annotation_path.is_file(),
            "score_generation_allowed": False,
        },
    }
    if not annotation_path.is_file():
        return result
    genes = collapse_probes_to_genes(matrix, read_geo_platform_annotation(annotation_path))
    complete = complete.copy()
    complete["case_id"] = complete["gsm"].astype(str)
    complete["cohort"] = "GSE14520"
    complete["platform"] = platform
    retained_gsm = complete["gsm"].astype(str).tolist()
    canonical_gene = genes.loc[:, retained_gsm].copy()
    clinical_output = complete.loc[:, ["case_id", "cohort", "platform", "source_sample_id", "Survival months", "Survival status"]].rename(
        columns={"Survival months": "survival_months", "Survival status": "event"}
    )
    CANONICAL.mkdir(parents=True, exist_ok=True)
    gene_path = CANONICAL / f"GSE14520_{platform}_15gene_unscored.csv"
    clinical_path = CANONICAL / f"GSE14520_{platform}_clinical_os.csv"
    canonical_gene.to_csv(gene_path, index_label="gene")
    clinical_output.to_csv(clinical_path, index=False)
    result["annotation_gate"].update(
        {"annotation_sha256": sha256_file(annotation_path), "all_15_genes_available": True, "score_generation_allowed": True}
    )
    result["status"] = "CANONICAL_INPUT_READY_NO_SCORE_GENERATED"
    result["canonical_outputs"] = {
        "gene_matrix": str(gene_path.relative_to(ROOT)),
        "gene_matrix_sha256": sha256_file(gene_path),
        "clinical_os": str(clinical_path.relative_to(ROOT)),
        "clinical_os_sha256": sha256_file(clinical_path),
    }
    return result


def _audit_gse14520() -> dict:
    clinical = pd.read_csv(GSE14520_CLINICAL, sep="\t")
    required = {"Affy_GSM", "Tissue Type", "Survival status", "Survival months"}
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(f"GSE14520 clinical supplement missing fields: {missing}")
    if clinical["Affy_GSM"].dropna().duplicated().any():
        raise ValueError("GSE14520 clinical supplement has duplicate Affy_GSM IDs.")
    return {
        "status": "SOURCE_AND_CLINICAL_AUDITED_NO_SCORE_GENERATED",
        "clinical_file": _hash_record(GSE14520_CLINICAL, "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/suppl/GSE14520_Extra_Supplement.txt.gz"),
        "platforms": [
            _audit_gse14520_platform("GSE14520-GPL571_series_matrix.txt.gz", "GPL571", clinical),
            _audit_gse14520_platform("GSE14520-GPL3921_series_matrix.txt.gz", "GPL3921", clinical),
        ],
    }


def main() -> int:
    result = {
        "protocol": "PHASE_3B_PROTOCOL_AMENDMENT_V3_MICROARRAY_TRANSPORT.md",
        "analysis_scope": "SECONDARY_EXPLORATORY_CROSS_PLATFORM_GENE_LAYER_ONLY",
        "legacy_files_imported": False,
        "risk_scores_generated": False,
        "gse116174": _audit_gse116174(),
        "gse14520": _audit_gse14520(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "MICROARRAY_SOURCE_AND_INPUT_AUDIT.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
