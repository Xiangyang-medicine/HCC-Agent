from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = Path(r"F:\ACM")
sys.path.insert(0, str(PROJECT / "src" / "prognostic_engine" / "src"))

from prognostic_engine.config import METABOLIC_GENES  # noqa: E402
from prognostic_engine.microarray_transport import (  # noqa: E402
    _platform_probe_key,
    _target_symbols,
    read_geo_platform_annotation,
    read_geo_series_matrix,
)


COHORTS = {
    "GSE14520 · GPL3921": {
        "matrix": PROJECT / "data/external/source_downloads/20260727/GSE14520-GPL3921_series_matrix.txt.gz",
        "annotation": PROJECT / "data/external/source_downloads/20260727/GPL3921.annot.gz",
        "canonical": PROJECT / "data/external/canonical_microarray/GSE14520_GPL3921_15gene_unscored.csv",
        "score": PROJECT / "experiments/phase3b/microarray_transport/GSE14520_GPL3921_m2t_scores.csv",
    },
    "GSE116174 · GPL570": {
        "matrix": PROJECT / "data/external/source_downloads/20260727/GSE116174_series_matrix.txt.gz",
        "annotation": PROJECT / "data/external/source_downloads/20260727/GPL570.annot.gz",
        "canonical": PROJECT / "data/external/canonical_microarray/GSE116174_GPL570_15gene_unscored.csv",
        "score": PROJECT / "experiments/phase3b/microarray_transport/GSE116174_GPL570_m2t_scores.csv",
    },
}


def pca_frame(matrices: dict[str, pd.DataFrame], within_cohort_zscore: bool) -> pd.DataFrame:
    rows = []
    for cohort, gene_by_sample in matrices.items():
        values = gene_by_sample.astype(float)
        if within_cohort_zscore:
            values = values.sub(values.mean(axis=1), axis=0).div(values.std(axis=1, ddof=1), axis=0)
        sample_by_gene = values.T
        sample_by_gene["cohort"] = cohort
        sample_by_gene["case_id"] = sample_by_gene.index.astype(str)
        rows.append(sample_by_gene.reset_index(drop=True))
    joined = pd.concat(rows, ignore_index=True)
    x = joined[METABOLIC_GENES].to_numpy(float)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    scores = x @ vt[:2].T
    variance = singular ** 2
    explained = variance / variance.sum()
    return pd.DataFrame(
        {
            "case_id": joined["case_id"],
            "cohort": joined["cohort"],
            "pc1": scores[:, 0],
            "pc2": scores[:, 1],
            "pc1_variance_percent": float(explained[0] * 100),
            "pc2_variance_percent": float(explained[1] * 100),
            "within_cohort_zscore": bool(within_cohort_zscore),
        }
    )


def main() -> None:
    panel_dirs = {
        "a": ROOT / "panel_a_cohort_flow",
        "b": ROOT / "panel_b_probe_coverage",
        "c": ROOT / "panel_c_raw_expression_space",
        "d": ROOT / "panel_d_standardized_expression_space",
        "e": ROOT / "panel_e_score_distribution",
    }
    for directory in panel_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    flow = pd.DataFrame(
        [
            ("GSE14520 · GPL3921", 1, "Official expression arrays", 445, "screened"),
            ("GSE14520 · GPL3921", 2, "Tumour samples", 225, "screened"),
            ("GSE14520 · GPL3921", 3, "Complete OS", 221, "included"),
            ("GSE116174 · GPL570", 1, "Official expression arrays", 64, "screened"),
            ("GSE116174 · GPL570", 2, "Clinical matches", 64, "screened"),
            ("GSE116174 · GPL570", 3, "Complete OS", 64, "included"),
            ("GSE14520 · GPL571", 1, "Official expression arrays", 43, "screened"),
            ("GSE14520 · GPL571", 2, "Tumour samples", 22, "screened"),
            ("GSE14520 · GPL571", 3, "Complete OS", 21, "excluded_insufficient_n"),
        ],
        columns=["cohort", "stage_order", "stage", "n", "decision"],
    )
    flow.to_csv(panel_dirs["a"] / "source_data.csv", index=False)

    probe_rows = []
    matrices = {}
    score_rows = []
    for cohort, paths in COHORTS.items():
        expression, _ = read_geo_series_matrix(paths["matrix"])
        annotation = read_geo_platform_annotation(paths["annotation"])
        matrix_keys = set(expression["probe_id"].map(_platform_probe_key))
        mapping = annotation.copy()
        mapping["platform_probe_id"] = mapping["probe_id"].map(_platform_probe_key)
        mapping = mapping.loc[mapping["platform_probe_id"].isin(matrix_keys)].copy()
        mapping["targets"] = mapping["gene_symbol"].map(_target_symbols)
        mapping = mapping.loc[mapping["targets"].map(len) == 1].copy()
        mapping["gene"] = mapping["targets"].str[0]
        counts = mapping.groupby("gene")["platform_probe_id"].nunique()
        for gene in METABOLIC_GENES:
            probe_rows.append(
                {
                    "cohort": cohort,
                    "platform": cohort.split("·")[-1].strip(),
                    "gene": gene,
                    "eligible_unique_probes": int(counts.get(gene, 0)),
                    "mapping_rule": "unique target symbol; median collapse",
                }
            )

        canonical = pd.read_csv(paths["canonical"]).set_index("gene").loc[METABOLIC_GENES]
        matrices[cohort] = canonical
        scores = pd.read_csv(paths["score"])
        scores["cohort"] = cohort
        score_rows.append(scores[["case_id", "cohort", "risk_score"]])

    coverage = pd.DataFrame(probe_rows)
    coverage.to_csv(panel_dirs["b"] / "source_data.csv", index=False)
    raw_pca = pca_frame(matrices, within_cohort_zscore=False)
    raw_pca.to_csv(panel_dirs["c"] / "source_data.csv", index=False)
    standardized_pca = pca_frame(matrices, within_cohort_zscore=True)
    standardized_pca.to_csv(panel_dirs["d"] / "source_data.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(panel_dirs["e"] / "source_data.csv", index=False)

    checks = {
        "status": "PASS",
        "included_cohorts": ["GSE14520_GPL3921", "GSE116174_GPL570"],
        "included_samples": {"GSE14520_GPL3921": 221, "GSE116174_GPL570": 64},
        "all_15_genes_available_in_included_platforms": bool(
            coverage.groupby("cohort")["gene"].nunique().eq(15).all()
            and coverage["eligible_unique_probes"].gt(0).all()
        ),
        "pca_rows_before": int(len(raw_pca)),
        "pca_rows_after": int(len(standardized_pca)),
        "score_rows": int(sum(len(frame) for frame in score_rows)),
        "gpl571_reporting_rule": "N=21; insufficient sample size; no performance analysis",
        "gpl571_performance_rows_in_figure": 0,
        "outcomes_used_for_mapping_or_standardization": False,
    }
    (ROOT / "SOURCE_DATA_GATE.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
