import numpy as np
import pandas as pd
import pytest

from prognostic_engine.config import METABOLIC_GENES
from prognostic_engine.microarray_transport import bootstrap_external_cindices, collapse_probes_to_genes, external_cohort_zscore


def _expression_and_annotation():
    expression = pd.DataFrame({"probe_id": ["p1", "p2", "p3"], "gsm1": [1.0, 3.0, 10.0], "gsm2": [3.0, 5.0, 20.0]})
    annotation = pd.DataFrame({"probe_id": ["p1", "p2", "p3"], "gene_symbol": ["HK2", "HK2", "PKM /// LDHA"]})
    return expression, annotation


def test_probe_collapse_median_and_multitarget_exclusion():
    expression, annotation = _expression_and_annotation()
    with pytest.raises(ValueError, match="FEATURE_INCOMPATIBLE"):
        collapse_probes_to_genes(expression, annotation)
    rows = []
    for index, gene in enumerate(METABOLIC_GENES):
        rows.append({"probe_id": f"p{index}", "gsm1": float(index), "gsm2": float(index + 1)})
    full_expression = pd.DataFrame(rows)
    full_annotation = pd.DataFrame({"probe_id": full_expression["probe_id"], "gene_symbol": METABOLIC_GENES})
    collapsed = collapse_probes_to_genes(full_expression, full_annotation)
    assert list(collapsed.index) == METABOLIC_GENES
    assert collapsed.loc["HK2", "gsm1"] == 0.0


def test_probe_id_pm_decoration_matches_platform_annotation_key():
    rows = []
    for index, gene in enumerate(METABOLIC_GENES):
        rows.append({"probe_id": f"p{index}_PM_at", "gsm1": float(index), "gsm2": float(index + 1)})
    expression = pd.DataFrame(rows)
    annotation = pd.DataFrame({"probe_id": [f"p{index}_at" for index in range(15)], "gene_symbol": METABOLIC_GENES})
    assert list(collapse_probes_to_genes(expression, annotation).index) == METABOLIC_GENES


def test_outcome_blind_standardisation_has_expected_shape_and_no_nan():
    values = pd.DataFrame({"a": np.arange(15, dtype=float), "b": np.arange(1, 16, dtype=float)}, index=METABOLIC_GENES)
    z = external_cohort_zscore(values)
    assert z.shape == values.shape
    assert np.isfinite(z.to_numpy()).all()
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0)


def test_standardisation_rejects_zero_variance_gene():
    values = pd.DataFrame({"a": np.arange(15, dtype=float), "b": np.arange(1, 16, dtype=float)}, index=METABOLIC_GENES)
    values.loc["HK2"] = 1.0
    with pytest.raises(ValueError, match="zero-variance"):
        external_cohort_zscore(values)


def test_external_bootstrap_keeps_known_risk_direction_and_reports_iterations():
    times = np.arange(1, 31, dtype=float)
    events = np.ones(30, dtype=bool)
    risk = -times
    result = bootstrap_external_cindices(times, events, risk, tau=29.0, n_bootstrap=25, seed=11)
    assert result["metrics"]["harrell_c"]["point_estimate"] == pytest.approx(1.0)
    assert result["valid_iterations"] + result["invalid_iterations"] == 25
