from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prognostic_engine.config import METABOLIC_GENES
from prognostic_engine.external_validation import M4FeatureTransformer


def _frame(n=8):
    data = {
        "case_id": [f"case-{index}" for index in range(n)],
        "age_at_diagnosis": np.linspace(45, 65, n),
        "ajcc_stage": ["Stage I", "Stage II"] * (n // 2),
        "tumor_grade": ["G1", "G2"] * (n // 2),
    }
    for index, gene in enumerate(METABOLIC_GENES):
        data[f"{gene}_log2tpm"] = np.linspace(index, index + 1, n)
    return pd.DataFrame(data)


def test_transformer_rejects_missing_external_features():
    transformer = M4FeatureTransformer().fit(_frame())
    external = _frame().drop(columns=["HK2_log2tpm"])
    with pytest.raises(ValueError, match="missing required columns"):
        transformer.transform_external(external)


def test_transformer_rejects_external_missing_values_and_unknown_categories():
    transformer = M4FeatureTransformer().fit(_frame())
    missing = _frame()
    missing.loc[0, "tumor_grade"] = np.nan
    with pytest.raises(ValueError, match="missing required values"):
        transformer.transform_external(missing)
    unknown = _frame()
    unknown.loc[0, "ajcc_stage"] = "Stage IV"
    with pytest.raises(ValueError, match="unmapped AJCC stages"):
        transformer.transform_external(unknown)


def test_derivation_transform_preserves_training_missing_category_and_feature_order():
    training = _frame()
    training.loc[0, "ajcc_stage"] = np.nan
    transformer = M4FeatureTransformer().fit(training)
    transformed = transformer.transform_derivation(training)
    assert transformed.shape == (len(training), len(transformer.feature_names))
    assert np.isfinite(transformed).all()
    assert transformer.feature_names[-15:] == [f"{gene}_log2tpm" for gene in METABOLIC_GENES]
