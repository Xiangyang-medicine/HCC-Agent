"""Frozen M4 derivation and scoring artifact for Phase 3B.

This module is intentionally separate from nested-CV reporting.  It fits one
derivation model on the verified TCGA cohort only after selecting the RSF
configuration through inner cross-validation on that same cohort.  External
data are never read during fitting or tuning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from .config import EVALUATION_TIMES, INNER_SEED, METABOLIC_GENES
from .inner_splits import generate_inner_splits


M4_CONFIGS = (
    {"n_estimators": 50, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 3},
)
MODEL_ID = "M4_combined_rsf_phase3b_frozen_v1"
ARTIFACT_STATUS = "EXTERNAL_VALIDATION_ONLY_NOT_FOR_CLINICAL_DEPLOYMENT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_series(series: pd.Series) -> pd.Series:
    """Match Phase 3A preprocessing, where missing categories become 'nan'."""
    return series.fillna("nan").astype(str)


@dataclass
class M4FeatureTransformer:
    """Fit TCGA-only transformations and reject incompatible external inputs."""

    age_mean: float | None = None
    age_std: float | None = None
    stage_categories: tuple[str, ...] = ()
    grade_categories: tuple[str, ...] = ()
    gene_columns: tuple[str, ...] = ()
    gene_mean: dict[str, float] | None = None
    gene_std: dict[str, float] | None = None

    @property
    def clinical_feature_names(self) -> list[str]:
        return [
            "age_z",
            *[f"stage_{value}" for value in self.stage_categories],
            *[f"grade_{value}" for value in self.grade_categories],
            "gender_Undocumented",
        ]

    @property
    def feature_names(self) -> list[str]:
        return self.clinical_feature_names + list(self.gene_columns)

    def fit(self, derivation_df: pd.DataFrame) -> "M4FeatureTransformer":
        self.gene_columns = tuple(f"{gene}_log2tpm" for gene in METABOLIC_GENES)
        self.age_mean = float(derivation_df["age_at_diagnosis"].mean())
        self.age_std = float(derivation_df["age_at_diagnosis"].std())
        if not np.isfinite(self.age_std) or self.age_std < 1e-8:
            self.age_std = 1.0
        self.stage_categories = tuple(sorted(_category_series(derivation_df["ajcc_stage"]).unique()))
        self.grade_categories = tuple(sorted(_category_series(derivation_df["tumor_grade"]).unique()))
        means = derivation_df.loc[:, self.gene_columns].mean()
        stds = derivation_df.loc[:, self.gene_columns].std().replace(0, 1.0).fillna(1.0)
        self.gene_mean = {column: float(means[column]) for column in self.gene_columns}
        self.gene_std = {column: float(stds[column]) for column in self.gene_columns}
        return self

    def _assert_fitted(self) -> None:
        if self.age_mean is None or self.gene_mean is None or self.gene_std is None:
            raise RuntimeError("Feature transformer has not been fitted.")

    def validate_external_frame(self, frame: pd.DataFrame) -> None:
        """Reject missing or unmapped external features before scoring."""
        self._assert_fitted()
        required = {"age_at_diagnosis", "ajcc_stage", "tumor_grade", *self.gene_columns}
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(f"External frame missing required columns: {missing_columns}")
        null_columns = [column for column in required if frame[column].isna().any()]
        if null_columns:
            raise ValueError(f"External frame contains missing required values: {sorted(null_columns)}")
        unknown_stages = set(_category_series(frame["ajcc_stage"])) - set(self.stage_categories)
        unknown_grades = set(_category_series(frame["tumor_grade"])) - set(self.grade_categories)
        if unknown_stages:
            raise ValueError(f"External frame contains unmapped AJCC stages: {sorted(unknown_stages)}")
        if unknown_grades:
            raise ValueError(f"External frame contains unmapped tumor grades: {sorted(unknown_grades)}")

    def transform_derivation(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform TCGA derivation rows, retaining its explicit missing categories."""
        return self._transform(frame, allow_training_missing_categories=True)

    def transform_external(self, frame: pd.DataFrame) -> np.ndarray:
        self.validate_external_frame(frame)
        return self._transform(frame, allow_training_missing_categories=False)

    def _transform(self, frame: pd.DataFrame, allow_training_missing_categories: bool) -> np.ndarray:
        self._assert_fitted()
        age = (frame["age_at_diagnosis"].astype(float).to_numpy() - self.age_mean) / self.age_std
        stage = _category_series(frame["ajcc_stage"])
        grade = _category_series(frame["tumor_grade"])
        stage_array = np.column_stack([(stage == value).to_numpy(dtype=float) for value in self.stage_categories])
        grade_array = np.column_stack([(grade == value).to_numpy(dtype=float) for value in self.grade_categories])
        gender_placeholder = np.zeros((len(frame), 1), dtype=float)
        genes = frame.loc[:, self.gene_columns].astype(float).to_numpy()
        gene_mean = np.asarray([self.gene_mean[column] for column in self.gene_columns])
        gene_std = np.asarray([self.gene_std[column] for column in self.gene_columns])
        genes = (genes - gene_mean) / gene_std
        matrix = np.hstack([age.reshape(-1, 1), stage_array, grade_array, gender_placeholder, genes])
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing produced non-finite values.")
        return matrix


@dataclass
class FrozenM4ExternalArtifact:
    model: RandomSurvivalForest
    transformer: M4FeatureTransformer
    selected_params: dict
    model_id: str
    artifact_status: str
    derivation_manifest: dict

    def predict(self, external_frame: pd.DataFrame, times: tuple[int, ...] = tuple(EVALUATION_TIMES)) -> pd.DataFrame:
        """Score only an already harmonized, feature-complete external cohort."""
        features = self.transformer.transform_external(external_frame)
        risk = self.model.predict(features)
        functions = self.model.predict_survival_function(features)
        probabilities = np.array([[function(time) for time in times] for function in functions])
        output = pd.DataFrame({"risk_score": risk})
        for index, time in enumerate(times):
            output[f"survival_probability_{time}m"] = probabilities[:, index]
        if "case_id" in external_frame.columns:
            output.insert(0, "case_id", external_frame["case_id"].astype(str).to_numpy())
        return output


def _structured_outcome(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [(bool(event), float(time)) for event, time in zip(frame["event"], frame["survival_months"])],
        dtype=[("event", bool), ("time", float)],
    )


def _select_m4_params(derivation_df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Select M4 configuration with TCGA-only inner CV and fold-local preprocessing."""
    case_ids = derivation_df["case_id"].astype(str).tolist()
    splits = generate_inner_splits(case_ids, repeat=0, outer_fold=0, seed=INNER_SEED)
    scores_by_config: list[dict] = []
    best_params = None
    best_score = -np.inf

    for config in M4_CONFIGS:
        fold_scores = []
        for fold in splits["folds"]:
            train = derivation_df.loc[derivation_df["case_id"].isin(fold["train_case_ids"])].copy()
            validation = derivation_df.loc[derivation_df["case_id"].isin(fold["val_case_ids"])].copy()
            transformer = M4FeatureTransformer().fit(train)
            model = RandomSurvivalForest(**config, random_state=42, n_jobs=-1)
            model.fit(transformer.transform_derivation(train), _structured_outcome(train))
            # Validation is still TCGA derivation data, so preserve the Phase 3A
            # missing-category convention rather than applying external input gates.
            prediction = model.predict(transformer.transform_derivation(validation))
            outcome = _structured_outcome(validation)
            score = concordance_index_censored(outcome["event"], outcome["time"], prediction)[0]
            fold_scores.append(float(score))
        mean_score = float(np.mean(fold_scores))
        result = {"params": config, "fold_scores": fold_scores, "mean_harrell_c": mean_score}
        scores_by_config.append(result)
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(config)
    if best_params is None:
        raise RuntimeError("No M4 configuration was selected.")
    return best_params, scores_by_config


def fit_frozen_m4_external_artifact(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Fit and persist the external-validation-only M4 artifact and manifest."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    derivation = pd.read_parquet(data_path)
    required = {"case_id", "survival_months", "event", "age_at_diagnosis", "ajcc_stage", "tumor_grade"}
    required.update(f"{gene}_log2tpm" for gene in METABOLIC_GENES)
    missing = sorted(required - set(derivation.columns))
    if missing:
        raise ValueError(f"Derivation dataset missing required columns: {missing}")
    if len(derivation) != 363 or int(derivation["event"].sum()) != 129:
        raise ValueError("Derivation cohort does not match the locked 363-patient, 129-event Phase 3A cohort.")

    params, tuning_results = _select_m4_params(derivation)
    transformer = M4FeatureTransformer().fit(derivation)
    final_model = RandomSurvivalForest(**params, random_state=42, n_jobs=-1)
    final_model.fit(transformer.transform_derivation(derivation), _structured_outcome(derivation))

    source_path = Path(__file__).resolve()
    manifest = {
        "model_id": MODEL_ID,
        "artifact_status": ARTIFACT_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "derivation_data_path": str(data_path),
        "derivation_data_sha256": sha256_file(data_path),
        "derivation_n": int(len(derivation)),
        "derivation_events": int(derivation["event"].sum()),
        "source_code_sha256": sha256_file(source_path),
        "inner_cv": {"n_folds": 5, "seed": INNER_SEED, "preprocessing": "fit on each inner training fold only"},
        "candidate_configs": list(M4_CONFIGS),
        "tuning_results": tuning_results,
        "selected_params": params,
        "feature_names": transformer.feature_names,
        "feature_unit": "log2(TPM + 1) for all gene columns",
        "external_input_policy": {
            "missing_values": "REJECT",
            "unknown_stage_or_grade": "REJECT",
            "microarray_input": "REJECT_UNTIL_A_SEPARATE_PRESPECIFIED_TRANSFORMATION_IS_VALIDATED",
            "external_outcomes_used_for_fitting": False,
        },
    }
    artifact = FrozenM4ExternalArtifact(
        model=final_model,
        transformer=transformer,
        selected_params=params,
        model_id=MODEL_ID,
        artifact_status=ARTIFACT_STATUS,
        derivation_manifest=manifest,
    )
    artifact_path = output_dir / "m4_external_validation_artifact.joblib"
    manifest_path = output_dir / "M4_EXTERNAL_VALIDATION_MANIFEST.json"
    joblib.dump(artifact, artifact_path)
    manifest["artifact_sha256"] = sha256_file(artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact_path, manifest_path
