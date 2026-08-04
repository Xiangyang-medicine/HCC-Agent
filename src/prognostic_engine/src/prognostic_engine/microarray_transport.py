"""Outcome-blind microarray preparation and TCGA-only gene transport model.

This module implements the restricted secondary analysis in Phase 3B protocol
amendment v3.  It is deliberately separate from ``external_validation.py``:
the frozen M4 RNA-seq model must never receive microarray values.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from .config import INNER_SEED, M2_M3_ALPHA_RANGE, M2_M3_L1_RATIO_RANGE, METABOLIC_GENES
from .inner_splits import generate_inner_splits


MODEL_ID = "M2T_15gene_crossplatform_v1"
ARTIFACT_STATUS = "EXPLORATORY_EXTERNAL_TRANSPORT_ONLY_NOT_FOR_CLINICAL_DEPLOYMENT"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without retaining its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tab_fields(line: str) -> list[str]:
    return next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))


def read_geo_series_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read one GEO matrix and sample metadata; no clinical/outcome file is read."""
    import gzip

    metadata: dict[str, list[str]] = {}
    start = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.startswith("!Sample_"):
                fields = _tab_fields(line)
                metadata[fields[0].removeprefix("!Sample_")] = fields[1:]
            if line.startswith("!series_matrix_table_begin"):
                start = number + 1
                break
    if start is None:
        raise ValueError(f"GEO matrix-table marker not found in {path}")
    expression = pd.read_csv(path, sep="\t", compression="gzip", skiprows=start)
    expression = expression.loc[~expression.iloc[:, 0].astype(str).str.startswith("!")].copy()
    expression = expression.rename(columns={expression.columns[0]: "probe_id"})
    if expression.empty or expression.shape[1] < 2:
        raise ValueError(f"No expression matrix found in {path}")
    return expression, metadata


def read_geo_platform_annotation(path: Path) -> pd.DataFrame:
    """Read the official GEO platform annotation file."""
    import gzip

    start = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.startswith("!platform_table_begin"):
                start = number + 1
                break
    if start is None:
        raise ValueError(f"GEO platform-table marker not found in {path}")
    table = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=start,
        low_memory=False,
    )
    table = table.loc[~table.iloc[:, 0].astype(str).str.startswith("!")].copy()
    required = {"ID", "Gene symbol"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Platform annotation missing columns: {sorted(missing)}")
    return table.loc[:, ["ID", "Gene symbol"]].rename(columns={"ID": "probe_id", "Gene symbol": "gene_symbol"})


def _target_symbols(value: object) -> list[str]:
    """Return target genes in a GEO symbol field; multi-target probes are excluded."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text in {"---", "NA", "nan"}:
        return []
    tokens = [part.strip().upper() for part in text.replace("///", ";").replace(",", ";").split(";")]
    return sorted(set(token for token in tokens if token in METABOLIC_GENES))


def _platform_probe_key(value: object) -> str:
    """Normalise the GEO matrix's PM/MM probe decoration to its platform key.

    GSE116174's official matrix uses IDs such as ``1007_PM_s_at`` whereas
    GPL570 lists the corresponding platform feature as ``1007_s_at``.  This
    is an identifier-format conversion only, not a feature-selection step.
    """
    return str(value).replace("_PM_", "_").replace("_MM_", "_")


def collapse_probes_to_genes(expression: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Median-collapse uniquely mapped probes to the 15 prespecified symbols.

    The expression frame must contain probe IDs as the first ``probe_id`` column
    and GSM IDs as remaining columns.  No clinical or outcome data participate.
    """
    if expression.columns[0] != "probe_id":
        raise ValueError("Expression input must start with a probe_id column.")
    matrix = expression.copy()
    mapping = annotation.copy()
    matrix["platform_probe_id"] = matrix["probe_id"].map(_platform_probe_key)
    mapping["platform_probe_id"] = mapping["probe_id"].map(_platform_probe_key)
    merged = matrix.merge(mapping[["platform_probe_id", "gene_symbol"]], on="platform_probe_id", how="inner", validate="many_to_many")
    merged["targets"] = merged["gene_symbol"].map(_target_symbols)
    merged = merged.loc[merged["targets"].map(len) == 1].copy()
    merged["gene"] = merged["targets"].str[0]
    value_columns = list(expression.columns[1:])
    numeric = merged.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Official GEO matrix contains non-numeric retained expression values.")
    collapsed = numeric.assign(gene=merged["gene"].to_numpy()).groupby("gene", sort=False).median()
    missing = [gene for gene in METABOLIC_GENES if gene not in collapsed.index]
    if missing:
        raise ValueError(f"FEATURE_INCOMPATIBLE: required genes unavailable after mapping: {missing}")
    return collapsed.loc[METABOLIC_GENES, value_columns]


def external_cohort_zscore(gene_by_sample: pd.DataFrame) -> pd.DataFrame:
    """Outcome-blind z-scoring within one eligible cohort-platform stratum."""
    if list(gene_by_sample.index) != list(METABOLIC_GENES):
        raise ValueError("Gene rows must exactly follow the prespecified 15-gene order.")
    values = gene_by_sample.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite gene value before cross-platform standardisation.")
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=1)
    zero_variance = std.index[std <= 1e-12].tolist()
    if zero_variance:
        raise ValueError(f"FEATURE_INCOMPATIBLE: zero-variance genes: {zero_variance}")
    result = values.sub(mean, axis=0).div(std, axis=0)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Non-finite value after outcome-blind standardisation.")
    return result


def external_survival_metrics(times: np.ndarray, events: np.ndarray, risk: np.ndarray, tau: float) -> dict[str, float]:
    """Compute external-cohort C indices with a locked, reported IPCW horizon."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    if len(times) != len(events) or len(times) != len(risk) or len(times) < 2:
        raise ValueError("External metric arrays must be same-length and contain at least two cases.")
    if not (np.isfinite(times).all() and np.isfinite(risk).all()) or (times <= 0).any():
        raise ValueError("External survival metric input contains invalid time or risk values.")
    outcome = np.array(list(zip(events, times)), dtype=[("event", bool), ("time", float)])
    harrell = float(concordance_index_censored(events, times, risk)[0])
    uno = float(concordance_index_ipcw(outcome, outcome, risk, tau=float(tau))[0])
    return {"harrell_c": harrell, "uno_c": uno}


def bootstrap_external_cindices(
    times: np.ndarray, events: np.ndarray, risk: np.ndarray, tau: float, n_bootstrap: int = 1000, seed: int = 456
) -> dict:
    """Patient bootstrap for one external cohort; reports invalid iterations explicitly."""
    point = external_survival_metrics(times, events, risk, tau)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"harrell_c": [], "uno_c": []}
    invalid = 0
    n = len(times)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n, size=n)
        try:
            metrics = external_survival_metrics(np.asarray(times)[sampled], np.asarray(events)[sampled], np.asarray(risk)[sampled], tau)
            for name in values:
                values[name].append(metrics[name])
        except (ValueError, ArithmeticError):
            invalid += 1
    intervals = {}
    for name, result in values.items():
        if result:
            intervals[name] = {"point_estimate": point[name], "ci95": [float(np.percentile(result, 2.5)), float(np.percentile(result, 97.5))]}
        else:
            intervals[name] = {"point_estimate": point[name], "ci95": None}
    return {
        "tau_months": float(tau),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "valid_iterations": int(len(values["harrell_c"])),
        "invalid_iterations": int(invalid),
        "metrics": intervals,
    }


def _structured_outcome(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [(bool(event), float(time)) for event, time in zip(frame["event"], frame["survival_months"])],
        dtype=[("event", bool), ("time", float)],
    )


def _select_m2t_params(derivation: pd.DataFrame) -> tuple[dict[str, float], list[dict]]:
    """TCGA-only inner-CV hyperparameter selection with fold-local scaling."""
    genes = [f"{gene}_log2tpm" for gene in METABOLIC_GENES]
    splits = generate_inner_splits(derivation["case_id"].astype(str).tolist(), repeat=0, outer_fold=0, seed=INNER_SEED)
    best_score, best = -np.inf, None
    results = []
    for alpha in M2_M3_ALPHA_RANGE:
        for l1_ratio in M2_M3_L1_RATIO_RANGE:
            scores = []
            for fold in splits["folds"]:
                train = derivation.iloc[fold["train_indices"]]
                validation = derivation.iloc[fold["val_indices"]]
                mean = train[genes].mean()
                std = train[genes].std().replace(0, np.nan)
                if std.isna().any():
                    raise ValueError("Zero-variance TCGA gene in inner training fold.")
                x_train = ((train[genes] - mean) / std).to_numpy()
                x_val = ((validation[genes] - mean) / std).to_numpy()
                try:
                    model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio, max_iter=100000)
                    model.fit(x_train, _structured_outcome(train))
                    if np.abs(model.coef_).sum() < 1e-8:
                        continue
                    score = concordance_index_censored(
                        validation["event"].astype(bool), validation["survival_months"], model.predict(x_val)
                    )[0]
                    scores.append(float(score))
                except (ArithmeticError, ValueError):
                    continue
            result = {"alpha": float(alpha), "l1_ratio": float(l1_ratio), "fold_scores": scores,
                      "mean_harrell_c": float(np.mean(scores)) if scores else None}
            results.append(result)
            if scores and float(np.mean(scores)) > best_score:
                best_score = float(np.mean(scores))
                best = {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}
    if best is None:
        raise RuntimeError("No non-degenerate M2T hyperparameter configuration was selected on TCGA.")
    return best, results


@dataclass
class FrozenM2TArtifact:
    """TCGA-fitted coefficient model accepting only outcome-blind z-scored genes."""

    model: CoxnetSurvivalAnalysis
    selected_params: dict[str, float]
    model_id: str
    artifact_status: str
    derivation_manifest: dict

    def predict(self, z_scored_gene_by_sample: pd.DataFrame) -> pd.DataFrame:
        if list(z_scored_gene_by_sample.index) != list(METABOLIC_GENES):
            raise ValueError("External genes are absent or in an invalid order.")
        matrix = z_scored_gene_by_sample.loc[METABOLIC_GENES].T.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("External M2T matrix contains NaN or Inf.")
        risk = self.model.predict(matrix)
        if not np.isfinite(risk).all():
            raise ValueError("M2T generated a non-finite risk score.")
        return pd.DataFrame({"case_id": z_scored_gene_by_sample.columns.astype(str), "risk_score": risk})


def fit_frozen_m2t_artifact(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Derive and freeze M2T from TCGA only; external files are never read here."""
    data_path, output_dir = Path(data_path), Path(output_dir)
    derivation = pd.read_parquet(data_path)
    genes = [f"{gene}_log2tpm" for gene in METABOLIC_GENES]
    required = {"case_id", "survival_months", "event", *genes}
    missing = sorted(required - set(derivation.columns))
    if missing:
        raise ValueError(f"Derivation dataset missing columns: {missing}")
    if len(derivation) != 363 or int(derivation["event"].sum()) != 129:
        raise ValueError("Derivation data do not match the locked 363-case, 129-event TCGA cohort.")
    params, tuning = _select_m2t_params(derivation)
    mean, std = derivation[genes].mean(), derivation[genes].std().replace(0, np.nan)
    if std.isna().any():
        raise ValueError("Zero-variance TCGA gene in final M2T derivation data.")
    model = CoxnetSurvivalAnalysis(alphas=[params["alpha"]], l1_ratio=params["l1_ratio"], max_iter=100000)
    model.fit(((derivation[genes] - mean) / std).to_numpy(), _structured_outcome(derivation))
    if np.abs(model.coef_).sum() < 1e-8:
        raise RuntimeError("Final M2T model has no non-zero coefficient.")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    manifest = {
        "model_id": MODEL_ID,
        "artifact_status": ARTIFACT_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "derivation_data_sha256": sha256_file(data_path),
        "derivation_n": 363,
        "derivation_events": 129,
        "source_code_sha256": sha256_file(source_path),
        "genes": METABOLIC_GENES,
        "tcga_standardisation": "gene-wise mean and sample SD fit on TCGA derivation only",
        "external_standardisation": "outcome-blind gene-wise mean and sample SD within each cohort-platform stratum",
        "selected_params": params,
        "tcga_inner_cv": {"n_folds": 5, "seed": INNER_SEED, "preprocessing": "fit on inner training rows only"},
        "tuning_results": tuning,
        "external_outcomes_used_for_fitting": False,
    }
    artifact = FrozenM2TArtifact(model, params, MODEL_ID, ARTIFACT_STATUS, manifest)
    artifact_path = output_dir / "m2t_crossplatform_artifact.joblib"
    manifest_path = output_dir / "M2T_CROSSPLATFORM_MANIFEST.json"
    joblib.dump(artifact, artifact_path)
    manifest["artifact_sha256"] = sha256_file(artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact_path, manifest_path
