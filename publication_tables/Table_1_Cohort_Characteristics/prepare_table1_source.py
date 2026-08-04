from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source_data"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

ACM = Path(r"F:\ACM")
TCGA_MODEL = ACM / "data/modeling/tcga_lihc_modeling_dataset.parquet"
TCGA_CBIO = ACM / "data/external/hcc_tcga_gdc_patient_clinical_data.json"
GSE14520_PRED = (
    ACM
    / "experiments/phase3b/microarray_transport/"
    "GSE14520_GPL3921_m2t_evaluation_predictions.csv"
)
GSE14520_CLIN = ACM / "GSE14520/GSE14520_Extra_Supplement.txt"
GSE116174_PRED = (
    ACM
    / "experiments/phase3b/microarray_transport/"
    "GSE116174_GPL570_m2t_evaluation_predictions.csv"
)
GSE116174_CLIN = ACM / "GSE116174/GSE116174_HCC-64-u133_plus_2_clinical_data.xls"
FIGURE3_FLOW = (
    ACM
    / "publication_figures/Figure_3_External_Transport/"
    "panel_b_cohort_flow/source_data.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_group(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    text = text.replace("STAGE", "").strip()
    text = text.replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV")
    if text.startswith("IV") or text.startswith("III"):
        return "III-IV"
    if text.startswith("II"):
        return "II"
    if text.startswith("I"):
        return "I"
    return "Missing"


def count_rate(series: pd.Series, value: object) -> tuple[int, float]:
    count = int((series == value).sum())
    return count, count / len(series)


def summary_record(
    cohort: str,
    platform: str,
    role: str,
    expression_technology: str,
    data: pd.DataFrame,
    *,
    age_col: str | None,
    sex_col: str | None,
    stage_col: str | None,
    grade_col: str | None,
) -> dict[str, object]:
    n = int(len(data))
    events = int(pd.to_numeric(data["event"], errors="coerce").sum())
    observed = pd.to_numeric(data["survival_months"], errors="coerce")
    stage = (
        data[stage_col].map(stage_group)
        if stage_col is not None
        else pd.Series(["Missing"] * n)
    )

    result: dict[str, object] = {
        "cohort": cohort,
        "platform": platform,
        "role": role,
        "expression_technology": expression_technology,
        "patients_n": n,
        "events_n": events,
        "event_rate": events / n,
        "observed_time_median_months": float(observed.median()),
        "observed_time_q1_months": float(observed.quantile(0.25)),
        "observed_time_q3_months": float(observed.quantile(0.75)),
        "genes_available_n": 15,
        "genes_required_n": 15,
    }

    if age_col is not None:
        age = pd.to_numeric(data[age_col], errors="coerce")
        result.update(
            {
                "age_available_n": int(age.notna().sum()),
                "age_median_years": float(age.median()),
                "age_q1_years": float(age.quantile(0.25)),
                "age_q3_years": float(age.quantile(0.75)),
            }
        )
    else:
        result.update(
            {
                "age_available_n": 0,
                "age_median_years": np.nan,
                "age_q1_years": np.nan,
                "age_q3_years": np.nan,
            }
        )

    if sex_col is not None:
        sex = data[sex_col].astype("string").str.strip().str.lower()
        male_n = int(sex.isin(["m", "male"]).sum())
        result.update(
            {
                "sex_available_n": int(sex.notna().sum()),
                "male_n": male_n,
                "male_rate": male_n / n,
            }
        )
    else:
        result.update(
            {
                "sex_available_n": 0,
                "male_n": np.nan,
                "male_rate": np.nan,
            }
        )

    for group, suffix in [("I", "stage_i"), ("II", "stage_ii"), ("III-IV", "stage_iii_iv"), ("Missing", "stage_missing")]:
        count, rate = count_rate(stage, group)
        result[f"{suffix}_n"] = count
        result[f"{suffix}_rate"] = rate

    if grade_col is not None:
        grade = data[grade_col].astype("string").str.upper()
        low = grade.isin(["G1", "G2"])
        high = grade.isin(["G3", "G4"])
        missing = ~(low | high)
        result.update(
            {
                "grade_g1_g2_n": int(low.sum()),
                "grade_g1_g2_rate": float(low.mean()),
                "grade_g3_g4_n": int(high.sum()),
                "grade_g3_g4_rate": float(high.mean()),
                "grade_missing_n": int(missing.sum()),
                "grade_missing_rate": float(missing.mean()),
            }
        )
    else:
        result.update(
            {
                "grade_g1_g2_n": np.nan,
                "grade_g1_g2_rate": np.nan,
                "grade_g3_g4_n": np.nan,
                "grade_g3_g4_rate": np.nan,
                "grade_missing_n": np.nan,
                "grade_missing_rate": np.nan,
            }
        )
    return result


def fmt_n_pct(count: object, rate: object) -> str:
    if pd.isna(count) or pd.isna(rate):
        return "NA"
    return f"{int(count)} ({float(rate) * 100:.1f}%)"


def fmt_median_iqr(median: object, q1: object, q3: object) -> str:
    if pd.isna(median) or pd.isna(q1) or pd.isna(q3):
        return "NA"
    return f"{float(median):.1f} ({float(q1):.1f}–{float(q3):.1f})"


def main() -> None:
    tcga = pd.read_parquet(TCGA_MODEL)
    cbio_records = json.loads(TCGA_CBIO.read_text(encoding="utf-8"))
    cbio_sex = pd.DataFrame(
        [
            {
                "submitter_id": item["patientId"],
                "sex": item["value"],
            }
            for item in cbio_records
            if item.get("clinicalAttributeId") == "SEX"
        ]
    ).drop_duplicates("submitter_id")
    tcga = tcga.merge(cbio_sex, on="submitter_id", how="left", validate="one_to_one")
    if tcga["sex"].notna().sum() != len(tcga):
        raise ValueError("TCGA sex matching is incomplete.")

    gse145_pred = pd.read_csv(GSE14520_PRED)
    gse145_clin = pd.read_csv(GSE14520_CLIN, sep="\t")
    gse145 = gse145_pred.merge(
        gse145_clin[["Affy_GSM", "Gender", "Age", "TNM staging"]],
        left_on="case_id",
        right_on="Affy_GSM",
        how="left",
        validate="one_to_one",
    )
    if gse145["Age"].notna().sum() != len(gse145):
        raise ValueError("GSE14520 clinical matching is incomplete.")

    gse116_pred = pd.read_csv(GSE116174_PRED)
    gse116_clin = pd.read_excel(GSE116174_CLIN, sheet_name="mRNA")
    gse116 = gse116_pred.merge(
        gse116_clin[["sampleID", "Gender", "Age", "clinstage"]],
        left_on="source_sample_id",
        right_on="sampleID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if gse116["Gender"].notna().sum() != len(gse116):
        raise ValueError("GSE116174 clinical matching is incomplete.")

    records = [
        summary_record(
            "TCGA-LIHC",
            "GDC RNA-seq",
            "Development and repeated nested cross-validation",
            "Illumina RNA sequencing; log2(TPM + 1)",
            tcga,
            age_col="age_at_diagnosis",
            sex_col="sex",
            stage_col="ajcc_stage",
            grade_col="tumor_grade",
        ),
        summary_record(
            "GSE14520",
            "GPL3921",
            "Secondary exploratory cross-platform transport",
            "Affymetrix microarray",
            gse145,
            age_col="Age",
            sex_col="Gender",
            stage_col="TNM staging",
            grade_col=None,
        ),
        summary_record(
            "GSE116174",
            "GPL570",
            "Secondary exploratory cross-platform transport",
            "Affymetrix Human Genome U133 Plus 2.0 microarray",
            gse116,
            age_col="Age",
            sex_col="Gender",
            stage_col="clinstage",
            grade_col=None,
        ),
    ]
    summary = pd.DataFrame(records)

    flow = pd.read_csv(FIGURE3_FLOW)
    expected = {
        "GSE14520": (221, 85),
        "GSE116174": (64, 27),
    }
    for cohort, (n, events) in expected.items():
        row = summary.loc[summary["cohort"] == cohort].iloc[0]
        flow_row = flow.loc[flow["cohort"] == cohort].iloc[0]
        if int(row["patients_n"]) != n or int(row["events_n"]) != events:
            raise ValueError(f"{cohort} does not match locked Figure 3 counts.")
        if int(flow_row["n"]) != n or int(flow_row["events"]) != events:
            raise ValueError(f"{cohort} source-flow mismatch.")

    summary.to_csv(SOURCE_DIR / "cohort_characteristics_numeric.csv", index=False)
    numeric_records = json.loads(summary.to_json(orient="records"))

    columns = ["Characteristic", "TCGA-LIHC", "GSE14520 (GPL3921)", "GSE116174 (GPL570)"]
    table_rows = [
        ["Analysis role", *summary["role"].tolist()],
        ["Expression technology", *summary["expression_technology"].tolist()],
        ["Patients, n", *summary["patients_n"].astype(int).astype(str).tolist()],
        ["Deaths, n (%)", *[fmt_n_pct(r.events_n, r.event_rate) for r in summary.itertuples()]],
        [
            "Observed OS/censoring time, median (IQR), months",
            *[
                fmt_median_iqr(
                    r.observed_time_median_months,
                    r.observed_time_q1_months,
                    r.observed_time_q3_months,
                )
                for r in summary.itertuples()
            ],
        ],
        [
            "Age, median (IQR), years",
            *[
                fmt_median_iqr(r.age_median_years, r.age_q1_years, r.age_q3_years)
                for r in summary.itertuples()
            ],
        ],
        ["Male sex, n (%)", *[fmt_n_pct(r.male_n, r.male_rate) for r in summary.itertuples()]],
        ["AJCC/TNM stage, n (%)", "", "", ""],
        ["  I", *[fmt_n_pct(r.stage_i_n, r.stage_i_rate) for r in summary.itertuples()]],
        ["  II", *[fmt_n_pct(r.stage_ii_n, r.stage_ii_rate) for r in summary.itertuples()]],
        ["  III–IV", *[fmt_n_pct(r.stage_iii_iv_n, r.stage_iii_iv_rate) for r in summary.itertuples()]],
        ["  Missing", *[fmt_n_pct(r.stage_missing_n, r.stage_missing_rate) for r in summary.itertuples()]],
        ["Tumour grade, n (%)", "", "", ""],
        ["  G1–G2", *[fmt_n_pct(r.grade_g1_g2_n, r.grade_g1_g2_rate) for r in summary.itertuples()]],
        ["  G3–G4", *[fmt_n_pct(r.grade_g3_g4_n, r.grade_g3_g4_rate) for r in summary.itertuples()]],
        ["  Missing", *[fmt_n_pct(r.grade_missing_n, r.grade_missing_rate) for r in summary.itertuples()]],
        ["Locked metabolic genes available", "15/15", "15/15", "15/15"],
    ]
    pd.DataFrame(table_rows, columns=columns).to_csv(
        ROOT / "Table_1_Cohort_Characteristics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded = pd.DataFrame(
        [
            {
                "cohort": "GSE14520",
                "platform": "GPL571",
                "complete_os_cases": 21,
                "events": 11,
                "analysis_status": "NOT_ANALYSED",
                "reason": "Sample size too small for the prespecified analysis.",
            }
        ]
    )
    excluded.to_csv(SOURCE_DIR / "excluded_cohort_record.csv", index=False)

    provenance_rows = []
    source_urls = {
        TCGA_MODEL: "https://portal.gdc.cancer.gov/projects/TCGA-LIHC",
        TCGA_CBIO: "https://www.cbioportal.org/study/summary?id=hcc_tcga_gdc",
        GSE14520_PRED: "Derived locked Figure 3 evaluation file",
        GSE14520_CLIN: "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/suppl/GSE14520_Extra_Supplement.txt.gz",
        GSE116174_PRED: "Derived locked Figure 3 evaluation file",
        GSE116174_CLIN: "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116174/suppl/GSE116174_HCC-64-u133_plus_2_clinical_data.xls.gz",
        FIGURE3_FLOW: "Locked Figure 3 source data",
    }
    for path, url in source_urls.items():
        provenance_rows.append(
            {
                "input_file": str(path),
                "source_url_or_role": url,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(provenance_rows).to_csv(
        SOURCE_DIR / "input_provenance.csv",
        index=False,
    )
    payload = {
        "cohorts": numeric_records,
        "excluded_cohorts": json.loads(excluded.to_json(orient="records")),
        "provenance": provenance_rows,
    }
    (SOURCE_DIR / "table1_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    gate = {
        "status": "TABLE1_SOURCE_DATA_GENERATED",
        "success": True,
        "cohorts_analysed": 3,
        "tcga_patients": int(summary.loc[summary["cohort"] == "TCGA-LIHC", "patients_n"].iloc[0]),
        "gse14520_gpl3921_patients": 221,
        "gse116174_gpl570_patients": 64,
        "gpl571_performance_analysed": False,
        "gpl571_exclusion_reason": "Sample size too small (N=21).",
        "all_external_counts_match_figure3": True,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in [
                SOURCE_DIR / "cohort_characteristics_numeric.csv",
                SOURCE_DIR / "excluded_cohort_record.csv",
                SOURCE_DIR / "input_provenance.csv",
                SOURCE_DIR / "table1_payload.json",
                ROOT / "Table_1_Cohort_Characteristics.csv",
            ]
        },
    }
    (ROOT / "TABLE1_SOURCE_GATE.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
