"""
TCGA data loader for HCC (LIHC) dataset.

This module provides utilities for loading and processing TCGA-LIHC data,
including gene expression, clinical information, and survival outcomes.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import os

from src.state.schema import PatientData


class TCGALoader:
    """
    Loader for TCGA-LIHC (Liver Hepatocellular Carcinoma) dataset.

    This class handles:
    - Loading preprocessed TCGA data
    - Filtering by patient characteristics
    - Converting to PatientData format
    """

    def __init__(self, data_dir: str = "F:/ACM/data"):
        """
        Initialize TCGA loader.

        Args:
            data_dir: Directory containing TCGA data files
        """
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "tcga_lihc_data.parquet"

        # Metadata for TCGA-LIHC
        self.known_metabolic_genes = [
            # Glycolysis
            "HK2", "PKM2", "LDHA", "LDHB", "GPI", "PGAM1", "ENO1", "ENO2",
            # Glutamine metabolism
            "GLS", "GLS2", "GLUD1", "GLUD2",
            # Lipid metabolism
            "FASN", "SCD", "ACC1", "ACACA", "HMGCR", "LDLR",
            # TCA cycle
            "IDH1", "IDH2", "MDH1", "MDH2", "SDHA", "SDHB", "FH",
            # Oxidative stress
            "NFE2L2", "KEAP1", "GCLC", "GCLM", "TXN", "PRDX1",
            # Hypoxia response
            "CA9", "VEGFA", "PDGFA", "PDGFB", "HIF1A", "EPAS1",
            # Wnt/beta-catenin
            "CTNNB1", "AXIN1", "AXIN2", "APC", "GSK3B",
            # Other metabolism
            "GPC3", "ARG1", "OTC", "ASS1", "PSAT1", "PHGDH",
        ]

    def load_data(self) -> pd.DataFrame:
        """
        Load TCGA-LIHC data from file.

        Returns:
            DataFrame with patient data
        """
        if self.data_file.exists():
            return pd.read_parquet(self.data_file)
        else:
            # Return mock data for development/testing
            print(f"Warning: {self.data_file} not found. Using mock data.")
            return self._create_mock_data()

    def _create_mock_data(self, n_patients: int = 50) -> pd.DataFrame:
        """
        Create mock TCGA data for development.

        Args:
            n_patients: Number of mock patients to generate

        Returns:
            DataFrame with mock patient data
        """
        np.random.seed(42)

        data = {
            "patient_id": [f"TCGA-{chr(65+i//26)}{chr(65+i%26)}-{np.random.randint(1000,9999)}"
                          for i in range(n_patients)],
            "age": np.random.randint(30, 85, n_patients),
            "gender": np.random.choice(["M", "F"], n_patients, p=[0.7, 0.3]),
            "stage": np.random.choice(["I", "II", "III", "IV"], n_patients, p=[0.35, 0.25, 0.30, 0.10]),
            "grade": np.random.choice(["G1", "G2", "G3", "G4"], n_patients, p=[0.15, 0.40, 0.35, 0.10]),
            "bclc_stage": np.random.choice(["0", "A", "B", "C", "D"], n_patients, p=[0.05, 0.40, 0.25, 0.25, 0.05]),
            "afp_level": np.random.exponential(500, n_patients),
            "albumin": np.random.normal(4.0, 0.5, n_patients),
            "bilirubin": np.random.exponential(1.5, n_patients),
            "survival_months": np.random.exponential(30, n_patients),
            "vital_status": np.random.choice(["Alive", "Dead"], n_patients, p=[0.5, 0.5]),
        }

        # Add gene expression for known metabolic genes
        for gene in self.known_metabolic_genes[:10]:  # Use subset for mock
            data[gene] = np.random.normal(5, 1, n_patients)

        df = pd.DataFrame(data)
        return df

    def get_patient(self, patient_id: str) -> Optional[PatientData]:
        """
        Get a single patient by ID.

        Args:
            patient_id: TCGA patient barcode

        Returns:
            PatientData object or None if not found
        """
        df = self.load_data()

        patient_row = df[df["patient_id"] == patient_id]
        if patient_row.empty:
            return None

        row = patient_row.iloc[0]

        # Extract gene expression
        gene_cols = [col for col in df.columns if col in self.known_metabolic_genes]
        gene_expression = {gene: float(row[gene]) for gene in gene_cols if pd.notna(row[gene])}

        return PatientData(
            patient_id=str(row["patient_id"]),
            age=int(row["age"]) if pd.notna(row["age"]) else None,
            gender=str(row["gender"]) if pd.notna(row["gender"]) else None,
            stage=str(row["stage"]) if pd.notna(row["stage"]) else None,
            grade=str(row["grade"]) if pd.notna(row["grade"]) else None,
            bclc_stage=str(row["bclc_stage"]) if pd.notna(row["bclc_stage"]) else None,
            afp_level=float(row["afp_level"]) if pd.notna(row["afp_level"]) else None,
            albumin=float(row["albumin"]) if pd.notna(row["albumin"]) else None,
            bilirubin=float(row["bilirubin"]) if pd.notna(row["bilirubin"]) else None,
            survival_months=float(row["survival_months"]) if pd.notna(row["survival_months"]) else None,
            vital_status=str(row["vital_status"]) if pd.notna(row["vital_status"]) else None,
            gene_expression=gene_expression if gene_expression else None,
        )

    def get_cohort(self, n: Optional[int] = None) -> List[PatientData]:
        """
        Get a cohort of patients.

        Args:
            n: Number of patients to return (None for all)

        Returns:
            List of PatientData objects
        """
        df = self.load_data()
        if n:
            df = df.head(n)

        patients = []
        for _, row in df.iterrows():
            gene_cols = [col for col in df.columns if col in self.known_metabolic_genes]
            gene_expression = {gene: float(row[gene]) for gene in gene_cols if pd.notna(row[gene])}

            patient = PatientData(
                patient_id=str(row["patient_id"]),
                age=int(row["age"]) if pd.notna(row["age"]) else None,
                gender=str(row["gender"]) if pd.notna(row["gender"]) else None,
                stage=str(row["stage"]) if pd.notna(row["stage"]) else None,
                grade=str(row["grade"]) if pd.notna(row["grade"]) else None,
                bclc_stage=str(row["bclc_stage"]) if pd.notna(row["bclc_stage"]) else None,
                afp_level=float(row["afp_level"]) if pd.notna(row["afp_level"]) else None,
                albumin=float(row["albumin"]) if pd.notna(row["albumin"]) else None,
                bilirubin=float(row["bilirubin"]) if pd.notna(row["bilirubin"]) else None,
                survival_months=float(row["survival_months"]) if pd.notna(row["survival_months"]) else None,
                vital_status=str(row["vital_status"]) if pd.notna(row["vital_status"]) else None,
                gene_expression=gene_expression if gene_expression else None,
            )
            patients.append(patient)

        return patients

    def get_similar_patients(
        self,
        patient: PatientData,
        n: int = 5,
        features: Optional[List[str]] = None
    ) -> List[PatientData]:
        """
        Find similar patients based on clinical/molecular features.

        Args:
            patient: Reference patient
            n: Number of similar patients to return
            features: Features to use for similarity (default: key clinical)

        Returns:
            List of similar PatientData objects
        """
        df = self.load_data()
        target_id = patient.patient_id
        df = df[df["patient_id"] != target_id]

        if features is None:
            features = ["age", "stage", "grade", "bclc_stage", "afp_level"]

        # Simple similarity scoring
        scores = pd.Series(0.0, index=df.index)
        for feat in features:
            if feat in df.columns and getattr(patient, feat, None) is not None:
                ref_val = getattr(patient, feat)
                if isinstance(ref_val, (int, float)):
                    diff = (df[feat] - ref_val).abs()
                    scores += 1 / (1 + diff)
                elif isinstance(ref_val, str):
                    scores += (df[feat] == ref_val).astype(float)

        top_indices = scores.nlargest(n).index
        similar_df = df.loc[top_indices]

        patients = []
        for _, row in similar_df.iterrows():
            gene_cols = [col for col in df.columns if col in self.known_metabolic_genes]
            gene_expression = {gene: float(row[gene]) for gene in gene_cols if pd.notna(row[gene])}

            p = PatientData(
                patient_id=str(row["patient_id"]),
                age=int(row["age"]) if pd.notna(row["age"]) else None,
                gender=str(row["gender"]) if pd.notna(row["gender"]) else None,
                stage=str(row["stage"]) if pd.notna(row["stage"]) else None,
                grade=str(row["grade"]) if pd.notna(row["grade"]) else None,
                bclc_stage=str(row["bclc_stage"]) if pd.notna(row["bclc_stage"]) else None,
                afp_level=float(row["afp_level"]) if pd.notna(row["afp_level"]) else None,
                albumin=float(row["albumin"]) if pd.notna(row["albumin"]) else None,
                bilirubin=float(row["bilirubin"]) if pd.notna(row["bilirubin"]) else None,
                survival_months=float(row["survival_months"]) if pd.notna(row["survival_months"]) else None,
                vital_status=str(row["vital_status"]) if pd.notna(row["vital_status"]) else None,
                gene_expression=gene_expression if gene_expression else None,
            )
            patients.append(p)

        return patients


# Global loader instance
_tcga_loader: Optional[TCGALoader] = None


def get_tcga_loader() -> TCGALoader:
    """Get the global TCGA loader instance."""
    global _tcga_loader
    if _tcga_loader is None:
        _tcga_loader = TCGALoader()
    return _tcga_loader
