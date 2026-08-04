"""
Tests for the HCC Prognosis Assessment System.
"""

import pytest
from src.state.schema import PatientData, MetabolicFeatures, RiskLevel


class TestPatientData:
    """Test PatientData model."""

    def test_create_patient(self):
        """Test creating a patient data object."""
        patient = PatientData(
            patient_id="TEST-001",
            age=62,
            gender="M",
            stage="T2N0M0",
            grade="G2",
            bclc_stage="A"
        )

        assert patient.patient_id == "TEST-001"
        assert patient.age == 62
        assert patient.gender == "M"
        assert patient.stage == "T2N0M0"

    def test_patient_with_gene_expression(self):
        """Test patient with gene expression data."""
        patient = PatientData(
            patient_id="TEST-002",
            gene_expression={
                "CA9": 3.5,
                "VEGFA": 4.2,
                "HK2": 4.0
            }
        )

        assert patient.gene_expression is not None
        assert len(patient.gene_expression) == 3
        assert patient.gene_expression["CA9"] == 3.5


class TestMetabolicFeatures:
    """Test MetabolicFeatures model."""

    def test_create_features(self):
        """Test creating metabolic features."""
        features = MetabolicFeatures(
            predicted_subtype="Proliferation",
            subtype_confidence=0.85,
            pathway_activities={"hsa00010": 1.5, "hsa00020": 0.8}
        )

        assert features.predicted_subtype == "Proliferation"
        assert features.subtype_confidence == 0.85
        assert "hsa00010" in features.pathway_activities


class TestRiskAssessment:
    """Test risk assessment functionality."""
    from src.state.schema import RiskAssessment

    def test_risk_levels(self):
        """Test all risk levels are valid."""
        for level in RiskLevel:
            assessment = RiskAssessment(risk_level=level)
            assert assessment.risk_level == level

    def test_confidence_bounds(self):
        """Test confidence score is within bounds."""
        assessment = RiskAssessment(
            confidence_score=0.75
        )
        assert 0 <= assessment.confidence_score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
