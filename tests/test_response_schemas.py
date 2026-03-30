"""Tests for app/models/response_schemas.py Pydantic models."""
from app.models.response_schemas import (
    Claim, ClaimType, StructuredAnswer,
    SourceDocument, SupportLevel, VerifiedClaim,
)
try:
    from app.models.response_schemas import EvidenceTableRow, EvidenceTable
    _evidence_table_available = True
except ImportError:
    _evidence_table_available = False


class TestSourceDocument:
    def test_serialization(self):
        doc = SourceDocument(
            pmid="12345678",
            title="Test Article",
            journal="NEJM",
            year=2023,
            first_author="Smith",
            publication_type="RCT",
            rerank_score=1.5,
            section="body",
            snippet="Some text...",
        )
        data = doc.model_dump()
        assert data["pmid"] == "12345678"
        assert data["rerank_score"] == 1.5

    def test_defaults(self):
        doc = SourceDocument(pmid="99999999")
        assert doc.title == ""
        assert doc.year is None
        assert doc.rerank_score is None


class TestVerifiedClaim:
    def test_inherits_claim_fields(self):
        vc = VerifiedClaim(
            statement="Test claim",
            claim_type=ClaimType.EVIDENCE,
            pmids=["111"],
            support_level=SupportLevel.SUPPORTED,
            verification_reasoning="Directly stated.",
        )
        assert vc.statement == "Test claim"
        assert vc.claim_type == ClaimType.EVIDENCE
        assert vc.support_level == SupportLevel.SUPPORTED

    def test_support_level_values(self):
        assert SupportLevel.SUPPORTED == "supported"
        assert SupportLevel.PARTIAL == "partial"
        assert SupportLevel.UNSUPPORTED == "unsupported"
        assert SupportLevel.CONTRADICTORY == "contradictory"


import pytest

@pytest.mark.skipif(not _evidence_table_available, reason="EvidenceTableRow/EvidenceTable not yet defined")
class TestEvidenceTable:
    def test_row_serialization(self):
        row = EvidenceTableRow(
            pmid="12345678",
            study="Smith (2023)",
            population="Adults with BMI >= 30",
            intervention="Semaglutide 2.4mg",
            comparator="Placebo",
            outcome="14.9% mean weight loss",
            study_type="RCT",
        )
        data = row.model_dump()
        assert data["pmid"] == "12345678"
        assert data["study"] == "Smith (2023)"
        assert data["study_type"] == "RCT"

    def test_row_defaults(self):
        row = EvidenceTableRow(pmid="99999999", study="Unknown (2020)")
        assert row.population == ""
        assert row.intervention == ""
        assert row.comparator == ""
        assert row.outcome == ""
        assert row.study_type == ""

    def test_evidence_table_with_rows(self):
        table = EvidenceTable(rows=[
            EvidenceTableRow(pmid="111", study="A (2023)"),
            EvidenceTableRow(pmid="222", study="B (2022)"),
        ])
        assert len(table.rows) == 2

    def test_evidence_table_empty(self):
        table = EvidenceTable()
        assert table.rows == []
