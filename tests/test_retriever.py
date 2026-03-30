"""Tests for Retriever pure/deterministic methods."""
import datetime
import math
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from app.models.schemas import ParsedQuery, MetadataFilters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_retriever():
    """Return a Retriever instance with VectorStore and QueryParser mocked out."""
    with patch("app.core.retriever.VectorStore"), patch("app.core.retriever.QueryParser"):
        from app.core.retriever import Retriever
        return Retriever()


def make_doc(pmid="99999999", pub_year=2022, publication_types=None, header2="", content="text"):
    """Convenience factory for a Document with the metadata _stage_3_rerank reads."""
    return Document(
        page_content=content,
        metadata={
            "pmid": pmid,
            "pub_year": pub_year,
            "publication_types": publication_types or [],
            "Header 2": header2,
        },
    )


# ---------------------------------------------------------------------------
# _stage_3_rerank
# ---------------------------------------------------------------------------

class TestStage3Rerank:
    """Tests for Retriever._stage_3_rerank."""

    def test_meta_analysis_boost(self):
        """Meta-Analysis publication type adds +1.00 to the base score."""
        retriever = make_retriever()
        doc = make_doc(publication_types=["Meta-Analysis"], header2="")
        parsed_query = ParsedQuery(optimized_query="obesity treatment")

        result = retriever._stage_3_rerank([(doc, 0.5)], parsed_query)

        _, final_score = result[0]
        # +1.00 boost + recency bonus (small, positive).  At minimum 1.50.
        assert final_score >= 1.50

    def test_rct_boost(self):
        """Randomized Controlled Trial publication type adds +0.85 to the base score."""
        retriever = make_retriever()
        doc = make_doc(publication_types=["Randomized Controlled Trial"], header2="")
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank([(doc, 0.5)], parsed_query)

        _, final_score = result[0]
        assert final_score >= 0.5 + 0.85

    def test_meta_analysis_boost_exceeds_rct(self):
        """Meta-Analysis score is higher than RCT for identical base scores and years."""
        retriever = make_retriever()
        doc_meta = make_doc(pmid="AAA", pub_year=2022, publication_types=["Meta-Analysis"])
        doc_rct = make_doc(pmid="BBB", pub_year=2022, publication_types=["Randomized Controlled Trial"])
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank([(doc_meta, 0.5), (doc_rct, 0.5)], parsed_query)

        scores = {doc.metadata["pmid"]: score for doc, score in result}
        assert scores["AAA"] > scores["BBB"]

    def test_results_section_boost(self):
        """Header 2 containing 'result' adds +1.5 to the score."""
        retriever = make_retriever()
        doc = make_doc(header2="Results", publication_types=["Review"])
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank([(doc, 0.0)], parsed_query)

        _, final_score = result[0]
        # 0.0 base + 0.55 (Review) + 1.5 (Results) + tiny recency
        assert final_score >= 0.55 + 1.5

    def test_introduction_section_penalty(self):
        """Header 2 containing 'introduction' subtracts 0.5 from the score."""
        retriever = make_retriever()
        doc_intro = make_doc(pmid="INTRO", header2="Introduction", publication_types=["Review"], pub_year=2022)
        doc_other = make_doc(pmid="OTHER", header2="Methods", publication_types=["Review"], pub_year=2022)
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank(
            [(doc_intro, 0.5), (doc_other, 0.5)], parsed_query
        )

        scores = {doc.metadata["pmid"]: score for doc, score in result}
        # Introduction penalised (-0.5) vs Methods (+1.0)
        assert scores["INTRO"] < scores["OTHER"]

    def test_output_sorted_descending(self):
        """Output list is sorted by final_score highest first."""
        retriever = make_retriever()
        docs = [
            (make_doc(pmid="A", publication_types=["Case Reports"], pub_year=2020, header2="Introduction"), 0.1),
            (make_doc(pmid="B", publication_types=["Meta-Analysis"], pub_year=2023, header2="Results"), 0.9),
            (make_doc(pmid="C", publication_types=["Review"], pub_year=2021, header2="Methods"), 0.5),
        ]
        parsed_query = ParsedQuery(optimized_query="obesity")

        result = retriever._stage_3_rerank(docs, parsed_query)

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_recency_boost_decreases_with_age(self):
        """A more recent article should receive a larger recency boost than an older one."""
        retriever = make_retriever()
        current_year = datetime.datetime.now().year

        doc_recent = make_doc(pmid="RECENT", pub_year=current_year - 1, publication_types=[], header2="")
        doc_old = make_doc(pmid="OLD", pub_year=current_year - 20, publication_types=[], header2="")
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank(
            [(doc_recent, 0.0), (doc_old, 0.0)], parsed_query
        )

        scores = {doc.metadata["pmid"]: score for doc, score in result}
        assert scores["RECENT"] > scores["OLD"]

    def test_recency_boost_is_positive(self):
        """Recency boost should be strictly positive for any article within the current year."""
        retriever = make_retriever()
        current_year = datetime.datetime.now().year
        doc = make_doc(pub_year=current_year, publication_types=[], header2="")
        parsed_query = ParsedQuery(optimized_query="weight loss")

        result = retriever._stage_3_rerank([(doc, 0.0)], parsed_query)

        _, final_score = result[0]
        expected_recency = 0.25 * math.exp(0)  # year_diff == 0
        # 0.0 base + 0.60 (no pub type match) + 0.25 recency
        assert final_score == pytest.approx(0.60 + expected_recency, abs=1e-6)


# ---------------------------------------------------------------------------
# _stage_4_diversity_filter
# ---------------------------------------------------------------------------

class TestStage4DiversityFilter:
    """Tests for Retriever._stage_4_diversity_filter."""

    def test_max_chunks_per_pmid(self):
        """No more than max_chunks_per_article (5) documents from the same PMID."""
        retriever = make_retriever()
        # 10 docs all from the same PMID
        docs = [
            (Document(page_content=f"chunk {i}", metadata={"pmid": "SAME"}), 1.0)
            for i in range(10)
        ]

        result = retriever._stage_4_diversity_filter(docs)

        same_pmid_count = sum(1 for d in result if d.metadata["pmid"] == "SAME")
        assert same_pmid_count == retriever.max_chunks_per_article

    def test_target_return_size_cap(self):
        """Total results never exceed target_return_size (30)."""
        retriever = make_retriever()
        # 100 docs, each from a unique PMID so diversity limit never kicks in
        docs = [
            (Document(page_content=f"doc {i}", metadata={"pmid": str(i)}), 1.0)
            for i in range(100)
        ]

        result = retriever._stage_4_diversity_filter(docs)

        assert len(result) == retriever.target_return_size

    def test_diversity_allows_multiple_pmids(self):
        """Documents from different PMIDs all pass through (up to target_return_size)."""
        retriever = make_retriever()
        docs = [
            (Document(page_content=f"doc {i}", metadata={"pmid": str(i)}), 1.0)
            for i in range(5)
        ]

        result = retriever._stage_4_diversity_filter(docs)

        assert len(result) == 5
        pmids = {d.metadata["pmid"] for d in result}
        assert len(pmids) == 5

    def test_mixed_pmids_enforces_per_pmid_limit(self):
        """Mix of PMIDs: only 5 chunks allowed per PMID, others unaffected."""
        retriever = make_retriever()
        # 8 from PMID "X", 3 from PMID "Y"
        docs = (
            [(Document(page_content=f"x{i}", metadata={"pmid": "X"}), 1.0) for i in range(8)]
            + [(Document(page_content=f"y{i}", metadata={"pmid": "Y"}), 0.9) for i in range(3)]
        )

        result = retriever._stage_4_diversity_filter(docs)

        x_count = sum(1 for d in result if d.metadata["pmid"] == "X")
        y_count = sum(1 for d in result if d.metadata["pmid"] == "Y")
        assert x_count == retriever.max_chunks_per_article
        assert y_count == 3


# ---------------------------------------------------------------------------
# _build_qdrant_filter
# ---------------------------------------------------------------------------

class TestBuildQdrantFilter:
    """Tests for Retriever._build_qdrant_filter."""

    def test_returns_none_when_no_metadata_filters(self):
        """Returns None when parsed_query.metadata_filters is None."""
        retriever = make_retriever()
        parsed_query = ParsedQuery(optimized_query="weight loss", metadata_filters=None)

        result = retriever._build_qdrant_filter(parsed_query)

        assert result is None

    def test_returns_none_when_all_filter_fields_empty(self):
        """Returns None when MetadataFilters has no populated fields."""
        retriever = make_retriever()
        parsed_query = ParsedQuery(
            optimized_query="weight loss",
            metadata_filters=MetadataFilters(),
        )

        result = retriever._build_qdrant_filter(parsed_query)

        assert result is None

    def test_creates_filter_for_publication_year(self):
        """Creates a Qdrant Filter with a FieldCondition for pub_year."""
        retriever = make_retriever()
        parsed_query = ParsedQuery(
            optimized_query="weight loss",
            metadata_filters=MetadataFilters(publication_year=2023),
        )

        result = retriever._build_qdrant_filter(parsed_query)

        assert result is not None
        # Inspect the must conditions
        must = result.must
        assert len(must) == 1
        condition = must[0]
        assert condition.key == "metadata.pub_year"
        assert condition.match.value == 2023

    def test_creates_filter_for_first_author_lastname(self):
        """Creates a Qdrant Filter with a FieldCondition for first_author_lastname."""
        retriever = make_retriever()
        parsed_query = ParsedQuery(
            optimized_query="smith obesity",
            metadata_filters=MetadataFilters(first_author_lastname="Smith"),
        )

        result = retriever._build_qdrant_filter(parsed_query)

        assert result is not None
        must = result.must
        assert len(must) == 1
        assert must[0].key == "metadata.first_author_lastname"
        assert must[0].match.value == "Smith"

    def test_creates_filter_combining_multiple_fields(self):
        """Multiple populated fields produce multiple must conditions."""
        retriever = make_retriever()
        parsed_query = ParsedQuery(
            optimized_query="query",
            metadata_filters=MetadataFilters(publication_year=2022, is_human=True),
        )

        result = retriever._build_qdrant_filter(parsed_query)

        assert result is not None
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert "metadata.pub_year" in keys
        assert "metadata.is_human" in keys
