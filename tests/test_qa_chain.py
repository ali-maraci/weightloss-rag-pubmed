"""Tests for QAChain._format_docs and QAChain._standardize_citations."""
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from app.core.qa_chain import QAChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain():
    """Instantiate QAChain with all external dependencies mocked."""
    with patch("app.core.qa_chain.Retriever"), patch("app.core.qa_chain.ChatOpenAI"):
        return QAChain()


# ---------------------------------------------------------------------------
# _format_docs tests
# ---------------------------------------------------------------------------

class TestFormatDocs:

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_groups_chunks_by_pmid(self, mock_retriever, mock_chat_openai):
        """Two docs with the same PMID appear under a single ARTICLE header."""
        chain = QAChain()
        docs = [
            Document(
                page_content="Chunk A",
                metadata={"pmid": "11111111", "pub_year": 2023, "first_author_lastname": "Smith"},
            ),
            Document(
                page_content="Chunk B",
                metadata={"pmid": "11111111", "pub_year": 2023, "first_author_lastname": "Smith"},
            ),
        ]
        result = chain._format_docs(docs)
        assert result.count("ARTICLE 1") == 1
        assert "ARTICLE 2" not in result
        assert "Chunk A" in result
        assert "Chunk B" in result

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_preserves_relevance_ranking_order(self, mock_retriever, mock_chat_openai):
        """The first-seen PMID gets ARTICLE 1 regardless of PMID string sort order."""
        chain = QAChain()
        docs = [
            Document(
                page_content="First hit",
                metadata={"pmid": "99999999", "pub_year": 2020, "first_author_lastname": "Zara"},
            ),
            Document(
                page_content="Second hit",
                metadata={"pmid": "11111111", "pub_year": 2021, "first_author_lastname": "Alpha"},
            ),
        ]
        result = chain._format_docs(docs)
        idx_first = result.index("ARTICLE 1")
        idx_second = result.index("ARTICLE 2")
        assert idx_first < idx_second
        assert "99999999" in result[: result.index("ARTICLE 2")]
        assert "11111111" in result[result.index("ARTICLE 2"):]

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_citation_header_with_author_and_year(self, mock_retriever, mock_chat_openai):
        """Full header: --- ARTICLE 1: (Smith, 2023) [PMID: 11111111] ---"""
        chain = QAChain()
        docs = [
            Document(
                page_content="Some content.",
                metadata={"pmid": "11111111", "pub_year": 2023, "first_author_lastname": "Smith"},
            )
        ]
        result = chain._format_docs(docs)
        assert "--- ARTICLE 1: (Smith, 2023) [PMID: 11111111] ---" in result

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_citation_header_missing_author_uses_year_only(self, mock_retriever, mock_chat_openai):
        """When author is absent the header falls back to year-only format."""
        chain = QAChain()
        docs = [
            Document(
                page_content="Some content.",
                metadata={"pmid": "11111111", "pub_year": 2023},
            )
        ]
        result = chain._format_docs(docs)
        assert "--- ARTICLE 1: (2023) [PMID: 11111111] ---" in result

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_citation_header_missing_year_uses_author_only(self, mock_retriever, mock_chat_openai):
        """When year is absent the header falls back to author-only format."""
        chain = QAChain()
        docs = [
            Document(
                page_content="Some content.",
                metadata={"pmid": "11111111", "first_author_lastname": "Smith"},
            )
        ]
        result = chain._format_docs(docs)
        assert "--- ARTICLE 1: (Smith) [PMID: 11111111] ---" in result

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_citation_header_missing_both_uses_pmid_only(self, mock_retriever, mock_chat_openai):
        """When both author and year are absent the header is PMID-only."""
        chain = QAChain()
        docs = [
            Document(
                page_content="Some content.",
                metadata={"pmid": "11111111"},
            )
        ]
        result = chain._format_docs(docs)
        assert "--- ARTICLE 1: [PMID: 11111111] ---" in result

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_multi_chunk_joined_with_ellipsis_separator(self, mock_retriever, mock_chat_openai):
        """Multiple chunks for the same PMID are joined with '\\n...\\n'."""
        chain = QAChain()
        docs = [
            Document(
                page_content="Chunk A",
                metadata={"pmid": "11111111", "pub_year": 2023, "first_author_lastname": "Smith"},
            ),
            Document(
                page_content="Chunk B",
                metadata={"pmid": "11111111", "pub_year": 2023, "first_author_lastname": "Smith"},
            ),
        ]
        result = chain._format_docs(docs)
        assert "Chunk A\n...\nChunk B" in result


# ---------------------------------------------------------------------------
# _standardize_citations tests
# ---------------------------------------------------------------------------

class TestStandardizeCitations:

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_combined_author_pmid_bracket(self, mock_retriever, mock_chat_openai):
        """[Smith, 2023; PMID: 12345678] -> (Smith, 2023) [PMID: 12345678]"""
        chain = QAChain()
        result = chain._standardize_citations("[Smith, 2023; PMID: 12345678]")
        assert result == "(Smith, 2023) [PMID: 12345678]"

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_multiple_pmids_exploded(self, mock_retriever, mock_chat_openai):
        """[PMID: 123, 456, 789] -> [PMID: 123] [PMID: 456] [PMID: 789]"""
        chain = QAChain()
        result = chain._standardize_citations("[PMID: 123, 456, 789]")
        assert result == "[PMID: 123] [PMID: 456] [PMID: 789]"

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_single_pmid_passes_through(self, mock_retriever, mock_chat_openai):
        """A single-PMID bracket is returned unchanged."""
        chain = QAChain()
        result = chain._standardize_citations("[PMID: 12345678]")
        assert result == "[PMID: 12345678]"

    @patch("app.core.qa_chain.ChatOpenAI")
    @patch("app.core.qa_chain.Retriever")
    def test_normal_text_passes_through(self, mock_retriever, mock_chat_openai):
        """Text with no citation patterns is returned unchanged."""
        chain = QAChain()
        text = "GLP-1 receptor agonists reduce body weight."
        assert chain._standardize_citations(text) == text


# ---------------------------------------------------------------------------
# _docs_to_source_list tests
# ---------------------------------------------------------------------------

class TestDocsToSourceList:
    """Tests for QAChain._docs_to_source_list()"""

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_deduplicates_by_pmid(self, mock_llm, mock_retriever):
        """Multiple chunks from same PMID produce one SourceDocument."""
        chain = QAChain()
        docs = [
            Document(page_content="chunk 1", metadata={"pmid": "111", "article_title": "Title", "pub_year": 2023, "first_author_lastname": "Smith", "journal": "NEJM", "publication_types": ["RCT"], "rerank_score": 0.9, "section": "body"}),
            Document(page_content="chunk 2", metadata={"pmid": "111", "article_title": "Title", "pub_year": 2023, "first_author_lastname": "Smith", "journal": "NEJM", "publication_types": ["RCT"], "rerank_score": 0.8, "section": "body"}),
        ]
        result = chain._docs_to_source_list(docs)
        assert len(result) == 1
        assert result[0]["pmid"] == "111"

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_maps_metadata_fields(self, mock_llm, mock_retriever):
        """Metadata fields are correctly mapped to SourceDocument keys."""
        chain = QAChain()
        docs = [
            Document(page_content="Some content here for snippet", metadata={
                "pmid": "222",
                "article_title": "My Article",
                "pub_year": 2024,
                "first_author_lastname": "Jones",
                "journal": "Lancet",
                "publication_types": ["Meta-Analysis", "Review"],
                "rerank_score": 1.5,
                "section": "body",
            }),
        ]
        result = chain._docs_to_source_list(docs)
        assert len(result) == 1
        doc = result[0]
        assert doc["title"] == "My Article"
        assert doc["year"] == 2024
        assert doc["first_author"] == "Jones"
        assert doc["journal"] == "Lancet"
        assert doc["publication_type"] == "Meta-Analysis"
        assert doc["rerank_score"] == 1.5
        assert doc["snippet"].startswith("Some content")

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_sorted_by_rerank_score_descending(self, mock_llm, mock_retriever):
        """Source docs are sorted by rerank_score descending."""
        chain = QAChain()
        docs = [
            Document(page_content="low", metadata={"pmid": "A", "rerank_score": 0.5}),
            Document(page_content="high", metadata={"pmid": "B", "rerank_score": 2.0}),
            Document(page_content="mid", metadata={"pmid": "C", "rerank_score": 1.0}),
        ]
        result = chain._docs_to_source_list(docs)
        scores = [d.get("rerank_score") for d in result]
        assert scores == [2.0, 1.0, 0.5]

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_snippet_truncated_to_200_chars(self, mock_llm, mock_retriever):
        """Snippet is truncated to ~200 characters."""
        chain = QAChain()
        docs = [
            Document(page_content="x" * 500, metadata={"pmid": "333", "rerank_score": 1.0}),
        ]
        result = chain._docs_to_source_list(docs)
        assert len(result[0]["snippet"]) <= 200


# ---------------------------------------------------------------------------
# _apply_filters tests
# ---------------------------------------------------------------------------

class TestApplyFilters:
    """Tests for QAChain._apply_filters()"""

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_no_filters_returns_original_query(self, mock_llm, mock_retriever):
        chain = QAChain()
        result = chain._apply_filters("What is semaglutide?", None)
        assert result == "What is semaglutide?"

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_empty_filters_returns_original_query(self, mock_llm, mock_retriever):
        chain = QAChain()
        result = chain._apply_filters("What is semaglutide?", {})
        assert result == "What is semaglutide?"

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_year_range_filter(self, mock_llm, mock_retriever):
        chain = QAChain()
        result = chain._apply_filters("query", {"year_min": 2020, "year_max": 2024})
        assert "2020" in result
        assert "2024" in result

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_publication_types_filter(self, mock_llm, mock_retriever):
        chain = QAChain()
        result = chain._apply_filters("query", {"publication_types": ["RCT", "Meta-Analysis"]})
        assert "RCT" in result
        assert "Meta-Analysis" in result

    @patch("app.core.qa_chain.Retriever")
    @patch("app.core.qa_chain.ChatOpenAI")
    def test_human_only_filter(self, mock_llm, mock_retriever):
        chain = QAChain()
        result = chain._apply_filters("query", {"human_only": True})
        assert "human" in result.lower()
