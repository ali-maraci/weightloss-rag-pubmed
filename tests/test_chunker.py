"""Tests for app/core/chunker.py"""
import pytest
from app.core.chunker import Chunker


@pytest.fixture
def chunker():
    return Chunker()


@pytest.fixture
def result(chunker, sample_article_data):
    return chunker.process_article(sample_article_data)


# ---------------------------------------------------------------------------
# Index A (abstract)
# ---------------------------------------------------------------------------

def test_index_a_produces_exactly_one_document(result):
    assert len(result["index_a"]) == 1


def test_index_a_document_has_section_abstract(result):
    doc = result["index_a"][0]
    assert doc.metadata["section"] == "abstract"


def test_index_a_document_preserves_article_metadata(result):
    doc = result["index_a"][0]
    assert doc.metadata["pmid"] == "12345678"
    assert doc.metadata["pub_year"] == 2023
    assert doc.metadata["first_author_lastname"] == "Smith"
    assert doc.metadata["first_author_initials"] == "JA"
    assert doc.metadata["journal"] == "NEJM"
    assert doc.metadata["article_title"] == "Semaglutide for Weight Loss"
    assert doc.metadata["is_human"] is True
    assert doc.metadata["is_animal"] is False


def test_index_a_document_content_matches_abstract(result, sample_article_data):
    expected = sample_article_data["abstract_layer"]["content"]
    assert result["index_a"][0].page_content == expected


# ---------------------------------------------------------------------------
# Index B (body)
# ---------------------------------------------------------------------------

def test_index_b_produces_multiple_chunks(result):
    assert len(result["index_b"]) > 1


def test_index_b_chunks_have_section_body(result):
    for doc in result["index_b"]:
        assert doc.metadata["section"] == "body"


def test_index_b_chunks_have_header_2_metadata(result):
    """At least some chunks should carry a Header 2 key from markdown splitting."""
    headers = [doc.metadata.get("Header 2") for doc in result["index_b"]]
    assert any(h is not None for h in headers)


def test_index_b_chunks_have_header_3_metadata(result):
    """At least some chunks should carry a Header 3 key from markdown splitting."""
    headers = [doc.metadata.get("Header 3") for doc in result["index_b"]]
    assert any(h is not None for h in headers)


def test_index_b_chunks_preserve_article_metadata(result):
    for doc in result["index_b"]:
        assert doc.metadata["pmid"] == "12345678"
        assert doc.metadata["pub_year"] == 2023
        assert doc.metadata["first_author_lastname"] == "Smith"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_abstract_layer_produces_empty_index_a(chunker, sample_article_data):
    data = {**sample_article_data, "abstract_layer": {}}
    result = chunker.process_article(data)
    assert result["index_a"] == []


def test_empty_body_layer_produces_empty_index_b(chunker, sample_article_data):
    data = {**sample_article_data, "body_layer": {}}
    result = chunker.process_article(data)
    assert result["index_b"] == []
