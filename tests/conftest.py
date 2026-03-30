"""Shared fixtures for the weightloss-rag-pubmed test suite."""
import os

# Set required env vars before any app imports
os.environ.setdefault("NCBI_API_KEY", "test-key")
os.environ.setdefault("NCBI_EMAIL", "test@example.com")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_article_data():
    """Raw article JSON as returned by the ingestion pipeline."""
    return {
        "pmid": "12345678",
        "abstract_layer": {
            "pmid": "12345678",
            "article_title": "Semaglutide for Weight Loss",
            "journal": "NEJM",
            "pub_year": 2023,
            "first_author_lastname": "Smith",
            "first_author_initials": "JA",
            "mesh_major_terms": ["Obesity", "GLP-1"],
            "mesh_minor_terms": [],
            "publication_types": ["Randomized Controlled Trial"],
            "is_human": True,
            "is_animal": False,
            "content": "This randomized controlled trial evaluated semaglutide 2.4mg for weight management.",
        },
        "body_layer": {
            "pmid": "12345678",
            "article_title": "Semaglutide for Weight Loss",
            "journal": "NEJM",
            "pub_year": 2023,
            "first_author_lastname": "Smith",
            "first_author_initials": "JA",
            "mesh_major_terms": ["Obesity", "GLP-1"],
            "mesh_minor_terms": [],
            "publication_types": ["Randomized Controlled Trial"],
            "is_human": True,
            "is_animal": False,
            "content": (
                "## Methods\n"
                "### Study Design\n"
                "A double-blind, placebo-controlled trial was conducted across 16 countries.\n\n"
                "### Participants\n"
                "Adults with BMI >= 30 were enrolled.\n\n"
                "## Results\n"
                "### Primary Outcome\n"
                "Mean weight change was -14.9% with semaglutide vs -2.4% with placebo.\n\n"
                "### Secondary Outcomes\n"
                "Waist circumference decreased significantly in the treatment group.\n\n"
                "## Discussion\n"
                "These findings demonstrate clinically meaningful weight loss with semaglutide."
            ),
        },
    }


@pytest.fixture
def sample_documents():
    """A list of Documents simulating retrieval pipeline output."""
    return [
        Document(
            page_content="Semaglutide reduced body weight by 14.9% compared to placebo.",
            metadata={
                "pmid": "11111111",
                "pub_year": 2023,
                "first_author_lastname": "Smith",
                "first_author_initials": "JA",
                "article_title": "Semaglutide for Weight Loss",
                "journal": "NEJM",
                "publication_types": ["Randomized Controlled Trial"],
                "Header 2": "Results",
                "section": "body",
                "is_human": True,
                "is_animal": False,
            },
        ),
        Document(
            page_content="Tirzepatide showed dose-dependent reductions in body weight.",
            metadata={
                "pmid": "22222222",
                "pub_year": 2022,
                "first_author_lastname": "Jones",
                "first_author_initials": "RB",
                "article_title": "Tirzepatide for Obesity",
                "journal": "Lancet",
                "publication_types": ["Meta-Analysis"],
                "Header 2": "Conclusions",
                "section": "body",
                "is_human": True,
                "is_animal": False,
            },
        ),
        Document(
            page_content="GLP-1 receptor agonists are a class of incretin mimetics.",
            metadata={
                "pmid": "33333333",
                "pub_year": 2021,
                "first_author_lastname": "Lee",
                "first_author_initials": "CK",
                "article_title": "GLP-1 Mechanism Review",
                "journal": "Nature Reviews",
                "publication_types": ["Review"],
                "Header 2": "Introduction",
                "section": "body",
                "is_human": True,
                "is_animal": False,
            },
        ),
    ]
