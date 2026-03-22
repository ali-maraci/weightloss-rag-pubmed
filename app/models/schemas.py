from typing import List, Optional
from pydantic import BaseModel, Field

class ArticleMetadata(BaseModel):
    """Structured metadata extracted from a PubMed article's Medline XML."""
    pmid: str = Field(description="PubMed ID of the article.")
    doi: Optional[str] = Field(default=None, description="DOI of the article.")
    section: str = Field(description="The section this content belongs to, e.g., 'abstract' or 'body'.")
    article_title: str = Field(description="Title of the article.")
    journal: str = Field(description="Journal name.")
    pub_year: int = Field(description="Publication year.")
    first_author_lastname: str = Field(default="Unknown", description="Last name of the first author.")
    first_author_initials: str = Field(default="", description="Initials of the first author.")
    mesh_major_terms: List[str] = Field(default_factory=list, description="Major Medical Subject Headings (MeSH).")
    mesh_minor_terms: List[str] = Field(default_factory=list, description="Minor Medical Subject Headings (MeSH).")
    publication_types: List[str] = Field(default_factory=list, description="List of publication types.")
    is_human: bool = Field(default=False, description="Whether the study relates to humans.")
    is_animal: bool = Field(default=False, description="Whether the study relates to animals.")
    content: str = Field(description="The actual text content, either Abstract or Body.")

class MetadataFilters(BaseModel):
    """Explicitly defined extractable metadata filters for querying."""
    publication_year: Optional[int] = Field(
        default=None, 
        description="The specific 4-digit year. Do NOT guess or infer the year if the user says 'recent' or 'latest'."
    )
    first_author_lastname: Optional[str] = Field(
        default=None, 
        description="The last name of the author. If the user asks for papers by a specific name (e.g. 'Prof Shovlin' or 'author Smith'), extract the last name (e.g. 'Shovlin' or 'Smith')."
    )
    journal_name: Optional[str] = Field(default=None)
    mesh_major_terms: Optional[List[str]] = Field(default=None)
    is_human: Optional[bool] = Field(default=None)
    is_animal: Optional[bool] = Field(default=None)

class ParsedQuery(BaseModel):
    """The structured output format enforced by the LLM for optimized queries."""
    clarification_required: Optional[str] = Field(
        default=None,
        description="Populate only if the query is critically ambiguous. Explicitly name the ambiguous term and offer interpretations. If NO critical ambiguity exists, this field MUST be null."
    )
    optimized_query: Optional[str] = Field(
        default=None,
        description="The expanded hybrid BM25 query (max 120 words). Populate ONLY if clarification is NOT required."
    )
    metadata_filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Extracted metadata filters. Do not hallucinate fields."
    )
