"""Pydantic models for structured RAG responses with claim-level citation attribution."""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Classification of a claim's epistemic status."""
    EVIDENCE = "evidence"       # Directly stated in or derived from a cited source
    INFERENCE = "inference"     # Logical deduction combining multiple sources
    BACKGROUND = "background"   # General medical/scientific context


class Claim(BaseModel):
    """A single assertion extracted from the RAG answer, with its supporting citations."""
    statement: str = Field(description="The claim text as a single sentence.")
    claim_type: ClaimType = Field(description="Whether this is direct evidence, inference, or background.")
    pmids: List[str] = Field(
        default_factory=list,
        description="PMIDs of the sources supporting this claim.",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Qualifier like 'strong', 'moderate', 'limited' based on evidence quality.",
    )


class StructuredAnswer(BaseModel):
    """A fully structured RAG response decomposed into claims with citation metadata."""
    summary: str = Field(description="A 1-3 sentence high-level answer to the query.")
    claims: List[Claim] = Field(
        default_factory=list,
        description="Individual claims that compose the full answer.",
    )
    source_pmids: List[str] = Field(
        default_factory=list,
        description="All unique PMIDs referenced across all claims.",
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Caveats, gaps in the literature, or areas of uncertainty.",
    )
    conflicts: List[str] = Field(
        default_factory=list,
        description="Descriptions of conflicting findings between sources.",
    )
    uncertainty_notes: List[str] = Field(
        default_factory=list,
        description="Areas where the evidence is limited, mixed, or inconclusive.",
    )
    literature_gaps: List[str] = Field(
        default_factory=list,
        description="Topics mentioned in the query that lack sufficient published evidence.",
    )
    safety_level: Optional[str] = Field(
        default=None,
        description="Risk classification of the query: 'low', 'moderate', or 'high'.",
    )
    safety_disclaimer: Optional[str] = Field(
        default=None,
        description="Medical disclaimer text if safety_level is moderate or high.",
    )


class SupportLevel(str, Enum):
    """How well a claim is supported by the cited source text."""
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTORY = "contradictory"


class VerifiedClaim(Claim):
    """A claim that has been verified against its source documents."""
    support_level: SupportLevel = Field(description="How well the source text supports this claim.")
    verification_reasoning: str = Field(
        default="",
        description="Brief explanation of the verification result.",
    )


class SourceDocument(BaseModel):
    """Metadata about a retrieved source document, exposed to the frontend."""
    pmid: str
    title: str = ""
    journal: str = ""
    year: Optional[int] = None
    first_author: str = ""
    publication_type: str = ""
    rerank_score: Optional[float] = None
    section: str = ""
    snippet: str = Field(default="", description="First ~200 chars of the chunk content.")


class EvidenceTableRow(BaseModel):
    """One row of the evidence summary table (PICO format)."""
    pmid: str
    study: str = Field(description="Author (Year) citation string.")
    population: str = Field(default="", description="Study population description.")
    intervention: str = Field(default="", description="Treatment or intervention studied.")
    comparator: str = Field(default="", description="Control group or comparator.")
    outcome: str = Field(default="", description="Key findings or results.")
    study_type: str = Field(default="", description="Publication type (RCT, Meta-Analysis, etc.).")


class EvidenceTable(BaseModel):
    """PICO-format evidence summary table generated from source documents."""
    rows: List[EvidenceTableRow] = Field(default_factory=list)
