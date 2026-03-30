"""Generates PICO-format evidence summary tables from retrieved source documents."""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.config import settings
from app.models.response_schemas import EvidenceTable

logger = logging.getLogger(__name__)

EVIDENCE_TABLE_PROMPT = """You are a biomedical evidence synthesis assistant. Given source articles from a literature search, extract a PICO-format evidence summary table.

For each unique study (identified by PMID), extract:
- **study**: "Author (Year)" citation string from the header
- **population**: Who was studied (e.g., "Adults with BMI >= 30", "Type 2 diabetes patients")
- **intervention**: What treatment was given (e.g., "Semaglutide 2.4mg weekly")
- **comparator**: What it was compared to (e.g., "Placebo", "Lifestyle intervention alone"). Use "N/A" if no comparator.
- **outcome**: Key finding in one sentence (e.g., "14.9% mean weight loss at 68 weeks")
- **study_type**: Publication type (e.g., "RCT", "Meta-Analysis", "Cohort Study")

Extract ONLY information explicitly present in the source text. Do not infer or fabricate details.

SOURCE ARTICLES:
{context}
"""


class EvidenceTableGenerator:
    """Generates PICO evidence tables from retrieved documents via LLM extraction."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_TABLE_PROMPT),
        ])

    def generate(self, formatted_context: str) -> EvidenceTable:
        """Generate an evidence table from pre-formatted context (same format as QA chain)."""
        structured_llm = self.llm.with_structured_output(EvidenceTable)
        chain = self.prompt | structured_llm

        logger.info("Generating PICO evidence table from source documents.")
        result = chain.invoke({"context": formatted_context})
        return result
