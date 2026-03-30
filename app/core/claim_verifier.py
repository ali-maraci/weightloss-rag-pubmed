"""Verifies structured claims against their cited source documents."""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.config import settings
from app.models.response_schemas import Claim, VerifiedClaim, SupportLevel

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """You are a precise biomedical fact-checker. Given a CLAIM and the SOURCE TEXT from the cited papers, determine how well the source supports the claim.

Classify the support level as one of:
- "supported" — the source text directly states or strongly implies the claim
- "partial" — the source text partially supports the claim but with important caveats, qualifications, or missing details
- "unsupported" — the source text does not address or support this claim
- "contradictory" — the source text contradicts the claim

CLAIM:
{claim}

CITED PMIDs: {pmids}

SOURCE TEXT FROM CITED PAPERS:
{source_text}

Provide your assessment as a VerifiedClaim."""


class ClaimVerifier:
    """Compares each claim against its cited source text to classify support level."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", VERIFICATION_PROMPT),
        ])

    def verify(self, claims: List[Claim], source_docs: List[Document]) -> List[VerifiedClaim]:
        """Verify each claim against its cited sources. Returns VerifiedClaims."""
        # Build a PMID -> text mapping from source docs
        pmid_texts = {}
        for doc in source_docs:
            pmid = doc.metadata.get("pmid", "")
            if pmid:
                if pmid not in pmid_texts:
                    pmid_texts[pmid] = []
                pmid_texts[pmid].append(doc.page_content[:500])

        verified = []
        structured_llm = self.llm.with_structured_output(VerifiedClaim)
        chain = self.prompt | structured_llm

        for claim in claims:
            # Background claims without citations are automatically "supported"
            if claim.claim_type == "background" and not claim.pmids:
                verified.append(VerifiedClaim(
                    **claim.model_dump(),
                    support_level=SupportLevel.SUPPORTED,
                    verification_reasoning="Background knowledge claim without specific citation.",
                ))
                continue

            # Gather source text for this claim's PMIDs
            relevant_text = []
            for pmid in claim.pmids:
                if pmid in pmid_texts:
                    relevant_text.extend(pmid_texts[pmid])

            if not relevant_text:
                verified.append(VerifiedClaim(
                    **claim.model_dump(),
                    support_level=SupportLevel.UNSUPPORTED,
                    verification_reasoning="Cited PMIDs not found in retrieved source documents.",
                ))
                continue

            # LLM verification call
            try:
                result = chain.invoke({
                    "claim": claim.statement,
                    "pmids": ", ".join(claim.pmids),
                    "source_text": "\n---\n".join(relevant_text),
                })
                verified.append(result)
            except Exception as e:
                logger.error(f"Verification failed for claim: {e}")
                verified.append(VerifiedClaim(
                    **claim.model_dump(),
                    support_level=SupportLevel.PARTIAL,
                    verification_reasoning=f"Verification error: {str(e)}",
                ))

        return verified
