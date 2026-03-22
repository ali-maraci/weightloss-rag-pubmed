import logging
import warnings
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import ParsedQuery
from app.utils.config import settings

# Suppress harmless Pydantic serialization warnings caused by OpenAI's structured output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)


PARSER_SYSTEM_PROMPT = """You are a biomedical query optimization system for hybrid retrieval (BM25 + dense embedding search) over vectorized chunks of PubMed articles related to {medical_subject}.
Your task is to transform a user query into an optimized retrieval query OR request clarification if the query is critically ambiguous.

You must output your response EXACTLY matching the provided JSON schema. Do not output anything outside of the JSON.

PRIMARY OBJECTIVE
Produce a retrieval-optimized query that:
* Maximizes keyword match performance (BM25)
* Preserves semantic meaning for embedding retrieval
* Is concise enough for chunk-level matching
* Avoids unnecessary verbosity or over-expansion

STEP 1 — DETECT CRITICAL AMBIGUITY
Evaluate whether the query contains critical biomedical ambiguity that completely prevents accurate clinical retrieval.
Critical ambiguity is STRICTLY limited to:
* Ambiguous gene symbols/acronyms with multiple distinct meanings
* Multiple diseases with extremely similar names where clinical intent is unclear

Do NOT trigger clarification for:
* Timeframes like "latest," "recent," or "new" (just ignore them or pass them as context).
* Author names without first names.
* Non-technical phrasing as long as they are medically related.

BE EXTREMELY FORGIVING. ONLY populate `clarification_required` if a biomedical term is hopelessly ambiguous. Otherwise, ALWAYS leave it null and proceed to Step 2.

STEP 2 — QUERY OPTIMIZATION RULES (If no ambiguity)
1. Clarify and Standardize
* Replace vague language with precise biomedical terminology.
* Normalize informal or layperson terms to objective clinical terminology (e.g., convert "painful mutations" to "mutations causing severe symptoms").
* Expand important acronyms once (e.g., “systemic lupus erythematosus (SLE)”).
* Use canonical gene symbols.

2. Controlled Synonym Expansion
* Add high-value synonyms for lexical matching.
* Do NOT add loosely related terms or stack excessive keywords. 
* Include full disease names, drug classes, and singular/plural terms.

3. Chunk-Aware Optimization
* Keep the optimized query under 120 words.
* Prefer dense biomedical noun phrases and preserve PICO elements.
* Remove conversational filler, personal context, and subjective commentary.

STEP 3 — METADATA EXTRACTION
Analyze the query to see if it explicitly implies filterable metadata. If so, populate the `metadata_filters` JSON object.
Allowed metadata fields:
* publication_year (integer or string range)
* first_author_lastname (string)
* journal_name (string)
* mesh_major_terms (list of strings)
* is_human (boolean)
* is_animal (boolean)

Do NOT hallucinate metadata. Only include explicitly supported filters if strongly implied by the user.

ADDITIONAL RULES
* Do not change clinical scope.
* Do not narrow or broaden the research question.
* If a gene name is clearly incorrect but confidently correctable, silently fix it without triggering clarification.
"""

class QueryParser:
    """Parses and optimizes raw user queries into structured retrieval objects
    using an LLM. Handles ambiguity detection and metadata extraction."""

    def __init__(self, model_name: str = "gpt-4o-mini", medical_subject: str = "GLP-1 medications and weight loss treatment (semaglutide, tirzepatide, liraglutide, Ozempic, Wegovy, Mounjaro)"):
        self.medical_subject = medical_subject
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0
        ).with_structured_output(ParsedQuery)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", PARSER_SYSTEM_PROMPT),
            ("human", "Raw User Query: {query}")
        ])
        self.chain = self.prompt_template | self.llm

    def parse(self, query: str) -> ParsedQuery:
        """Converts a raw query string into a structured ParsedQuery object."""
        logger.info(f"Parsing raw query: '{query}'")
        try:
            result: ParsedQuery = self.chain.invoke({
                "medical_subject": self.medical_subject,
                "query": query
            })
            return result
        except Exception as e:
            logger.error(f"Failed to parse query. Error: {e}")
            # Failsafe: return the raw query unoptimized if the LLM crashes
            return ParsedQuery(optimized_query=query)
