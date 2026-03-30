import json
import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.config import settings
from app.core.retriever import Retriever
from app.models.response_schemas import StructuredAnswer

logger = logging.getLogger(__name__)

STRUCTURED_EXTRACTION_PROMPT = """You are a precise biomedical text analyst. Your task is to decompose a free-text medical answer into a structured format.

Given the ORIGINAL QUESTION and the ANSWER below, extract the following:

1. **summary**: A 1-3 sentence high-level answer to the query, capturing the key takeaway.

2. **claims**: Decompose the answer into individual, atomic claims. For each claim:
   - **statement**: A single-sentence assertion.
   - **claim_type**: Classify as one of:
     - "evidence" — directly stated in or clearly derived from a cited source
     - "inference" — a logical deduction or synthesis combining multiple sources
     - "background" — general medical/scientific context not tied to a specific citation
   - **pmids**: Extract ALL PMIDs cited in support of this claim. Use only the numeric PMID (e.g. "12345678"), not the full bracket notation. If no PMID is cited for the claim, leave this as an empty list.
   - **confidence**: Based on language in the answer, assign one of: "strong" (definitive language, large studies, meta-analyses), "moderate" (suggestive evidence, smaller studies), "limited" (case reports, preliminary data, mixed results), or null if not determinable.

3. **source_pmids**: A flat list of ALL unique PMIDs referenced anywhere in the answer.

4. **limitations**: Any caveats, gaps in the literature, conflicting evidence, or areas of uncertainty mentioned in the answer. Set to null if none are mentioned.

5. **conflicts**: List any cases where different studies in the answer present contradictory or conflicting findings. Each entry should be a brief description of the disagreement. Set to an empty list if no conflicts are present.

6. **uncertainty_notes**: List areas where the answer uses hedging language or notes limited evidence. Each entry should describe what is uncertain and why. Set to an empty list if the evidence is clear.

7. **literature_gaps**: List any aspects of the question that the answer could not address due to lack of published evidence. Set to an empty list if the answer fully covers the question.

ORIGINAL QUESTION:
{question}

ANSWER:
{answer}
"""

QA_SYSTEM_PROMPT = """You are a an expert biomedical AI assistant designed to answer highly complex medical questions using ONLY the provided verified literature.

You must answer the user's question accurately, directly, and comprehensively based explicitly on the Context Provided below.

CRITICAL RULES:
1. NO HALLUCINATION: If the context does not contain the answer, state clearly: "I couldn't find sufficient evidence in the literature to answer this question." Do not attempt to guess or use outside knowledge.
2. CITATION DENSITY (CRITICAL): Every distinct medical claim, statistic, mutation, or observation MUST be cited in-line using the exact metadata provided in each chunk's header.
   - You MUST include the PMID in EVERY citation.
   - Format: (First Author Last Name, Year) [PMID: XXXXXX]

3. CONTEXT RANKING & COMPREHENSIVE SYNTHESIS
   - The provided articles are ordered by RELEVANCE (Article 1 is the most highly ranked match to the user's query).
   - You should prioritize information from higher-ranked articles, but actively synthesize relevant facts from as many distinct articles as possible to build a holistic, comprehensive answer.
   - DO NOT force citations from irrelevant articles just to increase citation count. Accuracy is paramount.
   - However, if multiple articles provide relevant, accurate nuances, you MUST synthesize and cite them all rather than relying solely on the first article you read.

4. TONE: Professional, clinical, and objective.

5. CONFLICT DETECTION: If different studies in the context present conflicting findings (e.g., one shows benefit while another shows no effect), you MUST explicitly acknowledge the disagreement. State what each study found and let the reader assess the balance of evidence.

6. CALIBRATED UNCERTAINTY: Match your language to the strength of the evidence:
   - Strong evidence (multiple RCTs, meta-analyses): "Evidence demonstrates...", "Studies consistently show..."
   - Moderate evidence (single RCT, cohort studies): "Evidence suggests...", "A study found..."
   - Limited evidence (case reports, small samples): "Preliminary evidence indicates...", "Limited data suggest..."
   - Do NOT use definitive language for weak evidence or hedging language for strong evidence.

7. LITERATURE GAPS: If the provided context does not adequately cover an aspect of the user's question, explicitly note what is missing rather than silently omitting it. For example: "The available literature does not specifically address [topic]."

CONTEXT PROVIDED:
{context}
"""

class QAChain:
    """Orchestrates the final LLM response by running a query through the retriever,
    formatting retrieved chunks into a context block, and generating a cited response."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.retriever = Retriever()
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

    def _apply_filters(self, raw_query: str, filters: dict = None) -> str:
        """Append filter hints to the query for the retriever's query parser."""
        if not filters:
            return raw_query

        parts = [raw_query]
        if "year_min" in filters or "year_max" in filters:
            year_min = filters.get("year_min", "")
            year_max = filters.get("year_max", "")
            if year_min and year_max:
                parts.append(f"[Filter: published between {year_min} and {year_max}]")
            elif year_min:
                parts.append(f"[Filter: published after {year_min}]")
            elif year_max:
                parts.append(f"[Filter: published before {year_max}]")
        if "publication_types" in filters:
            types = ", ".join(filters["publication_types"])
            parts.append(f"[Filter: study types: {types}]")
        if "human_only" in filters and filters["human_only"]:
            parts.append("[Filter: human studies only]")
        return " ".join(parts)

    def query(self, raw_query: str, filters: dict = None) -> Tuple[str, str]:
        """Executes the full RAG pipeline and returns the Markdown answer and strategy used."""
        logger.info(f"Executing End-to-End RAG for query: {raw_query}")

        augmented_query = self._apply_filters(raw_query, filters)
        docs = self.retriever.retrieve(augmented_query)

        if not docs:
            return "No relevant literature could be found to answer this query.", "Failed"

        if docs[0].metadata.get("type") == "clarification":
            return docs[0].page_content.replace("System Alert: Do not answer the user's question. Instead, ask them this clarification: ", ""), "Clarification"

        formatted_context = self._format_docs(docs)
        logger.info(f"Context compiled. Sending {len(docs)} chunks to LLM payload.")

        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": formatted_context,
            "question": raw_query
        })
        answer = self._standardize_citations(response.content)

        # Fallback: if the LLM reports insufficient evidence, bypass Stage 1 and
        # force a deep global search on Index B.
        if "I couldn't find sufficient evidence" in answer:
            logger.warning("LLM reported insufficient evidence from Stage 1 Abstracts. Triggering Global Index B Fallback Search.")
            fallback_docs = self.retriever.retrieve(augmented_query, bypass_stage_1=True)

            if fallback_docs:
                logger.info(f"Fallback retrieved {len(fallback_docs)} chunks from Index B. Re-prompting LLM.")
                fallback_context = self._format_docs(fallback_docs)
                fallback_response = chain.invoke({
                    "context": fallback_context,
                    "question": raw_query
                })
                return self._standardize_citations(fallback_response.content), "Bypassed Index A"
            else:
                logger.warning("Fallback global search yielded no results either.")
                return answer, "Index A -> Index B"

        return answer, "Index A -> Index B"

    def structured_query(self, raw_query: str, filters: dict = None) -> StructuredAnswer:
        """Executes the full RAG pipeline, then parses the free-text answer into a StructuredAnswer."""
        logger.info(f"Executing structured query for: {raw_query}")

        answer, strategy = self.query(raw_query, filters=filters)

        # For failure or clarification responses, return a minimal StructuredAnswer
        if strategy in ("Failed", "Clarification"):
            return StructuredAnswer(summary=answer)

        # Second LLM call: parse the free-text answer into structured claims
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", STRUCTURED_EXTRACTION_PROMPT),
        ])

        structured_llm = self.llm.with_structured_output(StructuredAnswer)
        extraction_chain = extraction_prompt | structured_llm

        logger.info("Invoking structured extraction LLM call.")
        structured_answer = extraction_chain.invoke({
            "question": raw_query,
            "answer": answer,
        })

        return structured_answer

    def stream_query(self, raw_query: str, filters: dict = None):
        """Executes the full RAG pipeline and yields SSE json chunks."""
        logger.info(f"Executing Streaming End-to-End RAG for query: {raw_query}")

        augmented_query = self._apply_filters(raw_query, filters)
        docs = []
        is_early_fallback = False

        for event in self.retriever.stream_retrieve(augmented_query):
            if event["type"] == "status":
                yield json.dumps({"type": "status", "message": event["message"]}) + "\n\n"
            elif event["type"] == "fallback_trigger":
                is_early_fallback = True
            elif event["type"] == "result":
                docs = event["docs"]

        if is_early_fallback:
            logger.warning("Early fallback triggered. Bypassing Stage 1.")
            fallback_docs = []
            for event in self.retriever.stream_retrieve(augmented_query, bypass_stage_1=True):
                if event["type"] == "status":
                    yield json.dumps({"type": "status", "message": event["message"]}) + "\n\n"
                elif event["type"] == "result":
                    fallback_docs = event["docs"]
            docs = fallback_docs

        if not docs:
            yield json.dumps({"type": "token", "content": "No relevant literature could be found to answer this query."}) + "\n\n"
            return

        if docs[0].metadata.get("type") == "clarification":
            yield json.dumps({"type": "token", "content": docs[0].page_content.replace("System Alert: Do not answer the user's question. Instead, ask them this clarification: ", "")}) + "\n\n"
            return

        formatted_context = self._format_docs(docs)
        logger.info(f"Context compiled. Sending {len(docs)} chunks to LLM payload.")
        
        yield json.dumps({"type": "status", "message": "Synthesizing clinical evidence..."}) + "\n\n"
        chain = self.prompt_template | self.llm
        
        full_answer = ""
        buffer = ""
        is_fallback = False

        for chunk in chain.stream({
            "context": formatted_context,
            "question": raw_query
        }):
            content = chunk.content
            full_answer += content

            # Buffer the start of the response to detect the fallback phrase
            if not is_fallback and len(full_answer) < 50:
                buffer += content
                if "I couldn't find sufficient evidence" in full_answer:
                    is_fallback = True
                    break
            else:
                if buffer:
                    yield json.dumps({"type": "token", "content": buffer}) + "\n\n"
                    buffer = ""
                yield json.dumps({"type": "token", "content": content}) + "\n\n"

        # Flush any remaining buffer (short answers that never exceeded 50 chars)
        if buffer and not is_fallback:
            yield json.dumps({"type": "token", "content": buffer}) + "\n\n"
            buffer = ""

        # Fallback: if the LLM decided context was insufficient, try Index B globally
        if is_fallback and not is_early_fallback:
            logger.warning("LLM reported insufficient evidence from Stage 1 Abstracts. Triggering Global Index B Fallback Search.")

            fallback_docs = []
            for event in self.retriever.stream_retrieve(augmented_query, bypass_stage_1=True):
                if event["type"] == "status":
                    yield json.dumps({"type": "status", "message": event["message"]}) + "\n\n"
                elif event["type"] == "result":
                    fallback_docs = event["docs"]


            if fallback_docs:
                logger.info(f"Fallback retrieved {len(fallback_docs)} chunks from Index B. Re-prompting LLM.")
                fallback_context = self._format_docs(fallback_docs)
                yield json.dumps({"type": "status", "message": "Synthesizing clinical evidence..."}) + "\n\n"

                for chunk in chain.stream({
                    "context": fallback_context,
                    "question": raw_query
                }):
                    yield json.dumps({"type": "token", "content": chunk.content}) + "\n\n"

                yield json.dumps({"type": "sources", "docs": self._docs_to_source_list(fallback_docs)}) + "\n\n"
                try:
                    from app.core.evidence_table import EvidenceTableGenerator
                    evidence_table = EvidenceTableGenerator().generate(fallback_context)
                    yield json.dumps({"type": "evidence_table", "rows": [row.model_dump() for row in evidence_table.rows]}) + "\n\n"
                except Exception as e:
                    logger.warning(f"Evidence table generation failed: {e}")
            else:
                logger.warning("Fallback global search yielded no results either.")
        else:
            # Normal completion path (or early-fallback that streamed fine) — emit sources
            yield json.dumps({"type": "sources", "docs": self._docs_to_source_list(docs)}) + "\n\n"
            try:
                from app.core.evidence_table import EvidenceTableGenerator
                evidence_table = EvidenceTableGenerator().generate(formatted_context)
                yield json.dumps({"type": "evidence_table", "rows": [row.model_dump() for row in evidence_table.rows]}) + "\n\n"
            except Exception as e:
                logger.warning(f"Evidence table generation failed: {e}")

    def _docs_to_source_list(self, docs: List[Document]) -> List[dict]:
        """Converts retrieved Documents to a deduplicated list of SourceDocument-compatible dicts.

        Multiple chunks from the same article are collapsed into one entry using the
        first chunk's metadata. Results are sorted by rerank_score descending."""
        seen: dict = {}  # pmid -> dict

        for doc in docs:
            pmid = doc.metadata.get("pmid", "Unknown")
            if pmid in seen:
                continue  # keep first chunk's metadata only

            meta = doc.metadata
            pub_types = meta.get("publication_types", [])
            publication_type = pub_types[0] if pub_types else ""

            seen[pmid] = {
                "pmid": pmid,
                "title": meta.get("article_title", ""),
                "journal": meta.get("journal", ""),
                "year": meta.get("pub_year") or None,
                "first_author": meta.get("first_author_lastname", ""),
                "publication_type": publication_type,
                "rerank_score": meta.get("rerank_score"),
                "section": meta.get("section", ""),
                "snippet": doc.page_content[:200],
            }

        return sorted(
            seen.values(),
            key=lambda d: (d["rerank_score"] is None, -(d["rerank_score"] or 0)),
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Groups chunks by PMID while maintaining relevance ranking order,
        then formats them into a context block for the LLM prompt."""
        grouped_docs = {}
        for idx, doc in enumerate(docs):
            pmid = doc.metadata.get("pmid", "Unknown")
            if pmid not in grouped_docs:
                grouped_docs[pmid] = {
                    "chunks": [],
                    "best_rank": idx
                }
            grouped_docs[pmid]["chunks"].append(doc)

        sorted_pmids = sorted(grouped_docs.keys(), key=lambda p: grouped_docs[p]["best_rank"])

        formatted_strings = []
        article_counter = 1

        for pmid in sorted_pmids:
            chunks = grouped_docs[pmid]["chunks"]
            meta = chunks[0].metadata
            year = meta.get("pub_year", meta.get("publication_year"))
            author = meta.get("first_author_lastname")

            if author and year and author != "Unknown" and year != "Unknown":
                citation_header = f"--- ARTICLE {article_counter}: ({author}, {year}) [PMID: {pmid}] ---"
            elif author and author != "Unknown":
                citation_header = f"--- ARTICLE {article_counter}: ({author}) [PMID: {pmid}] ---"
            elif year and year != "Unknown":
                citation_header = f"--- ARTICLE {article_counter}: ({year}) [PMID: {pmid}] ---"
            else:
                citation_header = f"--- ARTICLE {article_counter}: [PMID: {pmid}] ---"


            combined_content = "\n...\n".join([chunk.page_content.strip() for chunk in chunks])
            formatted_strings.append(f"{citation_header}\n{combined_content}\n")
            article_counter += 1

        return "\n".join(formatted_strings)

    def _standardize_citations(self, text: str) -> str:
        """Normalizes rogue citation formats back to the strict `[PMID: XXXXXX]` format
        expected by the frontend citation renderer."""
        # Handle combined author/PMID brackets: [Author, Year; PMID: 123456] -> (Author, Year) [PMID: 123456]
        pattern_author_combined = r'\[([^\]]*?);\s*PMID:\s*(\d+)\]'
        text = re.sub(pattern_author_combined, r'(\1) [PMID: \2]', text)


        # Handle multiple PMIDs in one bracket: [PMID: 123, 456, 789] -> [PMID: 123] [PMID: 456] [PMID: 789]
        def explode_pmids(match):
            numbers = re.findall(r'\d+', match.group(1))
            if numbers:
                return ' '.join([f'[PMID: {num}]' for num in numbers])
            return match.group(0)
            
        pattern_multi_pmid = r'\[PMID:\s*([\d,\s]+)\]'
        text = re.sub(pattern_multi_pmid, explode_pmids, text)
        
        return text
