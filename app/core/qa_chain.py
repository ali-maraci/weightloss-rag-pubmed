import json
import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.config import settings
from app.core.retriever import AuraRetriever

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are AuraQuery, an expert biomedical AI assistant designed to answer highly complex medical questions using ONLY the provided verified literature.

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

CONTEXT PROVIDED:
{context}
"""

class AuraQAChain:
    """Orchestrates the final LLM response by running a query through the retriever,
    formatting retrieved chunks into a context block, and generating a cited response."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.retriever = AuraRetriever()
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

    def query(self, raw_query: str) -> Tuple[str, str]:
        """Executes the full RAG pipeline and returns the Markdown answer and strategy used."""
        logger.info(f"Executing End-to-End RAG for query: {raw_query}")
        
        docs = self.retriever.retrieve(raw_query)

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
            fallback_docs = self.retriever.retrieve(raw_query, bypass_stage_1=True)
            
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

    def stream_query(self, raw_query: str):
        """Executes the full RAG pipeline and yields SSE json chunks."""
        logger.info(f"Executing Streaming End-to-End RAG for query: {raw_query}")
        
        docs = []
        is_early_fallback = False

        for event in self.retriever.stream_retrieve(raw_query):
            if event["type"] == "status":
                yield json.dumps({"type": "status", "message": event["message"]}) + "\n\n"
            elif event["type"] == "fallback_trigger":
                is_early_fallback = True
            elif event["type"] == "result":
                docs = event["docs"]

        if is_early_fallback:
            logger.warning("Early fallback triggered. Bypassing Stage 1.")
            fallback_docs = []
            for event in self.retriever.stream_retrieve(raw_query, bypass_stage_1=True):
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

        # Fallback: if the LLM decided context was insufficient, try Index B globally
        if is_fallback and not is_early_fallback:
            logger.warning("LLM reported insufficient evidence from Stage 1 Abstracts. Triggering Global Index B Fallback Search.")
            
            fallback_docs = []
            for event in self.retriever.stream_retrieve(raw_query, bypass_stage_1=True):
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
            else:
                logger.warning("Fallback global search yielded no results either.")

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
