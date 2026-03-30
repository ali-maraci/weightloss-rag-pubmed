import datetime
import logging
import math
import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from app.db.vector_store import VectorStore
from app.core.query_parser import QueryParser
from app.models.schemas import ParsedQuery

logger = logging.getLogger(__name__)

_PMID_PATTERN = re.compile(r'PMID:?\s*(\d{7,8})', re.IGNORECASE)

class Retriever:
    """Two-stage hybrid retrieval pipeline: Index A (abstracts) selects candidate PMIDs,
    Index B (body text) retrieves chunks, followed by reranking and diversity filtering."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.query_parser = QueryParser()

        self.abstract_top_n: int = 50
        self.chunk_top_k: int = 80
        self.max_chunks_per_article: int = 5
        self.target_return_size: int = 30

    def retrieve(self, raw_query: str, bypass_stage_1: bool = False) -> List[Document]:
        """Main retrieval pipeline: parse query, retrieve candidates, rerank, and filter."""
        logger.info(f"Starting retrieval pipeline for query: '{raw_query}' (Bypass Stage 1: {bypass_stage_1})")

        parsed_query = self.query_parser.parse(raw_query)
        logger.debug("Parsed query: %s", parsed_query.model_dump())

        if parsed_query.clarification_required:
            logger.warning(f"Ambiguous query detected: {parsed_query.clarification_required}")
            return [Document(
                page_content=f"System Alert: Do not answer the user's question. Instead, ask them this clarification: {parsed_query.clarification_required}",
                metadata={"type": "clarification"}
            )]

        search_term = parsed_query.optimized_query or raw_query
        extracted_pmids = _PMID_PATTERN.findall(raw_query)

        if bypass_stage_1:
            logger.info("Bypassing Stage 1: Executing Global Fallback Search on Index B.")
            raw_chunks = self._stage_2_global_chunk_search(search_term, parsed_query)
            final_chunks = [doc for doc, _ in raw_chunks][:self.target_return_size]
            logger.info(f"Fallback complete. Yielding {len(final_chunks)} chunks directly from Native Hybrid search.")
            return final_chunks

        elif extracted_pmids:
            logger.info(f"Explicit PMIDs detected in query: {extracted_pmids}. Bypassing Stage 1.")
            candidate_pmids = list(set(extracted_pmids))
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)
        else:
            candidate_pmids = self._stage_1_abstract_search(search_term, parsed_query)
            if not candidate_pmids:
                logger.warning("No candidate abstracts found. Aborting retrieval.")
                return []
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)

        if not raw_chunks:
            logger.warning("No body chunks found for query.")
            return []

        reranked_chunks = self._stage_3_rerank(raw_chunks, parsed_query)
        final_chunks = self._stage_4_diversity_filter(reranked_chunks)

        logger.info(f"Retrieval complete. Yielding {len(final_chunks)} chunks.")
        return final_chunks

    def stream_retrieve(self, raw_query: str, bypass_stage_1: bool = False):
        """Streaming version of retrieve that yields status updates and document results."""
        logger.info(f"Starting stream retrieval pipeline for query: '{raw_query}' (Bypass Stage 1: {bypass_stage_1})")

        parsed_query = self.query_parser.parse(raw_query)

        if parsed_query.clarification_required:
            logger.warning(f"Ambiguous query detected: {parsed_query.clarification_required}")
            yield {"type": "result", "docs": [Document(
                page_content=f"System Alert: Do not answer the user's question. Instead, ask them this clarification: {parsed_query.clarification_required}",
                metadata={"type": "clarification"}
            )]}
            return

        search_term = parsed_query.optimized_query or raw_query
        extracted_pmids = _PMID_PATTERN.findall(raw_query)
        
        if bypass_stage_1:
            logger.info("Bypassing Stage 1: Executing Global Fallback Search on Index B.")
            yield {"type": "status", "message": "Performing more extensive research..."}
            raw_chunks = self._stage_2_global_chunk_search(search_term, parsed_query)
            final_chunks = [doc for doc, _ in raw_chunks][:self.target_return_size]
            yield {"type": "result", "docs": final_chunks}
            return

        elif extracted_pmids:
            logger.info(f"Explicit PMIDs detected in query: {extracted_pmids}. Bypassing Stage 1.")
            candidate_pmids = list(set(extracted_pmids))
            yield {"type": "status", "message": "Retrieving the relevant articles..."}
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)
        else:
            yield {"type": "status", "message": "Scanning PubMed abstracts..."}
            candidate_pmids = self._stage_1_abstract_search(search_term, parsed_query)
            if not candidate_pmids:
                logger.warning("No candidate abstracts found. Yielding fallback trigger.")
                yield {"type": "fallback_trigger"}
                yield {"type": "result", "docs": []}
                return
            yield {"type": "status", "message": "Retrieving the relevant articles..."}
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)

        if not raw_chunks:
            logger.warning("No body chunks found for query. Yielding fallback trigger.")
            yield {"type": "fallback_trigger"}
            yield {"type": "result", "docs": []}
            return

        reranked_chunks = self._stage_3_rerank(raw_chunks, parsed_query)
        final_chunks = self._stage_4_diversity_filter(reranked_chunks)

        logger.info(f"Stream retrieval complete. Yielding {len(final_chunks)} chunks.")
        yield {"type": "result", "docs": final_chunks}

    def _build_qdrant_filter(self, parsed_query: ParsedQuery) -> Any:
        """Translates ParsedQuery metadata filters into a Qdrant models.Filter object."""
        from qdrant_client.http import models
        if not parsed_query.metadata_filters:
            return None
            
        must_conditions = []
        meta = parsed_query.metadata_filters
        
        if meta.publication_year:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.pub_year",
                    match=models.MatchValue(value=meta.publication_year)
                )
            )
        if meta.first_author_lastname:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.first_author_lastname",
                    match=models.MatchValue(value=meta.first_author_lastname)
                )
            )
        # Note: is_human / is_animal filters are skipped because Qdrant
        # collections don't have payload indexes for these boolean fields.
        # To enable, create indexes first via setup_qdrant_collections.py.

        if not must_conditions:
            return None
            
        return models.Filter(must=must_conditions)

    def _stage_1_abstract_search(self, search_term: str, parsed_query: ParsedQuery) -> List[str]:
        """Searches Index A (abstracts) to derive candidate PMIDs."""
        logger.info(f"Executing Stage 1 Search on Index A. Term: '{search_term}'")
        
        stage_1_filter = self._build_qdrant_filter(parsed_query)
        kwargs = {"query": search_term, "k": self.abstract_top_n * 2}
        
        if stage_1_filter:
            logger.info(f"Applying strict Stage 1 Qdrant metadata filter.")
            kwargs["filter"] = stage_1_filter

        try:
            results = self.vector_store.collection_a.similarity_search_with_relevance_scores(**kwargs)
        except Exception as e:
            logger.error(f"Stage 1 search failed with filter: {e}")
            # Failsafe: drop the filter and try again
            results = self.vector_store.collection_a.similarity_search_with_relevance_scores(
                query=search_term, k=self.abstract_top_n * 2
            )

        unique_pmids: List[str] = []
        seen: set = set()
        for doc, score in results:
            pmid = doc.metadata.get("pmid")
            if pmid and pmid not in seen:
                seen.add(pmid)
                unique_pmids.append(pmid)
                if len(unique_pmids) == self.abstract_top_n:
                    break

        logger.info(f"Stage 1 complete. Isolated {len(unique_pmids)} candidate PMIDs.")
        return unique_pmids

    def _stage_2_chunk_search(self, search_term: str, candidate_pmids: List[str]) -> List[Tuple[Document, float]]:
        """Searches Index B (body text) restricted to the given candidate PMIDs."""
        from qdrant_client.http import models
        logger.info("Executing Stage 2 Deep Search on Index B.")
        
        pmid_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.pmid",
                    match=models.MatchAny(any=candidate_pmids)
                )
            ]
        )

        dense_results = self.vector_store.collection_b.similarity_search_with_relevance_scores(
            query=search_term,
            k=self.chunk_top_k,
            filter=pmid_filter
        )

        return dense_results

    def _stage_2_global_chunk_search(self, search_term: str, parsed_query: ParsedQuery) -> List[Tuple[Document, float]]:
        """Executes a global fallback search across all chunks in Index B."""
        logger.info("Executing Global Stage 2 Deep Search uniformly across Index B.")
        
        global_filter = self._build_qdrant_filter(parsed_query)
        kwargs = {"query": search_term, "k": self.chunk_top_k * 2}
        
        if global_filter:
            logger.info("Applying strict Global Qdrant metadata filter on Index B.")
            kwargs["filter"] = global_filter

        try:
            dense_results = self.vector_store.collection_b.similarity_search_with_relevance_scores(**kwargs)
        except Exception as e:
            logger.error(f"Global Index B search failed with filter: {e}")
            dense_results = self.vector_store.collection_b.similarity_search_with_relevance_scores(
                query=search_term, k=self.chunk_top_k * 2
            )

        return dense_results

    def _stage_3_rerank(self, scored_docs: List[Tuple[Document, float]], parsed_query: ParsedQuery) -> List[Tuple[Document, float]]:
        """Applies boosts for publication type, section relevance, and recency."""
        logger.info("Executing Stage 3 Custom Reranking.")
        reranked: List[Tuple[Document, float]] = []
        current_year = datetime.datetime.now().year

        for doc, base_score in scored_docs:
            meta = doc.metadata
            final_score = base_score

            pub_types = meta.get("publication_types", "")
            if "Meta-Analysis" in pub_types:
                final_score += 1.00
            elif "Systematic Review" in pub_types:
                final_score += 0.95
            elif "Guideline" in pub_types or "Practice Guideline" in pub_types:
                final_score += 0.90
            elif "Randomized Controlled Trial" in pub_types:
                final_score += 0.85
            elif "Clinical Trial" in pub_types:
                final_score += 0.75
            elif "Review" in pub_types:
                final_score += 0.55
            elif "Case Reports" in pub_types:
                final_score += 0.50
            else:
                final_score += 0.60

            h2 = meta.get("Header 2", "").lower()
            if "result" in h2:
                final_score += 1.5
            elif "conclusion" in h2:
                final_score += 1.5
            elif "method" in h2:
                final_score += 1.0
            elif "discussion" in h2:
                final_score += 0.5
            elif "introduction" in h2 or "background" in h2:
                final_score -= 0.5

            pub_year = meta.get("pub_year", meta.get("publication_year"))
            if pub_year and pub_year != "Unknown":
                try:
                    year_diff = current_year - int(pub_year)
                    if year_diff >= 0:
                        recency_weight = 0.25 * math.exp(-year_diff / 8)
                        final_score += recency_weight
                except ValueError:
                    pass

            doc.metadata["rerank_score"] = final_score
            doc.metadata["base_vector_score"] = base_score
            reranked.append((doc, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _stage_4_diversity_filter(self, ranked_docs: List[Tuple[Document, float]]) -> List[Document]:
        """Enforces max_chunks per PMID to prevent any single article from dominating."""
        logger.info("Executing Stage 4 Diversity Filtering.")
        final_list: List[Document] = []
        pmid_counts: Dict[str, int] = {}

        for doc, _ in ranked_docs:
            pmid = doc.metadata.get("pmid", "Unknown")
            count = pmid_counts.get(pmid, 0)

            if count >= self.max_chunks_per_article:
                continue

            final_list.append(doc)
            pmid_counts[pmid] = count + 1

            if len(final_list) >= self.target_return_size:
                break

        return final_list
