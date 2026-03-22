import logging
from typing import List, Dict, Any, Tuple, Optional
import datetime
import re
import math

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.db.vector_store import AuraVectorStore
from app.core.query_parser import QueryParser
from app.models.schemas import ParsedQuery

logger = logging.getLogger(__name__)

_PMID_PATTERN = re.compile(r'PMID:?\s*(\d{7,8})', re.IGNORECASE)

class AuraRetriever:
    """
    Implements a High-Performance Two-Stage Hybrid Retrieval Pipeline.
    Stage 1: Index A (Abstracts) -> Selects Top unique PMIDs
    Stage 2: Index B (Body Text) -> Drills into specifically selected PMIDs
    Stage 3: Custom Reranking (Methodologies, Recency)
    Stage 4: Diversity Filtering (Max chunks per paper)
    """

    def __init__(self) -> None:
        """Initializes the retriever with vector store connection and tunable hyperparameters."""
        self.vector_store = AuraVectorStore()
        self.query_parser = QueryParser()
        
        # Adjustable Hyperparameters
        self.abstract_top_n: int = 50
        self.chunk_top_k: int = 80
        self.max_chunks_per_article: int = 5
        self.target_return_size: int = 30

    def retrieve(self, raw_query: str, bypass_stage_1: bool = False) -> List[Document]:
        """
        The main orchestration pipeline for retrieving relevant documents based on a query.

        Args:
            raw_query (str): The raw user or chat-engine query.
            bypass_stage_1 (bool, optional): If true, skips abstract filtering and queries Index B globally. Defaults to False.

        Returns:
            List[Document]: A list of LangChain Document objects representing the most relevant chunks.
        """
        logger.info(f"Starting retrieval pipeline for query: '{raw_query}' (Bypass Stage 1: {bypass_stage_1})")
        
        # 1. Parse Query
        parsed_query = self.query_parser.parse(raw_query)
        
        logger.debug("Parsed query: %s", parsed_query.model_dump())

        if parsed_query.clarification_required:
            logger.warning(f"Ambiguous query detected: {parsed_query.clarification_required}")
            return [Document(
                page_content=f"System Alert: Do not answer the user's question. Instead, ask them this clarification: {parsed_query.clarification_required}",
                metadata={"type": "clarification"}
            )]
            
        search_term = parsed_query.optimized_query or raw_query
        
        # --- EXPLICIT PMID OVERRIDE ---
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
            # 2. Stage 1: Candidate Article Retrieval (Index A)
            candidate_pmids = self._stage_1_abstract_search(search_term, parsed_query)
            if not candidate_pmids:
                logger.warning("No candidate abstracts found. Aborting retrieval.")
                return []
                
            # 3. Stage 2: Chunk-Level Retrieval (Index B) Restricted to Candidates
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)
            
        if not raw_chunks:
            logger.warning("No body chunks found for query.")
            return []
            
        # 4. Stage 3: Metadata-Aware Reranking
        reranked_chunks = self._stage_3_rerank(raw_chunks, parsed_query)
        
        # 5. Stage 4: Diversity Filtering
        final_chunks = self._stage_4_diversity_filter(reranked_chunks)
        
        logger.info(f"Retrieval complete. Yielding {len(final_chunks)} perfectly curated chunks.")
        return final_chunks

    def stream_retrieve(self, raw_query: str, bypass_stage_1: bool = False):
        """Streaming generator version of retrieve that yields status updates and returns docs."""
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
            # 2. Stage 1: Candidate Article Retrieval (Index A)
            yield {"type": "status", "message": "Scanning PubMed abstracts..."}
            candidate_pmids = self._stage_1_abstract_search(search_term, parsed_query)
            if not candidate_pmids:
                logger.warning("No candidate abstracts found. Yielding fallback trigger.")
                yield {"type": "fallback_trigger"}
                yield {"type": "result", "docs": []}
                return
                
            # 3. Stage 2: Chunk-Level Retrieval (Index B) Restricted to Candidates
            yield {"type": "status", "message": "Retrieving the relevant articles..."}
            raw_chunks = self._stage_2_chunk_search(search_term, candidate_pmids)
            
        if not raw_chunks:
            logger.warning("No body chunks found for query. Yielding fallback trigger.")
            yield {"type": "fallback_trigger"}
            yield {"type": "result", "docs": []}
            return
            
        # 4. Stage 3: Metadata-Aware Reranking
        reranked_chunks = self._stage_3_rerank(raw_chunks, parsed_query)
        
        # 5. Stage 4: Diversity Filtering
        final_chunks = self._stage_4_diversity_filter(reranked_chunks)
        
        logger.info(f"Stream Retrieval complete. Yielding {len(final_chunks)} perfectly curated chunks.")
        yield {"type": "result", "docs": final_chunks}

    def _build_qdrant_filter(self, parsed_query: ParsedQuery) -> Any:
        """
        Translates the Pydantic MetadataFilters into a Qdrant-compliant models.Filter object.

        Args:
            parsed_query (ParsedQuery): The parsed query containing metadata filters.

        Returns:
            Any: A Qdrant models.Filter object or None.
        """
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
        if meta.is_human is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.is_human",
                    match=models.MatchValue(value=meta.is_human)
                )
            )
        if meta.is_animal is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.is_animal",
                    match=models.MatchValue(value=meta.is_animal)
                )
            )
            
        if not must_conditions:
            return None
            
        return models.Filter(must=must_conditions)

    def _stage_1_abstract_search(self, search_term: str, parsed_query: ParsedQuery) -> List[str]:
        """
        Performs Dense/Hybrid Search on Index A (Abstracts) to derive candidate PMIDs.

        Args:
            search_term (str): The search query.
            parsed_query (ParsedQuery): The parsed object containing filters.

        Returns:
            List[str]: A list of candidate PMIDs extracted from the top retrieved abstracts.
        """
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
        """
        Restricts deep body search strictly to the subset of candidate PMIDs using Qdrant MatchAny.

        Args:
            search_term (str): The term to search within the chunks.
            candidate_pmids (List[str]): List of PMIDs to restrict the search to.

        Returns:
            List[Tuple[Document, float]]: List of retrieved documents and their vector similarity scores.
        """
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
        """
        Executes a global search directly against all chunks in Index B (Fallback method).

        Args:
            search_term (str): The search query.
            parsed_query (ParsedQuery): The parsed query with active metadata filters.

        Returns:
            List[Tuple[Document, float]]: The raw documents and baseline similarity scores.
        """
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
        """
        Applies algorithmic boosts to favor RCTs, robust methodologies, and recent papers.

        Args:
            scored_docs (List[Tuple[Document, float]]): The baseline ranked documents.
            parsed_query (ParsedQuery): Input queries.

        Returns:
            List[Tuple[Document, float]]: The custom reranked list of documents.
        """
        logger.info("Executing Stage 3 Custom Reranking.")
        reranked: List[Tuple[Document, float]] = []
        
        current_year = datetime.datetime.now().year
        
        for doc, base_score in scored_docs:
            meta = doc.metadata
            final_score = base_score
            
            # --- PUBLICATION TYPE WEIGHTING ---
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
                
            # --- SECTION WEIGHTING (Index B) ---
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
                
            # --- RECENCY BOOST ---
            pub_year = meta.get("pub_year", meta.get("publication_year"))
            if pub_year and pub_year != "Unknown":
                try:
                    year_diff = current_year - int(pub_year)
                    if year_diff >= 0:
                        recency_weight = 0.25 * math.exp(-year_diff / 8)
                        final_score += recency_weight
                except ValueError:
                    pass
                    
            doc.metadata["aura_rerank_score"] = final_score
            doc.metadata["aura_base_vector_score"] = base_score
            reranked.append((doc, final_score))
            
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _stage_4_diversity_filter(self, ranked_docs: List[Tuple[Document, float]]) -> List[Document]:
        """
        Enforces max_chunks per PMID to prevent review articles from dominating the LLM context.

        Args:
            ranked_docs (List[Tuple[Document, float]]): Custom reranked chunks.

        Returns:
            List[Document]: Truncated, diverse list of chunks constrained per article.
        """
        logger.info("Executing Stage 4 Diversity Filtering.")
        final_list: List[Document] = []
        pmid_counts: Dict[str, int] = {}
        
        for doc, score in ranked_docs:
            pmid = doc.metadata.get("pmid", "Unknown")
            
            if pmid not in pmid_counts:
                pmid_counts[pmid] = 0
                
            if pmid_counts[pmid] >= self.max_chunks_per_article:
                continue 
                
            final_list.append(doc)
            pmid_counts[pmid] += 1
            
            if len(final_list) >= self.target_return_size:
                break
                
        return final_list
