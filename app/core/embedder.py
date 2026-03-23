import logging
from typing import Dict, Any, List

from langchain_core.documents import Document
from app.db.vector_store import VectorStore

logger = logging.getLogger(__name__)

class Embedder:
    """
    Business logic for embedding processed articles.
    Takes clean JSON dictionaries, converts them to LangChain Documents,
    and coordinates with the VectorStore to embed them.
    """

    def __init__(self):
        self.vector_store = VectorStore()

    def ingest_article(self, article_data: Dict[str, Any]) -> bool:
        """
        Takes a processed JSON object (containing index_a and index_b keys)
        and attempts to embed it. 
        Checks for existing PMIDs to avoid duplication.
        """
        # 1. Parse JSON into LangChain Documents
        index_a_docs = self._parse_to_documents(article_data.get("index_a", []))
        index_b_docs = self._parse_to_documents(article_data.get("index_b", []))

        if not index_a_docs and not index_b_docs:
            logger.warning("No valid chunks found in article_data.")
            return False

        # 2. Check for duplicate ingestion
        # We assume all chunks from one article share the same PMID
        pmid = None
        if index_a_docs:
            pmid = index_a_docs[0].metadata.get("pmid")
        elif index_b_docs:
            pmid = index_b_docs[0].metadata.get("pmid")

        if pmid:
            existing = self.vector_store.fetch_abstracts_by_pmid(pmid)
            if existing:
                logger.info(f"PMID {pmid} already embedded in Index A. Skipping ingestion.")
                return True # Technically a success, just skipped

        # 3. Add to Vector Store
        try:
            if index_a_docs:
                self.vector_store.add_abstracts(index_a_docs)
            if index_b_docs:
                self.vector_store.add_body_chunks(index_b_docs)
            logger.info(f"Successfully embedded PMID: {pmid}")
            return True
        except Exception as e:
            logger.error(f"Failed to embed PMID {pmid}: {str(e)}")
            return False

    def _parse_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Helper to convert generic dict chunks back to LangChain Documents."""
        docs = []
        for chunk in chunks:
            if "page_content" in chunk and "metadata" in chunk:
                clean_meta = {}
                for k, v in chunk["metadata"].items():
                    # ChromaDB metadata constraints:
                    # - Cannot contain None
                    # - Cannot contain empty lists []
                    # - Cannot contain dictionaries {}
                    if v is None:
                        continue
                    elif isinstance(v, list):
                        if len(v) == 0:
                            continue
                        # If list is not empty, ensure all items are primitives
                        clean_meta[k] = [str(item) for item in v]
                    elif isinstance(v, dict):
                        clean_meta[k] = str(v)
                    else:
                        clean_meta[k] = v

                docs.append(Document(
                    page_content=chunk["page_content"],
                    metadata=clean_meta
                ))
        return docs
