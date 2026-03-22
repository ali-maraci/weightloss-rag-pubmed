import logging
from typing import List

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.utils.config import settings

logger = logging.getLogger(__name__)

class AuraVectorStore:
    """Database wrapper for Qdrant Cloud. Handles embedding and retrieval
    from remote HTTP collections, abstracting database-specific logic."""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.OPENAI_API_KEY
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        self.collection_a = self._init_collection("aura_index_a_abstracts")
        self.collection_b = self._init_collection("aura_index_b_bodies")
        
        logger.info(f"AuraVectorStore initialized pointing to Qdrant Cloud at {settings.QDRANT_URL}")

    def _init_collection(self, collection_name: str) -> QdrantVectorStore:
        """Connects to a specific remote Qdrant collection."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            content_payload_key="page_content"
        )

    def add_abstracts(self, documents: List[Document]) -> List[str]:
        """Adds a batch of abstract documents to Index A."""
        if not documents:
            return []
        ids = self.collection_a.add_documents(documents)
        logger.info(f"Added {len(ids)} documents to Index A.")
        return ids

    def add_body_chunks(self, documents: List[Document]) -> List[str]:
        """Adds a batch of body chunks to Index B."""
        if not documents:
            return []
        ids = self.collection_b.add_documents(documents)
        logger.info(f"Added {len(ids)} documents to Index B.")
        return ids
    
    def fetch_abstracts_by_pmid(self, pmid: str) -> List[Document]:
        """Checks if a PMID exists in Index A and returns matching documents."""
        from qdrant_client.http import models
        results, _ = self.client.scroll(
            collection_name="aura_index_a_abstracts",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.pmid",
                        match=models.MatchValue(value=pmid)
                    )
                ]
            ),
            limit=10,
            with_payload=True
        )
        
        if not results:
            return []

        docs = []
        for point in results:
            content = point.payload.get("page_content", "")
            meta = point.payload.get("metadata", {})
            docs.append(Document(page_content=content, metadata=meta))

        return docs
