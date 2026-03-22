"""
Creates the required Qdrant collections for the weightloss RAG system.
Run this once before embed_to_qdrant.py.
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams, Modifier, PayloadSchemaType
)
from app.utils.config import settings
from app.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

COLLECTIONS = {
    "aura_index_a_abstracts": "Index A — abstract-level documents",
    "aura_index_b_bodies":    "Index B — full-text body chunks",
}

DENSE_VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


def create_collection(client: QdrantClient, name: str):
    if client.collection_exists(name):
        logger.info(f"Collection '{name}' already exists, skipping.")
        return

    client.create_collection(
        collection_name=name,
        vectors_config={"": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={
            "langchain-sparse": SparseVectorParams(modifier=Modifier.IDF)
        }
    )
    logger.info(f"Created collection '{name}'.")

    # Payload index on pmid for fast duplicate checks
    client.create_payload_index(
        collection_name=name,
        field_name="metadata.pmid",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True
    )
    logger.info(f"Created payload index on metadata.pmid for '{name}'.")


def main():
    logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    for name, desc in COLLECTIONS.items():
        logger.info(f"Setting up: {name} ({desc})")
        create_collection(client, name)

    logger.info("All collections ready.")


if __name__ == "__main__":
    main()
