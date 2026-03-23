import os
import sys
import logging
from uuid import uuid4

# Add the project root to the python path
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, main_dir)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVectorParams, Modifier, SparseVector
import chromadb

from app.utils.config import settings
from langchain_qdrant import FastEmbedSparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("migration")

def migrate_collection(chroma_name: str, qdrant_name: str, qdrant_client: QdrantClient):
    """Migrates a single Chroma collection to Qdrant synchronously by bypassing LangChain to preserve raw embeddings."""
    logger.info(f"--- Fast-Migrating {chroma_name} to {qdrant_name} ---")
    
    # 1. Connect natively to local ChromaDB (Bypassing LangChain)
    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    
    try:
        collection = chroma_client.get_collection(name=chroma_name)
    except Exception as e:
        logger.warning(f"Chroma collection '{chroma_name}' not found. Skipping.")
        return
        
    logger.info("Extracting raw data and computed embeddings from local Chroma collection...")
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    
    ids = all_data.get("ids", [])
    if not ids:
        logger.warning(f"No existing data found in local Chroma collection '{chroma_name}'. Skipping.")
        return
        
    texts = all_data["documents"]
    metadatas = all_data["metadatas"]
    vectors = all_data["embeddings"]
    
    total_docs = len(ids)
    logger.info(f"Found {total_docs} documents with pre-computed embeddings in '{chroma_name}'.")
    
    # Initialize FastEmbedSparse to compute sparse vectors during migration
    logger.info("Initializing FastEmbedSparse model (Qdrant/bm25)...")
    sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # 2. Prepare or Re-create Qdrant Collection
    try:
        if qdrant_client.collection_exists(collection_name=qdrant_name):
            logger.info(f"Qdrant collection {qdrant_name} already exists. Deleting it to start fresh...")
            qdrant_client.delete_collection(collection_name=qdrant_name)
            
        logger.info(f"Creating fresh Qdrant collection: {qdrant_name}")
        qdrant_client.create_collection(
            collection_name=qdrant_name,
            vectors_config={"": VectorParams(size=1536, distance=Distance.COSINE)},
            sparse_vectors_config={"langchain-sparse": SparseVectorParams(modifier=Modifier.IDF)}
        )
        
        # Extremely Important for LangChain Metadata Filtering:
        # We must explicitly tell Qdrant to build a fast-lookup index for our specific metadata fields
        # because our Retriever uses exact MatchValue and MatchAny filters on them.
        from qdrant_client.http.models import PayloadSchemaType
        qdrant_client.create_payload_index(
            collection_name=qdrant_name,
            field_name="metadata.pmid",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True
        )
        logger.info(f"Created Keyboard Payload Index on 'metadata.pmid' for {qdrant_name}")
        
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {e}")
        return

    # 3. Upload to Qdrant Fast using native PointStructs
    logger.info(f"Pushing {total_docs} vectors natively to Qdrant Cloud...")
    
    batch_size = 200
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        
        # Calculate sparse embeddings for the batch
        batch_texts = texts[i:end_idx]
        sparse_batch = sparse_model.embed_documents(batch_texts)
        
        points = []
        for j in range(i, end_idx):
            # LangChain exclusively looks for metadata inside a nested 'metadata' dictionary
            # And expects the main text to live at 'page_content'.
            payload = {
                "page_content": texts[j],
                "metadata": metadatas[j].copy() if metadatas[j] else {}
            }
            
            sparse_vec = sparse_batch[j - i]
            
            points.append(
                PointStruct(
                    id=str(uuid4()), 
                    vector={
                        "": vectors[j],
                        "langchain-sparse": SparseVector(indices=sparse_vec.indices, values=sparse_vec.values)
                    }, 
                    payload=payload
                )
            )
            
        # Add the batch synchronously
        qdrant_client.upload_points(
            collection_name=qdrant_name,
            points=points,
            wait=True
        )
        logger.info(f"  -> Uploaded chunk {i} to {end_idx} of {total_docs}")

    logger.info(f"✅ Successfully fast-migrated '{chroma_name}' to '{qdrant_name}'!")

def run_migration():
    """Main migration coordinator."""
    logger.info("=========================================")
    logger.info("Starting ChromaDB -> Qdrant Cloud Migration")
    logger.info("=========================================")
    
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )
    
    # Migrate Index A (Abstracts)
    migrate_collection(
        chroma_name="index_a_abstracts",
        qdrant_name="aura_index_a_abstracts",
        qdrant_client=qdrant_client
    )
    
    # Migrate Index B (Body Text)
    migrate_collection(
        chroma_name="index_b_bodies",
        qdrant_name="aura_index_b_bodies",
        qdrant_client=qdrant_client
    )
    
    logger.info("=========================================")
    logger.info("Migration Complete! The cloud database is now locked and loaded.")
    logger.info("=========================================")

if __name__ == "__main__":
    run_migration()
