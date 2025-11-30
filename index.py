import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from logger import Logger

log = Logger("Indexing Logs", log_file_needed=True, log_file="Logs/indexing.log")

CHUNKS_PATH = Path("embeddings/chunks_with_embeddings.json")
COLLECTION_NAME = "invoices"
QDRANT_PATH = Path("vector_stores/qdrant")
DIM = 384

def index_data():
    """Index chunks with embeddings into Qdrant."""
    log.info("Loading chunks with embeddings...")
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    log.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize Qdrant client (local storage)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(COLLECTION_NAME)
        log.info(f"Collection '{COLLECTION_NAME}' already exists, deleting...")
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    log.info(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    
    # Prepare points for insertion
    points = []
    for idx, chunk in enumerate(chunks):
        point = PointStruct(
            id=idx,
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            }
        )
        points.append(point)
    
    # Insert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        log.info(f"Inserted batch {i//batch_size+1}/{(len(points)-1)//batch_size+1}")
    
    log.info(f"Successfully indexed {len(points)} chunks into Qdrant")

if __name__ == "__main__":
    index_data()
