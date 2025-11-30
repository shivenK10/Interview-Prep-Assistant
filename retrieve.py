import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from logger import Logger

log = Logger("Retrieval Logs", log_file_needed=True, log_file="Logs/retrieval.log")

QDRANT_PATH = Path("vector_stores/qdrant")
COLLECTION_NAME = "invoices"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

class Retriever:
    def __init__(self):
        """Initialize retriever with MPS support if available."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"Initializing retriever on device: {device}")
        
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.model = SentenceTransformer(EMBED_MODEL, device=device)
        log.info("Retriever initialized successfully")
    
    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Retrieve top_k most relevant chunks for the given query.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of dicts with 'text', 'source', 'page', 'chunk_id', and 'score'
        """
        log.info(f"Retrieving for query: {query[:100]}...")
        
        # Embed the query
        query_embedding = self.model.encode(query).tolist()
        
        # Search in Qdrant
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        )
        
        # Format results
        retrieved = []
        for result in results.points:
            retrieved.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload["page"],
                "chunk_id": result.payload["chunk_id"],
                "score": result.score
            })
        
        log.info(f"Retrieved {len(retrieved)} chunks")
        return retrieved

if __name__ == "__main__":
    retriever = Retriever()
    test_query = "What is the OSI model?"
    results = retriever.retrieve(test_query)
    
    print(f"\nQuery: {test_query}\n")
    for idx, result in enumerate(results, 1):
        print(f"--- Result {idx} (Score: {result['score']:.4f}) ---")
        print(f"Source: {result['source']} | Page: {result['page']}")
        print(f"Text: {result['text'][:200]}...\n")
