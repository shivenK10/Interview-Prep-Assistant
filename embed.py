import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logger import Logger

log = Logger("Embeddings Logs", log_file_needed=True, log_file='Logs/embeddings.log')

CLEAN_CHUNKS_DIR = Path("Clean Interview Prep Chunks")    # Path where the cleaned chunks are stored
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 64
OUT_PATH    = Path("embeddings/chunks_with_embeddings.json")

def load_chunks():
    """Load all chunks from JSON files."""
    all_chunks = []
    for chunks_file in sorted(CLEAN_CHUNKS_DIR.glob("*_chunks.json")):
        log.info(f"Loading chunks from {chunks_file.name}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks

def embed_data():
    """Embed all chunks and save with metadata."""
    chunks = load_chunks()
    log.info(f"Loaded {len(chunks)} chunks total")
    
    model = SentenceTransformer(EMBED_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        log.info(f"Embedding batch {i//BATCH_SIZE+1}/{(len(texts)-1)//BATCH_SIZE+1}")
        embs = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
    
    # Combine chunks with embeddings
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding
    
    # Save as JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    log.info(f"Saved {len(chunks)} chunks with embeddings to {OUT_PATH}")

if __name__ == "__main__":
    embed_data()
