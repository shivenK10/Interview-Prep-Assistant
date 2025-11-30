# Interview Prep Assistant

A RAG (Retrieval-Augmented Generation) system for interview preparation using Llama 3.2 3B, Qdrant vector database, and sentence transformers.

## Features

- PDF ingestion with OCR fallback
- Smart text chunking with metadata preservation
- Vector search using Qdrant
- LLM-powered answer generation
- Terminal chat interface with logging

## Architecture

```
PDFs → Ingestion → Cleaning → Embedding → Qdrant → Retrieval → LLM → Answer
```

## Installation

```bash
# Install Python dependencies
pip install torch transformers sentence-transformers langchain langchain-community langchain-huggingface langchain-core qdrant-client pypdf pdf2image pytesseract nltk

# Install system dependencies (macOS)
brew install tesseract poppler
```

## Pipeline

### 1. Ingest PDFs
```bash
python ingest.py
```
- Extracts text from PDFs (digital + OCR)
- Creates chunks with page-level metadata

### 2. Clean Data
```bash
python data_cleaning.py
```
- Removes stopwords and normalizes text
- Cleans both full texts and chunks

### 3. Generate Embeddings
```bash
python embed.py
```
- Embeds chunks using `all-MiniLM-L6-v2`
- Outputs to `embeddings/chunks_with_embeddings.json`

### 4. Index in Qdrant
```bash
python index.py
```
- Creates Qdrant vector database
- Stores in `vector_stores/qdrant/`

### 5. Chat
```bash
python main.py
```
- Interactive chat interface
- Retrieves context and generates answers
- Logs to `Logs/main.log`

## Configuration

Update paths in each file:
- `ingest.py`: Set `PDF_DIR`, `TXT_DIR`, `CHUNKS_DIR`
- `data_cleaning.py`: Set `TXT_DIR`, `CLEAN_TXT_DIR`, `CHUNKS_DIR`, `CLEAN_CHUNKS_DIR`
- `embed.py`: Set `CLEAN_CHUNKS_DIR`
- `model_handler.py`: Set model name if different

## Usage

```bash
# Run chat
python main.py

# Example interaction
You: What is the OSI model?
Assistant: [Generated answer based on your materials]
```

Type `quit` to exit.

## Project Structure

```
.
├── ingest.py                  # PDF ingestion & chunking
├── data_cleaning.py           # Text cleaning
├── embed.py                   # Generate embeddings
├── index.py                   # Qdrant indexing
├── retrieve.py                # Retrieval logic
├── model_handler.py           # LLM model loader
├── generation_pipeline.py     # RAG pipeline
├── main.py                    # Chat interface
├── logger.py                  # Logging utility
├── Logs/                      # Log files
├── embeddings/                # Embeddings cache
└── vector_stores/qdrant/      # Vector database
└── raw_data/                  # Raw data (pdfs/screenshots)
```

## Logging

All operations are logged to `Logs/` directory with timestamps and rotation (5MB max).

## Requirements

- Python 3.8+
- macOS/Linux/Windows (MPS/CUDA support for faster inference)
- 8GB+ RAM for model loading
