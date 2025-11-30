import re
import json
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logger import Logger

log = Logger("Data Cleaning Logs", log_file_needed=True, log_file='Logs/data_cleaning.log')

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

TXT_DIR   = Path("Interview Prep TXTs")  # Path where the extracted/raw text files are stored
CLEAN_TXT_DIR = Path("Clean Interview Prep TXTs")    # Path where the clean text files are stored
CHUNKS_DIR = Path("Interview Prep Chunks")  # Path where the chunked data is stored
CLEAN_CHUNKS_DIR = Path("Clean Interview Prep Chunks")  # Path where cleaned chunks will be stored

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    The following steps will be followed:
    - convert the entire data into lower case.
    - remove all non-ASCII characters.
    - preserves a specially defined set of characters.
    - tokenization.
    - removal of stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]+', ' ', text)
    text = re.sub(r'[^a-z0-9\.\s#/@(),:;\-\+%A-Z&]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def clean_data():
    # Clean full text files
    for txt_file in TXT_DIR.glob("*.txt"):
        log.info(f"Cleaning {txt_file.name}")
        raw      = txt_file.read_text(encoding="utf-8")
        cleaned  = clean_text(raw)
        out_path = CLEAN_TXT_DIR / txt_file.name
        CLEAN_TXT_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
        log.info(f"Saved cleaned text to {out_path.name}")
    
    # Clean chunked data
    for chunks_file in CHUNKS_DIR.glob("*_chunks.json"):
        log.info(f"Cleaning chunks from {chunks_file.name}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        cleaned_chunks = []
        for chunk in chunks_data:
            cleaned_text = clean_text(chunk["text"])
            if len(cleaned_text.strip()) > 20:  # Skip very short chunks
                cleaned_chunks.append({
                    "text": cleaned_text,
                    "metadata": chunk["metadata"]
                })
        
        out_path = CLEAN_CHUNKS_DIR / chunks_file.name
        CLEAN_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_chunks, f, indent=2, ensure_ascii=False)
        log.info(f"Saved {len(cleaned_chunks)} cleaned chunks to {out_path.name}")

    log.info("Data cleaning complete.")

if __name__ == "__main__":
    clean_data()
