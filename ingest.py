from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
import json
from logger import Logger

log = Logger("Ingestion Logs", log_file_needed=True, log_file="Logs/ingestion.log")

PDF_DIR = Path("Interview Prep PDFs/New")  # Path to all the PDFs
TXT_DIR = Path("Interview Prep TXTs")    # Path where the extracted/raw text are stored
CHUNKS_DIR = Path("Interview Prep Chunks")  # Path where chunked data with metadata will be stored

def ocr_page_images(pdf_path: Path):
    """
    Convert all pages of the PDF into images and run Tesseract OCR.
    Returns a list of strings, one per page.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=500
    )
    texts = []
    for img in images:
        txt = pytesseract.image_to_string(
            img,
            lang='eng', 
            config='--oem 3 --psm 6'
        )
        texts.append(txt)
    return texts

def extract_and_save():
    """
    For each PDF:
      1. Extract digital text via PyPDFLoader.
      2. If any page's text is empty, fallback to OCR for that page.
      3. Split text into chunks with metadata.
      4. Save both full text and chunked data.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for pdf_file in PDF_DIR.glob("*.pdf"):
        log.info(f"Processing {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        digital_pages = [doc.page_content or "" for doc in docs]

        ocr_pages = ocr_page_images(pdf_file)

        merged_pages = []
        for page_num, (d, o) in enumerate(zip(digital_pages, ocr_pages), 1):
            if len(d.strip()) < 20:
                merged_pages.append((page_num, o))
            else:
                merged_pages.append((page_num, d))

        # Save full text
        full_text = "\n".join([text for _, text in merged_pages])
        txt_path = TXT_DIR / f"{pdf_file.stem}.txt"
        TXT_DIR.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text, encoding="utf-8")
        log.info(f"Wrote text to {txt_path.name}")
        
        # Create chunks with metadata
        chunks_data = []
        for page_num, page_text in merged_pages:
            chunks = text_splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                chunks_data.append({
                    "text": chunk,
                    "metadata": {
                        "source": pdf_file.name,
                        "page": page_num,
                        "chunk_id": f"{pdf_file.stem}_p{page_num}_c{chunk_idx}"
                    }
                })
        
        # Save chunks
        chunks_path = CHUNKS_DIR / f"{pdf_file.stem}_chunks.json"
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        log.info(f"Wrote {len(chunks_data)} chunks to {chunks_path.name}")

    log.info("PDF ingestion with OCR complete.")

if __name__ == "__main__":
    extract_and_save()
