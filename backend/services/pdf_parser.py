"""
SynapseFlow — PDF Parser
Extracts text and metadata from PDF documents using PyMuPDF.
"""

import fitz  # PyMuPDF
from pathlib import Path


def extract_text(file_path: str) -> dict:
    """
    Extract text from a PDF file with per-page breakdown and metadata.
    
    Returns:
        dict with keys: text, pages, metadata
    """
    file_path = str(file_path)
    doc = fitz.open(file_path)
    
    pages = []
    full_text = ""
    
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text,
            "char_count": len(text)
        })
        full_text += text + "\n"
    
    metadata = {
        "title": doc.metadata.get("title", "") or Path(file_path).stem,
        "author": doc.metadata.get("author", "") or "Unknown",
        "subject": doc.metadata.get("subject", ""),
        "page_count": len(doc),
        "total_chars": len(full_text)
    }
    
    doc.close()
    
    return {
        "text": full_text,
        "pages": pages,
        "metadata": metadata
    }


def extract_text_by_pages(file_path: str, start_page: int = 0, end_page: int = None) -> str:
    """Extract text from specific page range."""
    doc = fitz.open(str(file_path))
    end = end_page or len(doc)
    
    text = ""
    for i in range(start_page, min(end, len(doc))):
        text += doc[i].get_text("text") + "\n"
    
    doc.close()
    return text
