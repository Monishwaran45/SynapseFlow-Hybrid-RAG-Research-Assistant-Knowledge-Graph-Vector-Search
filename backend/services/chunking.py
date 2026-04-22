"""
SynapseFlow — Text Chunking
Sentence-aware chunking with overlap for optimal retrieval.
"""

import re
from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    doc_id: str = ""
) -> List[Dict]:
    """
    Split text into overlapping chunks with sentence-boundary awareness.
    
    Args:
        text: Full document text
        chunk_size: Target characters per chunk
        overlap: Number of overlap characters between chunks
        doc_id: Document identifier for tracking
    
    Returns:
        List of chunk dictionaries with id, text, doc_id, chunk_index, char_count
    """
    # Clean text
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_idx = 0
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk_size and we have content
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append({
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "text": current_chunk.strip(),
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "char_count": len(current_chunk.strip())
            })
            
            # Keep overlap — last N characters worth of words
            words = current_chunk.split()
            overlap_word_count = max(1, overlap // 5)
            overlap_words = words[-overlap_word_count:] if len(words) > overlap_word_count else words
            current_chunk = " ".join(overlap_words) + " " + sentence
            chunk_idx += 1
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "id": f"{doc_id}_chunk_{chunk_idx}",
            "text": current_chunk.strip(),
            "doc_id": doc_id,
            "chunk_index": chunk_idx,
            "char_count": len(current_chunk.strip())
        })
    
    return chunks
