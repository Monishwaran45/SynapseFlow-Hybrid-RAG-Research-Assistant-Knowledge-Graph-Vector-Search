"""
SynapseFlow — Embeddings Service
Generates semantic embeddings using sentence-transformers.
Uses all-MiniLM-L6-v2 (384-dim) for fast, high-quality embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import EMBEDDING_MODEL

# ─── Lazy Model Loading ──────────────────────────────────────
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (first call takes a few seconds)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed
    
    Returns:
        numpy array of shape (len(texts), 384)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32
    )
    return np.array(embeddings, dtype='float32')


def embed_query(query: str) -> np.ndarray:
    """
    Generate a normalized embedding for a single query.
    
    Args:
        query: Search query string
    
    Returns:
        numpy array of shape (1, 384)
    """
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return np.array(embedding, dtype='float32')
