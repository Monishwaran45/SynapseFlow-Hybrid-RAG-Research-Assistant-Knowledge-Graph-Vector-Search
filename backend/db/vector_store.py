"""
SynapseFlow — Vector Store
FAISS-based vector database for semantic similarity search.
Uses IndexFlatIP (inner product) with normalized embeddings for cosine similarity.
"""

import json
import faiss
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import EMBEDDING_DIM, FAISS_INDEX_PATH, METADATA_PATH


class VectorStore:
    """
    Persistent FAISS vector store with metadata tracking.
    Stores chunk embeddings and retrieves by cosine similarity.
    """
    
    def __init__(self):
        self.index = None
        self.metadata = []  # Parallel list of chunk metadata
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create a fresh one."""
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception:
                self._create_fresh()
        else:
            self._create_fresh()
    
    def _create_fresh(self):
        """Create a new empty FAISS index."""
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine sim with normalized vectors
        self.metadata = []
    
    def add(self, embeddings: np.ndarray, chunks: list):
        """
        Add embeddings and their metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, 384)
            chunks: list of chunk metadata dicts
        """
        if len(embeddings) == 0:
            return
        
        self.index.add(np.array(embeddings, dtype='float32'))
        self.metadata.extend(chunks)
        self._save()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """
        Search for the k most similar vectors.
        
        Args:
            query_embedding: numpy array of shape (1, 384)
            k: number of results
        
        Returns:
            List of chunk metadata dicts with added 'score' field
        """
        if self.index.ntotal == 0:
            return []
        
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(
            np.array(query_embedding, dtype='float32'), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(scores[0][i])
                results.append(result)
        
        return results
    
    def _save(self):
        """Persist index and metadata to disk."""
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    @property
    def total_vectors(self) -> int:
        """Total number of vectors in the index."""
        return self.index.ntotal if self.index else 0
    
    def get_doc_ids(self) -> list:
        """Get unique document IDs in the store."""
        return list(set(m.get("doc_id", "") for m in self.metadata))
    
    def clear(self):
        """Reset the store — deletes all vectors and metadata."""
        self._create_fresh()
        self._save()
