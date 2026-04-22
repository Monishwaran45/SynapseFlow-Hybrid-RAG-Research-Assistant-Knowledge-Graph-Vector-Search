"""
SynapseFlow — Hybrid Retriever
Combines vector similarity search with knowledge graph traversal
for comprehensive, multi-signal document retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.services.embeddings import embed_query
from backend.db.vector_store import VectorStore
from backend.db.graph_store import GraphStore


class HybridRetriever:
    """
    Hybrid retrieval engine that fuses:
    1. Dense vector search (semantic similarity via FAISS)
    2. Knowledge graph traversal (entity-relation lookup via NetworkX)
    """
    
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    def search(self, query: str, k: int = 5) -> dict:
        """
        Perform hybrid search combining vector and graph retrieval.
        
        Args:
            query: User's search query
            k: Number of vector results to return
        
        Returns:
            dict with 'vector_results' and 'graph_results'
        """
        # ── Vector Search ─────────────────────────────────────
        q_emb = embed_query(query)
        vector_results = self.vector_store.search(q_emb, k=k)
        
        # ── Graph Search ──────────────────────────────────────
        graph_results = []
        
        # Search the full query
        graph_results.extend(self.graph_store.search_keyword(query))
        
        # Also search individual significant words (>3 chars)
        words = [w.strip(".,!?;:\"'()[]") for w in query.split() if len(w) > 3]
        for word in words[:5]:
            graph_results.extend(self.graph_store.search_keyword(word))
        
        # ── Deduplicate Graph Results ─────────────────────────
        seen = set()
        unique_graph = []
        for r in graph_results:
            key = f"{r['source']}|{r['relation']}|{r['target']}"
            if key not in seen:
                seen.add(key)
                unique_graph.append(r)
        
        return {
            "vector_results": vector_results,
            "graph_results": unique_graph[:10]
        }
    
    def search_vectors_only(self, query: str, k: int = 5) -> list:
        """Vector-only search for simple queries."""
        q_emb = embed_query(query)
        return self.vector_store.search(q_emb, k=k)
    
    def search_graph_only(self, entity: str) -> list:
        """Graph-only search for entity exploration."""
        return self.graph_store.query(entity)
