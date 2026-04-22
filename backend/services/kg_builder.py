"""
SynapseFlow — Knowledge Graph Builder
Coordinates LLM-based entity/relation extraction and graph storage.
"""

from backend.services.llm_service import LLMService
from backend.db.graph_store import GraphStore


class KGBuilder:
    """
    Orchestrates knowledge graph construction from text chunks.
    Uses the LLM service for entity/relation extraction and
    the graph store for persistence.
    """
    
    def __init__(self, llm_service: LLMService, graph_store: GraphStore):
        self.llm = llm_service
        self.graph = graph_store
    
    def process_chunks(self, chunks: list, doc_id: str = "") -> dict:
        """
        Process a list of text chunks to extract and store KG relations.
        
        Args:
            chunks: List of chunk dicts with 'text' field
            doc_id: Document identifier
        
        Returns:
            dict with extraction statistics
        """
        all_relations = []
        errors = 0
        
        for chunk in chunks:
            try:
                text = chunk.get("text", "")
                if len(text) < 50:  # Skip very short chunks
                    continue
                
                relations = self.llm.extract_entities_relations(text)
                if relations:
                    self.graph.add_relations_batch(relations, doc_id)
                    all_relations.extend(relations)
            except Exception:
                errors += 1
        
        return {
            "total_relations": len(all_relations),
            "relations": all_relations,
            "errors": errors,
            "chunks_processed": len(chunks)
        }
    
    def process_single(self, text: str, doc_id: str = "") -> list:
        """Process a single text block."""
        try:
            relations = self.llm.extract_entities_relations(text)
            if relations:
                self.graph.add_relations_batch(relations, doc_id)
            return relations
        except Exception:
            return []
