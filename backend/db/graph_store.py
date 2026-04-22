"""
SynapseFlow — Graph Store
NetworkX-based knowledge graph storage with persistence.
Lightweight alternative to Neo4j — no external database needed.
"""

import networkx as nx
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import GRAPH_PATH


class GraphStore:
    """
    Persistent knowledge graph using NetworkX DiGraph.
    Stores entities as nodes and relationships as directed edges.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load()
    
    def _load(self):
        """Load graph from disk if it exists."""
        if GRAPH_PATH.exists():
            try:
                self.graph = nx.read_gml(str(GRAPH_PATH))
            except Exception:
                self.graph = nx.DiGraph()
    
    def _save(self):
        """Persist graph to GML file."""
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gml(self.graph, str(GRAPH_PATH))
    
    # ─── Write Operations ─────────────────────────────────────
    
    def add_relation(self, entity1: str, relation: str, entity2: str, doc_id: str = ""):
        """Add a single entity-relation-entity triple."""
        e1 = entity1.strip().lower()
        e2 = entity2.strip().lower()
        
        if not e1 or not e2 or e1 == e2:
            return
        
        # Add/update nodes with display labels
        self.graph.add_node(e1, label=entity1.strip(), doc_id=doc_id)
        self.graph.add_node(e2, label=entity2.strip(), doc_id=doc_id)
        
        # Add edge with relation type
        self.graph.add_edge(e1, e2, relation=relation.strip(), doc_id=doc_id)
    
    def add_relations_batch(self, relations: list, doc_id: str = ""):
        """
        Add multiple relations at once.
        
        Args:
            relations: List of [entity1, relation, entity2] triples
            doc_id: Source document identifier
        """
        for rel in relations:
            if isinstance(rel, (list, tuple)) and len(rel) == 3:
                self.add_relation(rel[0], rel[1], rel[2], doc_id)
        self._save()
    
    # ─── Read Operations ──────────────────────────────────────
    
    def query(self, entity: str, depth: int = 1) -> list:
        """
        Find all relations connected to an entity.
        
        Args:
            entity: Entity name to search for
            depth: Search depth (currently 1-hop)
        
        Returns:
            List of relation dicts
        """
        entity_lower = entity.strip().lower()
        results = []
        
        # Find matching nodes (exact and partial match)
        matching = [n for n in self.graph.nodes() if entity_lower in n]
        
        for node in matching:
            # Outgoing edges
            for _, target, data in self.graph.out_edges(node, data=True):
                results.append({
                    "source": self.graph.nodes[node].get("label", node),
                    "relation": data.get("relation", "related_to"),
                    "target": self.graph.nodes[target].get("label", target),
                    "doc_id": data.get("doc_id", ""),
                    "direction": "outgoing"
                })
            
            # Incoming edges
            for source, _, data in self.graph.in_edges(node, data=True):
                results.append({
                    "source": self.graph.nodes[source].get("label", source),
                    "relation": data.get("relation", "related_to"),
                    "target": self.graph.nodes[node].get("label", node),
                    "doc_id": data.get("doc_id", ""),
                    "direction": "incoming"
                })
        
        return results
    
    def search_keyword(self, keyword: str) -> list:
        """
        Search for any relation containing the keyword.
        
        Args:
            keyword: Search term
        
        Returns:
            List of matching relation dicts
        """
        keyword_lower = keyword.strip().lower()
        results = []
        
        if not keyword_lower:
            return results
        
        for source, target, data in self.graph.edges(data=True):
            relation = data.get("relation", "")
            if (keyword_lower in source or 
                keyword_lower in target or 
                keyword_lower in relation.lower()):
                results.append({
                    "source": self.graph.nodes[source].get("label", source),
                    "relation": relation,
                    "target": self.graph.nodes[target].get("label", target),
                    "doc_id": data.get("doc_id", "")
                })
        
        return results
    
    def get_all_relations(self) -> list:
        """Get every relation in the graph."""
        results = []
        for source, target, data in self.graph.edges(data=True):
            results.append({
                "source": self.graph.nodes[source].get("label", source),
                "relation": data.get("relation", ""),
                "target": self.graph.nodes[target].get("label", target),
                "doc_id": data.get("doc_id", "")
            })
        return results
    
    def get_entities(self) -> list:
        """Get all unique entities with their connection counts."""
        entities = []
        for node in self.graph.nodes():
            entities.append({
                "name": self.graph.nodes[node].get("label", node),
                "connections": self.graph.degree(node),
                "doc_id": self.graph.nodes[node].get("doc_id", "")
            })
        return sorted(entities, key=lambda x: x["connections"], reverse=True)
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "components": (
                nx.number_weakly_connected_components(self.graph)
                if self.graph.number_of_nodes() > 0 else 0
            )
        }
    
    def clear(self):
        """Reset the graph — removes all nodes and edges."""
        self.graph = nx.DiGraph()
        self._save()
