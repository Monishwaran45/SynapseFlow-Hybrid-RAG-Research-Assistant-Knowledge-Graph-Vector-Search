"""
SynapseFlow — FastAPI Server
REST API layer for the Smart Research Assistant.
Run: uvicorn backend.server:app --reload
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import config
from backend.services.pdf_parser import extract_text
from backend.services.chunking import chunk_text
from backend.services.embeddings import embed_texts
from backend.services.llm_service import LLMService
from backend.db.vector_store import VectorStore
from backend.db.graph_store import GraphStore
from backend.services.retriever import HybridRetriever

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="SynapseFlow API",
    description="Knowledge Graph + RAG Hybrid Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singletons ──────────────────────────────────────────────
vector_store = VectorStore()
graph_store = GraphStore()
retriever = HybridRetriever(vector_store, graph_store)
llm = None


def get_llm():
    global llm
    if llm is None:
        llm = LLMService()
    return llm


# ─── Request/Response Models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list
    graph_relations: list


class StatusResponse(BaseModel):
    status: str
    documents: int
    vectors: int
    kg_nodes: int
    kg_edges: int


# ─── Routes ──────────────────────────────────────────────────

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint."""
    return {
        "name": "SynapseFlow API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics."""
    stats = graph_store.get_stats()
    import json
    doc_count = 0
    if config.DOC_REGISTRY_PATH.exists():
        with open(config.DOC_REGISTRY_PATH, 'r') as f:
            doc_count = len(json.load(f))
    
    return StatusResponse(
        status="operational",
        documents=doc_count,
        vectors=vector_store.total_vectors,
        kg_nodes=stats["nodes"],
        kg_edges=stats["edges"]
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    Extracts text, generates embeddings, and builds knowledge graph.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Save uploaded file
    save_path = config.UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        doc_id = Path(file.filename).stem.replace(' ', '_').replace('-', '_')
        
        # Extract text
        result = extract_text(str(save_path))
        
        # Chunk
        chunks = chunk_text(
            result["text"],
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            doc_id=doc_id
        )
        
        # Embed & store
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        vector_store.add(embeddings, chunks)
        
        # Build KG
        llm_service = get_llm()
        kg_count = 0
        for chunk in chunks[:15]:
            try:
                relations = llm_service.extract_entities_relations(chunk["text"])
                if relations:
                    graph_store.add_relations_batch(relations, doc_id)
                    kg_count += len(relations)
            except Exception:
                pass
        
        # Update registry
        import json
        registry = []
        if config.DOC_REGISTRY_PATH.exists():
            with open(config.DOC_REGISTRY_PATH, 'r') as f:
                registry = json.load(f)
        
        registry = [d for d in registry if d.get("doc_id") != doc_id]
        registry.append({
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": result["metadata"]["page_count"],
            "chunks": len(chunks),
            "kg_relations": kg_count,
            "title": result["metadata"].get("title", doc_id),
            "author": result["metadata"].get("author", "Unknown"),
            "chars": result["metadata"]["total_chars"]
        })
        
        with open(config.DOC_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return {
            "status": "processed",
            "doc_id": doc_id,
            "pages": result["metadata"]["page_count"],
            "chunks": len(chunks),
            "kg_relations": kg_count
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a research question.
    Uses hybrid vector + graph retrieval and LLM generation.
    """
    if vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Upload a PDF first."
        )
    
    # Hybrid search
    results = retriever.search(req.question, k=req.k)
    
    # Build context
    text_context = "\n\n---\n\n".join(
        [r["text"] for r in results["vector_results"]]
    )
    graph_context = "\n".join([
        f"{r['source']} → [{r['relation']}] → {r['target']}"
        for r in results["graph_results"]
    ])
    
    # Generate answer
    llm_service = get_llm()
    answer = llm_service.generate_answer(req.question, text_context, graph_context)
    
    return QueryResponse(
        answer=answer,
        sources=[
            {"chunk_id": r["id"], "score": r["score"], "preview": r["text"][:150]}
            for r in results["vector_results"]
        ],
        graph_relations=results["graph_results"]
    )


@app.get("/graph")
async def get_graph():
    """Get all knowledge graph data."""
    return {
        "stats": graph_store.get_stats(),
        "entities": graph_store.get_entities()[:50],
        "relations": graph_store.get_all_relations()[:100]
    }


@app.get("/documents")
async def get_documents():
    """Get all processed documents."""
    import json
    if config.DOC_REGISTRY_PATH.exists():
        with open(config.DOC_REGISTRY_PATH, 'r') as f:
            return {"documents": json.load(f)}
    return {"documents": []}


@app.delete("/clear")
async def clear_all():
    """Clear all stored data."""
    vector_store.clear()
    graph_store.clear()
    
    import json
    with open(config.DOC_REGISTRY_PATH, 'w') as f:
        json.dump([], f)
    
    return {"status": "cleared"}
