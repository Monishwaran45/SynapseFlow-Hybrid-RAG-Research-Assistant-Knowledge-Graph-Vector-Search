"""
SynapseFlow — Configuration
Central configuration for the Smart Research Assistant.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Project Paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indices"
GRAPH_DIR = DATA_DIR / "graphs"

# Create directories
for _dir in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, GRAPH_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ─────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ─── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ─── LLM Configuration ───────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini-latest")  # gemini | groq
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─── Storage Paths ────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
GRAPH_PATH = GRAPH_DIR / "knowledge_graph.gml"
DOC_REGISTRY_PATH = DATA_DIR / "doc_registry.json"

# ─── App Info ─────────────────────────────────────────────────
APP_NAME = "SynapseFlow"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Knowledge Graph + RAG Hybrid Intelligence"
