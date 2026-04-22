<p align="center">
  <h1 align="center">SynapseFlow</h1>
  <p align="center"><strong>Knowledge Graph + RAG Hybrid Research Intelligence System</strong></p>
  <p align="center">
    <em>Upload research papers. Ask questions. Get answers powered by dual retrieval — semantic vectors + knowledge graph reasoning.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FAISS-vector%20search-green?style=flat-square" />
  <img src="https://img.shields.io/badge/NetworkX-knowledge%20graph-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Gemini%20%7C%20Groq-LLM-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Rich-Terminal%20UI-yellow?style=flat-square" />
</p>

---

## Why SynapseFlow?

Most RAG systems retrieve chunks and throw them at an LLM. SynapseFlow goes further:

| Traditional RAG | SynapseFlow |
|---|---|
| PDF → chunks → embeddings → answer | PDF → chunks → embeddings **+ entity-relation extraction → knowledge graph** → hybrid answer |
| Single-signal retrieval | **Dual-signal**: vector similarity + graph traversal |
| Isolated document answers | **Cross-document reasoning** via shared KG entities |
| Flat context | **Structured context** with entity relationships |

---

## System Architecture

```
                          ┌─────────────────────┐
                          │     PDF Upload       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Text Extraction    │
                          │     (PyMuPDF)        │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Sentence-Aware      │
                          │  Chunking (overlap)  │
                          └─────┬──────────┬────┘
                                │          │
                   ┌────────────▼──┐  ┌────▼────────────┐
                   │  Embedding    │  │  LLM Entity &   │
                   │  Generation   │  │  Relation        │
                   │ (MiniLM-L6)   │  │  Extraction      │
                   └──────┬────────┘  └───────┬─────────┘
                          │                   │
                   ┌──────▼────────┐  ┌───────▼─────────┐
                   │  FAISS Index  │  │  NetworkX KG    │
                   │  (Vector DB)  │  │  (Graph DB)     │
                   └──────┬────────┘  └───────┬─────────┘
                          │                   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  User Query        │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Hybrid Retriever  │
                          │  vector + graph    │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  LLM Answer Gen    │
                          │  (Gemini / Groq)   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Structured Answer │
                          │  + Sources + KG    │
                          └───────────────────┘
```

### Data Flow Detail

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                            │
│                                                                      │
│  PDF ──► PyMuPDF ──► Raw Text ──► Chunker (500ch, 50 overlap)       │
│                                      │                               │
│                          ┌───────────┴───────────┐                   │
│                          │                       │                   │
│                   SentenceTransformer       Gemini/Groq LLM          │
│                   (all-MiniLM-L6-v2)       Entity Extraction         │
│                          │                       │                   │
│                    384-dim vectors          [E1, rel, E2] triples    │
│                          │                       │                   │
│                     FAISS Index            NetworkX DiGraph           │
│                    (cosine sim)            (GML persistence)         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                                │
│                                                                      │
│  Question ──► Embed Query ──► FAISS search (top-k chunks)           │
│          └──► Keyword match ──► KG traversal (entity relations)     │
│                                      │                               │
│                          ┌───────────┴───────────┐                   │
│                     Text Context            Graph Context            │
│                   (retrieved chunks)     (entity-relation triples)   │
│                          └───────────┬───────────┘                   │
│                                      │                               │
│                              LLM Generation                          │
│                         (combined context prompt)                    │
│                                      │                               │
│                              Structured Answer                       │
│                          + citations + KG relations                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

#### Option A — uv (recommended)

```bash
# Install uv if you don't have it
pip install uv
# or on macOS/Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh

cd smart-research-assistant

# Create venv and install dependencies in one step
uv venv
uv pip install -r requirements.txt

# Activate
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
```

#### Option B — standard pip

```bash
cd smart-research-assistant
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Option 1 — Groq (free tier, fast)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Option 2 — Gemini
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
```

### 3. Run Terminal UI

```bash
python app.py
```

### 4. Run REST API (optional)

```bash
uvicorn backend.server:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## Project Structure

```
smart-research-assistant/
│
├── app.py                      # Entry point — Rich terminal UI
├── config.py                   # Central configuration
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored)
├── .env.example                # Template
│
├── backend/
│   ├── server.py               # FastAPI REST API
│   │
│   ├── services/
│   │   ├── pdf_parser.py       # PDF → text (PyMuPDF)
│   │   ├── chunking.py         # Text → sentence-aware chunks
│   │   ├── embeddings.py       # Chunks → 384-dim vectors (MiniLM)
│   │   ├── llm_service.py      # LLM wrapper (Gemini / Groq)
│   │   ├── kg_builder.py       # Orchestrates KG extraction
│   │   └── retriever.py        # Hybrid vector + graph search
│   │
│   └── db/
│       ├── vector_store.py     # FAISS index + metadata persistence
│       └── graph_store.py      # NetworkX DiGraph + GML persistence
│
└── data/                       # Created at runtime
    ├── uploads/
    ├── indices/                # faiss.index + metadata.json
    └── graphs/                 # knowledge_graph.gml
```

---

## Features

### Core

| Feature | How It Works |
|---|---|
| **PDF Ingestion** | PyMuPDF extracts text → sentence-aware chunking with configurable size/overlap |
| **Semantic Search** | sentence-transformers (all-MiniLM-L6-v2) → 384-dim embeddings → FAISS inner-product index |
| **Knowledge Graph** | LLM extracts `[entity, relation, entity]` triples → stored in NetworkX DiGraph |
| **Hybrid Retrieval** | Query hits both FAISS (top-k similar chunks) and KG (keyword entity traversal) |
| **Answer Generation** | LLM receives merged text + graph context → structured answer with citations |

### Terminal UI

| Command | Description |
|---|---|
| `[1] Upload PDF` | Full pipeline: extract → chunk → embed → KG build with progress bar |
| `[2] Ask Question` | Interactive Q&A loop with answer + sources + graph relations |
| `[3] View Knowledge Graph` | Tree visualization of entities & relations + entity search |
| `[4] View Documents` | Table of all processed papers with stats |
| `[5] Extract Insights` | AI-powered analysis: contributions, methodology, limitations, future work |
| `[6] Clear Data` | Reset FAISS index, KG, and registry |

### REST API

| Method | Endpoint | Description |
|---|---|
| `GET` | `/` | Health check |
| `GET` | `/status` | System statistics |
| `POST` | `/upload` | Upload & process PDF (multipart) |
| `POST` | `/query` | Ask a question (JSON body: `{"question": "..."}`) |
| `GET` | `/graph` | Get KG entities & relations |
| `GET` | `/documents` | List processed documents |
| `DELETE` | `/clear` | Reset all data |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Fast 384-dim semantic vectors |
| Vector DB | `faiss-cpu` | Efficient similarity search |
| Knowledge Graph | `networkx` | Lightweight graph DB (no Neo4j needed) |
| LLM | Google Gemini / Groq | Entity extraction + answer generation |
| PDF Parsing | `PyMuPDF` (fitz) | Text extraction with metadata |
| API | `FastAPI` + `uvicorn` | REST endpoints with auto-docs |
| Terminal UI | `Rich` | Panels, tables, trees, progress bars |
| Config | `python-dotenv` | Environment variable management |

---

## Configuration

All settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_DIM` | `384` | Vector dimensionality |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` |

---

## Example Usage

### Upload a paper
```
Select [1] → Enter path: C:\papers\attention_is_all_you_need.pdf
→ Extracts 15 pages, creates 47 chunks, generates 384-dim embeddings
→ Extracts 23 KG relations (Transformer → uses → Self-Attention, etc.)
```

### Ask a question
```
Question: What is the key innovation in the transformer architecture?

Answer: The transformer architecture's key innovation is the **self-attention
mechanism**, which allows the model to weigh the importance of different
positions in the input sequence...

Sources: chunk_12 (0.847), chunk_3 (0.831), ...
Graph:   Transformer → uses → Self-Attention
         Transformer → replaces → Recurrence
         Self-Attention → enables → Parallelization
```

### Explore knowledge graph
```
Knowledge Graph
├── Transformer
│   ├── uses → Self-Attention
│   ├── replaces → Recurrence
│   └── achieves → State-of-the-art BLEU
├── Self-Attention
│   ├── computes → Scaled Dot-Product
│   └── enables → Parallelization
└── Multi-Head Attention
    └── extends → Self-Attention
```

---

## License

MIT
