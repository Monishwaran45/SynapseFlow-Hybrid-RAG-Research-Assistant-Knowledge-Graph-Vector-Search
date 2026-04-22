"""
SynapseFlow v1.0 — Smart Research Assistant
Knowledge Graph + RAG Hybrid Intelligence

Run: python app.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Force UTF-8 on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console, Group
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich.columns import Columns
from rich.layout import Layout
from rich import box

import config

console = Console(force_terminal=True)

# ─── Claude-Inspired Palette ──────────────────────────────────
C_PRIMARY = "#D76140"    # Anthropic orange/peach
C_TEXT = "#E0E0E0"       # Off-white text
C_DIM = "#80746E"        # Warm dim grey
C_ACCENT = "#D1B280"     # Warm beige accent
C_SUCCESS = "#578C6A"    # Subdued green
C_BORDER = "#403A36"     # Dark grey border

# ═══════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════

def load_registry():
    if config.DOC_REGISTRY_PATH.exists():
        try:
            with open(config.DOC_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_registry(reg):
    config.DOC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DOC_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def boot_sequence():
    """Minimalist, professional startup sequence."""
    console.clear()
    console.print()
    
    logo = Text(" ✧ S Y N A P S E F L O W ", style=f"bold {C_PRIMARY}")
    subtitle = Text("   Knowledge Graph & RAG Reasoning Engine", style=f"italic {C_DIM}")
    
    console.print(logo)
    console.print(subtitle)
    console.print(f"[{C_BORDER}]" + "─" * 45 + "[/]")
    console.print()
    
    steps = [
        ("Vector Index", "Initializing semantic search"),
        ("Graph Core", "Loading dependency network"),
        ("Inference", "Establishing LLM gateway")
    ]
    
    for name, desc in steps:
        with console.status(f"[{C_DIM}]Boot: {desc}...[/]", spinner="point", spinner_style=C_PRIMARY):
            time.sleep(0.4)
        console.print(f"[{C_SUCCESS}]✓[/] [{C_TEXT}]{name}[/] [{C_DIM}]{desc}[/]")
    
    time.sleep(0.3)
    console.print()


def banner():
    """Elegant, minimalist header."""
    header = Table.grid(expand=True)
    header.add_column(justify="left", ratio=1)
    header.add_column(justify="right")
    
    title = Text("✧ SynapseFlow", style=f"bold {C_PRIMARY}")
    title.append(" / Research Intelligence", style=f"{C_DIM}")
    
    header.add_row(
        title,
        Text("v1.0.0", style=f"italic {C_DIM}")
    )
    console.print(Panel(header, border_style=C_BORDER, box=box.ROUNDED))


def stats_bar(vs, gs, reg):
    """Clean, inline stats."""
    v = vs.total_vectors if vs else 0
    g = gs.get_stats() if gs else {"nodes": 0, "edges": 0}
    d = len(reg)
    
    stats_text = Text()
    stats_text.append(" Documents: ", style=C_DIM)
    stats_text.append(f"{d} ", style=f"bold {C_PRIMARY}")
    stats_text.append("│ Entities: ", style=C_DIM)
    stats_text.append(f"{g['nodes']} ", style=f"bold {C_TEXT}")
    stats_text.append("│ Relations: ", style=C_DIM)
    stats_text.append(f"{g['edges']} ", style=f"bold {C_TEXT}")
    stats_text.append("│ Vectors: ", style=C_DIM)
    stats_text.append(f"{v}", style=f"bold {C_TEXT}")
    
    console.print(Panel(stats_text, border_style=C_BORDER, box=box.MINIMAL))


def menu():
    """Minimalist menu."""
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style=f"bold {C_PRIMARY}")
    t.add_column("Action", style=C_TEXT)
    t.add_column("Detail", style=C_DIM)
    
    t.add_row("1", "Process Document",   "Ingest PDF, build vectors & graph")
    t.add_row("2", "Research Query",     "Ask questions across documents")
    t.add_row("3", "Explore Graph",      "Visualize knowledge structures")
    t.add_row("4", "Document Library",   "View ingested papers")
    t.add_row("5", "Extract Insights",   "Analyze paper contributions")
    t.add_row("", "", "")
    t.add_row("c", "Clear Storage",      "Reset all data")
    t.add_row("q", "Quit",               "Exit interface")
    
    console.print(Panel(t, title=Text(" Commands ", style=f"italic {C_DIM}"), title_align="left", border_style=C_BORDER, box=box.ROUNDED))


def section(title_text):
    """Clean section header."""
    console.clear()
    banner()
    console.print()
    console.print(Text(f" {title_text} ", style=f"reverse {C_PRIMARY} black"))
    console.print()


def wait():
    console.print()
    Prompt.ask(f"[{C_DIM}]Press Enter to return[/]")


# ═══════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════

vector_store = None
graph_store = None
retriever = None
llm = None
doc_registry = []


def init_stores():
    global vector_store, graph_store, retriever
    from backend.db.vector_store import VectorStore
    from backend.db.graph_store import GraphStore
    from backend.services.retriever import HybridRetriever
    vector_store = VectorStore()
    graph_store = GraphStore()
    retriever = HybridRetriever(vector_store, graph_store)


def ensure_llm():
    global llm
    if llm is not None:
        return True
    provider = config.LLM_PROVIDER
    key = config.GEMINI_API_KEY if provider == "gemini" else config.GROQ_API_KEY
    if not key:
        key = Prompt.ask(f"[{C_DIM}]Provide {provider.upper()} API key[/]").strip()
        if not key:
            console.print(f"[{C_PRIMARY}]Access requires an API key.[/]"); return False
        if provider == "gemini":
            os.environ["GEMINI_API_KEY"] = key; config.GEMINI_API_KEY = key
        else:
            os.environ["GROQ_API_KEY"] = key; config.GROQ_API_KEY = key
    try:
        from backend.services.llm_service import LLMService
        llm = LLMService()
        return True
    except Exception as e:
        console.print(f"[{C_PRIMARY}]Model initialization failed: {e}[/]"); llm = None; return False


# ═══════════════════════════════════════════════════════════════
#  [1] UPLOAD PDF
# ═══════════════════════════════════════════════════════════════

def upload_pdf():
    global doc_registry
    section("Document Ingestion Pipeline")

    path = Prompt.ask(f"[{C_TEXT}]Enter PDF file path[/]").strip().strip('"').strip("'")
    if not os.path.exists(path):
        console.print(f"[{C_PRIMARY}]File not found: {path}[/]"); wait(); return
    if not path.lower().endswith('.pdf'):
        console.print(f"[{C_PRIMARY}]Target must be a PDF file.[/]"); wait(); return
    if not ensure_llm():
        wait(); return

    doc_id = Path(path).stem.replace(' ', '_').replace('-', '_')
    if any(d["doc_id"] == doc_id for d in doc_registry):
        if not Confirm.ask(f"\n[{C_ACCENT}]Document '{doc_id}' exists. Overwrite?[/]"):
            return

    console.print()
    with Progress(SpinnerColumn("point", style=C_PRIMARY),
                  TextColumn(f"[{C_TEXT}]" + "{task.description}"),
                  BarColumn(bar_width=40, style=C_BORDER, complete_style=C_PRIMARY, finished_style=C_SUCCESS),
                  TaskProgressColumn(), console=console) as prog:
        t = prog.add_task("Initializing...", total=100)

        prog.update(t, description="Reading binary stream...", completed=5)
        from backend.services.pdf_parser import extract_text
        result = extract_text(path)
        prog.update(t, completed=15)

        prog.update(t, description="Semantic chunking...")
        from backend.services.chunking import chunk_text
        chunks = chunk_text(result["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP, doc_id)
        prog.update(t, completed=30)

        prog.update(t, description="Computing dense vectors...")
        from backend.services.embeddings import embed_texts
        embeddings = embed_texts([c["text"] for c in chunks])
        prog.update(t, completed=50)

        prog.update(t, description="Indexing vectors...")
        vector_store.add(embeddings, chunks)
        prog.update(t, completed=60)

        kg_count = 0
        kg_chunks = chunks[:15]
        for i, ch in enumerate(kg_chunks):
            prog.update(t, description=f"Extracting entities ({i+1}/{len(kg_chunks)})...",
                        completed=60 + int(40 * (i+1) / len(kg_chunks)))
            try:
                rels = llm.extract_entities_relations(ch["text"])
                if rels:
                    graph_store.add_relations_batch(rels, doc_id)
                    kg_count += len(rels)
            except Exception:
                pass
        prog.update(t, description=f"Pipeline complete.", completed=100)

    entry = {
        "doc_id": doc_id, "filename": os.path.basename(path),
        "pages": result["metadata"]["page_count"], "chunks": len(chunks),
        "kg_relations": kg_count,
        "title": result["metadata"].get("title", doc_id),
        "author": result["metadata"].get("author", "Unknown"),
        "chars": result["metadata"]["total_chars"],
    }
    doc_registry = [d for d in doc_registry if d["doc_id"] != doc_id]
    doc_registry.append(entry)
    save_registry(doc_registry)

    console.print()
    tb = Table(box=box.MINIMAL, show_header=False)
    tb.add_column("Key", style=C_DIM)
    tb.add_column("Value", style=C_TEXT)
    tb.add_row("Document",   entry["filename"])
    tb.add_row("Author",     entry["author"])
    tb.add_row("Size",       f"{entry['pages']} pages  |  {entry['chars']:,} chars")
    tb.add_row("Index",      f"{entry['chunks']} vectors")
    tb.add_row("Graph",      f"{kg_count} relational edges")
    console.print(Panel(tb, border_style=C_BORDER, box=box.ROUNDED, padding=(1, 2)))
    wait()


# ═══════════════════════════════════════════════════════════════
#  [2] ASK QUESTION
# ═══════════════════════════════════════════════════════════════

def ask_question():
    section("Research Query")

    if vector_store.total_vectors == 0:
        console.print(f"[{C_DIM}]Index is empty. Ingest a document first.[/]"); wait(); return
    if not ensure_llm():
        wait(); return

    console.print(f"[{C_DIM}]Enter your question (natural language). Type 'q' to return.[/]")

    while True:
        console.print()
        q = Prompt.ask(f"[{C_PRIMARY}]✧[/]")
        if q.lower().strip() in ('q', 'quit', 'exit', 'back'):
            break
        if not q.strip():
            continue

        console.print()
        with console.status(f"[{C_DIM}]Reasoning over vectors & graph...[/]", spinner="point", spinner_style=C_PRIMARY):
            results = retriever.search(q, k=5)
            text_ctx = "\n---\n".join(r["text"] for r in results["vector_results"])
            graph_ctx = "\n".join(
                f"{r['source']} -> [{r['relation']}] -> {r['target']}"
                for r in results["graph_results"]
            ) if results["graph_results"] else ""
            answer = llm.generate_answer(q, text_ctx, graph_ctx)

        # Answer Output (Claude-style markdown rendering)
        console.print(Markdown(answer))
        console.print()

        # Context details (Subdued)
        if results["vector_results"] or results["graph_results"]:
            ctx_text = Text("Sources Used: ", style=C_DIM)
            
            # Show top 2 text sources
            for i, r in enumerate(results["vector_results"][:2]):
                doc_name = r.get("metadata", {}).get("doc_id", "Doc")
                ctx_text.append(f"[{doc_name} (sim:{r['score']:.2f})] ", style=f"italic {C_DIM}")
                
            if results["graph_results"]:
                ctx_text.append(f"| Graph Links: {len(results['graph_results'])}", style=C_DIM)
                
            console.print(ctx_text)

        console.print(f"\n[{C_BORDER}]" + "─" * console.width + "[/]")


# ═══════════════════════════════════════════════════════════════
#  [3] VIEW KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════

def view_graph():
    section("Knowledge Graph Explorer")
    gs = graph_store.get_stats()
    if gs["nodes"] == 0:
        console.print(f"[{C_DIM}]Graph is empty.[/]"); wait(); return

    console.print(f"[{C_DIM}]Network size:[/] {gs['nodes']} nodes [dim]│[/] {gs['edges']} edges [dim]│[/] {gs['components']} components\n")

    rels = graph_store.get_all_relations()
    tree = Tree(f"[{C_TEXT}]Ontology Root[/]")
    sources = {}
    for r in rels:
        sources.setdefault(r["source"], []).append(r)
        
    for src, rs in list(sources.items())[:15]:
        b = tree.add(f"[{C_ACCENT}]{src}[/]")
        for r in rs[:5]:
            b.add(f"[{C_DIM}]{r['relation']} ──>[/] [{C_TEXT}]{r['target']}[/]")
            
    console.print(Panel(tree, border_style=C_BORDER, box=box.ROUNDED, padding=(1, 2)))

    if Confirm.ask(f"\n[{C_TEXT}]Query specific entity?[/]", show_default=False):
        ent = Prompt.ask(f"[{C_PRIMARY}]Entity name[/]")
        hits = graph_store.query(ent)
        if hits:
            console.print()
            for h in hits:
                console.print(f"  [{C_ACCENT}]{h['source']}[/] [{C_DIM}]── {h['relation']} ──>[/] [{C_TEXT}]{h['target']}[/]")
        else:
            console.print(f"\n[{C_DIM}]No relations mapped for '{ent}'.[/]")
    wait()


# ═══════════════════════════════════════════════════════════════
#  [4] VIEW DOCUMENTS
# ═══════════════════════════════════════════════════════════════

def view_documents():
    section("Document Library")
    if not doc_registry:
        console.print(f"[{C_DIM}]Library is empty.[/]"); wait(); return

    t = Table(box=box.SIMPLE, show_edge=False, border_style=C_BORDER)
    t.add_column("File", style=C_TEXT)
    t.add_column("Pages", justify="right", style=C_DIM)
    t.add_column("Vectors", justify="right", style=C_DIM)
    t.add_column("Relations", justify="right", style=C_DIM)
    
    for d in doc_registry:
        t.add_row(
            d.get("filename", d["doc_id"]), 
            str(d.get("pages","0")),
            str(d.get("chunks","0")), 
            str(d.get("kg_relations","0"))
        )
    console.print(Panel(t, border_style=C_BORDER, box=box.ROUNDED))
    wait()


# ═══════════════════════════════════════════════════════════════
#  [5] EXTRACT INSIGHTS
# ═══════════════════════════════════════════════════════════════

def extract_insights():
    section("Analytical Insights")
    if not doc_registry:
        console.print(f"[{C_DIM}]Library is empty.[/]"); wait(); return
    if not ensure_llm():
        wait(); return

    doc = doc_registry[0]
    if len(doc_registry) > 1:
        for i, d in enumerate(doc_registry, 1):
            console.print(f"  [{C_PRIMARY}]{i}[/] [{C_TEXT}]{d.get('filename', d['doc_id'])}[/]")
        console.print()
        idx = Prompt.ask(f"[{C_DIM}]Select document index[/]", default="1")
        try:
            doc = doc_registry[int(idx) - 1]
        except (ValueError, IndexError):
            console.print(f"[{C_PRIMARY}]Invalid selection.[/]"); wait(); return

    chunks = [m for m in vector_store.metadata if m.get("doc_id") == doc["doc_id"]]
    if not chunks:
        console.print(f"[{C_DIM}]No index data available for this document.[/]"); wait(); return

    combined = "\n\n".join(c["text"] for c in chunks[:10])
    with console.status(f"[{C_DIM}]Synthesizing document insights...[/]", spinner="point", spinner_style=C_PRIMARY):
        insights = llm.extract_insights(combined)

    console.print()
    console.print(f"[{C_TEXT} bold]Insight Synthesis: {doc.get('filename', doc['doc_id'])}[/]")
    console.print(f"[{C_BORDER}]" + "─" * 50 + "[/]")

    for label, key in [("Core Contributions", "contributions"),
                       ("Methodological Approach", "methodology"),
                       ("Noted Constraints", "limitations"),
                       ("Forward Vectors", "future_work")]:
        items = insights.get(key, [])
        if items:
            console.print(f"\n[{C_ACCENT}]{label}[/]")
            for it in items:
                console.print(f" [{C_DIM}]•[/] [{C_TEXT}]{it}[/]")
    wait()


# ═══════════════════════════════════════════════════════════════
#  [6] CLEAR DATA
# ═══════════════════════════════════════════════════════════════

def clear_data():
    section("System Reset")
    console.print(f"[{C_PRIMARY}]Warning: This will purge all vectors, graphs, and document schemas.[/]")

    if Confirm.ask(f"\n[{C_TEXT}]Proceed with purge?[/]", default=False, show_default=False):
        vector_store.clear(); graph_store.clear()
        global doc_registry
        doc_registry = []; save_registry(doc_registry)
        console.print(f"[{C_SUCCESS}]System memory cleared.[/]")
    wait()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global doc_registry
    doc_registry = load_registry()

    try:
        boot_sequence()
        init_stores()

        while True:
            console.clear()
            banner()
            stats_bar(vector_store, graph_store, doc_registry)
            console.print()
            menu()
            console.print()

            choice = Prompt.ask(f"[{C_PRIMARY}]✧[/]", show_default=False).lower()
            if   choice == "1": upload_pdf()
            elif choice == "2": ask_question()
            elif choice == "3": view_graph()
            elif choice == "4": view_documents()
            elif choice == "5": extract_insights()
            elif choice == "c": clear_data()
            elif choice in ("0", "q", "quit", "exit"):
                console.print(f"\n[{C_DIM}]Session terminated.[/]")
                break

    except KeyboardInterrupt:
        console.print(f"\n[{C_DIM}]Session terminated.[/]")


if __name__ == "__main__":
    main()
