import fitz

def create_sample_paper():
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    title = "Neural Graph Reasoning: A Novel Approach to Hybrid RAG Systems"
    authors = "Dr. Alice Turing, Dr. Bob Von Neumann\nAI Research Institute"
    abstract = ("Abstract\n"
                "Current Retrieval-Augmented Generation (RAG) systems rely heavily on dense vector \n"
                "embeddings for semantic similarity. While effective for localized queries, they \n"
                "often fail at multi-hop reasoning over complex corpora. In this paper, we introduce \n"
                "Neural Graph Reasoning (NGR), a hybrid architecture that fuses FAISS-based vector \n"
                "retrieval with NetworkX-driven knowledge graphs holding entity-relation triples. \n"
                "Our evaluation demonstrates a 34% improvement in hallucination reduction over \n"
                "baseline dense retrieval systems.")
    
    intro = ("1. Introduction\n"
             "The advent of Large Language Models (LLMs) like Claude and Gemini has revolutionized \n"
             "natural language processing. However, these models suffer from context-window limitations \n"
             "and factual drift. RAG was proposed as a solution by Lewis et al. to provide grounding. \n"
             "Despite this, purely vector-based RAG struggles to connect disparate pieces of knowledge. \n"
             "For example, linking 'Agent A' to 'Event B' when they appear in entirely different \n"
             "documents is an open challenge.")

    page1.insert_text((50, 70), title, fontsize=16, fontname="helv", color=(0, 0, 0))
    page1.insert_text((50, 110), authors, fontsize=12, fontname="helv", color=(0.3, 0.3, 0.3))
    page1.insert_text((50, 160), abstract, fontsize=11, fontname="times-roman", color=(0, 0, 0))
    page1.insert_text((50, 280), intro, fontsize=11, fontname="times-roman", color=(0, 0, 0))

    # Page 2
    page2 = doc.new_page()
    methodology = ("2. Methodology\n"
                   "Our system utilizes an ingestion pipeline that first extracts unstructured text using \n"
                   "PyMuPDF. The text is chunked into 500-character segments with a 50-character overlap. \n"
                   "We then pass these segments through a 'sentence-transformers/all-MiniLM-L6-v2' model \n"
                   "to generate 384-dimensional embeddings. Simultaneously, an LLM extracts \n"
                   "structured [Entity-Relation-Entity] triples from the text, constructing a highly \n"
                   "connected NetworkX directed graph.\n\n"
                   "3. Results and Limitations\n"
                   "Testing on the MultiHop-QA dataset, our hybrid NGR system outperformed dense-only RAG \n"
                   "by a significant margin. However, the system's primary limitation lies in the LLM's \n"
                   "variable latency during entity extraction, adding processing overhead during ingestion.\n\n"
                   "4. Future Work\n"
                   "Future iterations will explore replacing NetworkX with Neo4j for scalable graph \n"
                   "storage across distributed nodes, and adapting the chunking algorithm to be natively \n"
                   "semantic rather than rigidly character-based.")
                   
    page2.insert_text((50, 70), methodology, fontsize=11, fontname="times-roman", color=(0, 0, 0))

    out_file = "sample_research_paper.pdf"
    doc.save(out_file)
    print(f"Created {out_file} successfully.")

if __name__ == "__main__":
    create_sample_paper()
