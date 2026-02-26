# Research Assistant — RAG Pipeline

A domain-specific research assistant built using Retrieval-Augmented Generation (RAG) for Offline Reinforcement Learning and Sepsis Treatment research papers.

## What it does
- Ingests research papers (PDFs) into a vector database
- Answers questions strictly grounded in the uploaded papers
- Cites which paper each answer comes from
- Runs as a web chat interface

## Tech Stack
- **PDF Parsing** — PyMuPDF
- **Text Chunking** — LangChain
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database** — ChromaDB
- **LLM** — Llama 3.3 70B via Groq
- **UI** — Gradio

## How to Run

### 1. Clone the repository
git clone https://github.com/Nandanakrish-na/Research-Assistant.git
cd Research-Assistant

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install pymupdf langchain langchain-text-splitters chromadb sentence-transformers groq gradio python-dotenv pypdf

### 4. Add API key
Create a `.env` file and add your Groq API key:
GROQ_API_KEY=your-key-here

### 5. Add your PDFs
Place your research papers in the `papers/` folder.

### 6. Ingest papers
python ingest.py

### 7. Run the assistant
python ui.py

Then open `http://127.0.0.1:7860` in your browser.

## Project Structure
research_assistant/
├── ingest.py       — PDF ingestion pipeline
├── ask.py          — Terminal version
├── ui.py           — Web chat interface
├── papers/         — Your PDF research papers
├── chroma_db/      — Vector database (auto-created)
└── .env            — API keys (never shared)
