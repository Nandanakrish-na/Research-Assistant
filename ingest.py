import os
from pathlib import Path
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()
ai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_papers")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    print(f"  Extracted {len(text)} characters")
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    return chunks

def embed_text(text):
    return embedding_model.encode(text).tolist()

def ingest_papers(folder_path):
    folder = Path(folder_path)
    pdfs = list(folder.glob("*.pdf"))

    print(f"Found {len(pdfs)} PDFs")

    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")

        text = read_pdf(pdf_path)
        chunks = chunk_text(text)

        print(f" {len(chunks)} chunks created")

        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            collection.add(
                ids=[f"{pdf_path.name}_{i}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": pdf_path.name}]
            )
        print(f" Done storing in ChromaDB")

    print("All papers ingested!") 


ingest_papers("./papers")           