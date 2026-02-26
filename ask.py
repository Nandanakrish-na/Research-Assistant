import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="research_papers")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    return chunks, sources

def generate_answer(query, chunks, sources):
    context =""
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        context += f"[{i+1}] From {source['source']}:\n{chunk}\n\n"

    prompt = f"""You are a research assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I cannot find this in the provided papers."
Always mention which paper your answer comes from. 

CONTEXT:
{context}

QUESTION: {query} 

ANSWER:"""
        
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content



def main():
    print("Research Assistant Ready!")
    print("Type your question and press Enter. Type 'quit' to exit. \n")

    while True:
        query = input("You:").strip()

        if query.lower() =="quit":
            break

        if not query:
            continue

        chunks, sources = retrieve_chunks(query)
        answer = generate_answer(query, chunks, sources)


        print(f"\nAssistant: {answer}\n")
        print("-" * 50 + "\n")

main()
