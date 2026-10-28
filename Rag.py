import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4
SHOW_CONTEXT = True
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DB_FOLDER = "chroma_store"  

load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
def get_setting(name):
    value = os.getenv(name)
    if not value:
        print(f"Error: '{name}' is missing. Add it to .env file")
        sys.exit(1)
    return value

API_KEY = get_setting("LLAMA_API_KEY")
BASE_URL = get_setting("LLAMA_BASE_URL")
MODEL = get_setting("LLAMA_MODEL")

llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("Loading embedding model...")
embedder = DefaultEmbeddingFunction()

def embed(texts):
    return [
        [float(value) for value in vector]
        for vector in embedder(texts)
    ]
db = chromadb.PersistentClient(path=DB_FOLDER)

collection = db.get_or_create_collection(
    name = "docs",
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            yield piece


def read_file(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append((page_number, page.extract_text() or ""))
        return pages
    
    if path.lower().endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8") as f:
            return [(1, f.read())]

    print("Unsupported file type. Use .pdf, .txt or md")
    return []

def ingest(path):
    if not os.path.exists(path):
        print(f"File not found : {path}")
        return
    file_name = os.path.basename(path)
    ids, docs, metas = [], [], []

    for page_number, text in read_file(path):
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{file_name}-p{page_number}-c{i}")
            docs.append(chunk)
            metas.append({"source": file_name, "page": page_number})
    if not docs:
        print("No text found. If this is scanner pdf, it needs OCR first.")
        return
    
    print(f"Embedding {len(docs)} chunks...")
    embeddings = embed(docs)

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas
    )
    print(f"Done! Stored {len(docs)} chunks from '{file_name}'.")

def retrieve(question, k=TOP_K):
    question_vector = embed([question])
    result = collection.query(query_embeddings=question_vector, n_results=k)
    hits = []
        
    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "similarity": 1 - distance,
        })
    return hits

def build_prompt(question, hits):
    context = "\n\n".join(
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in hits
    )

    system_message = (
        "You are a helpful document assistant. "
        "Answer the question using ONLY the context provided. "
        "If the answer is not in the context, reply exactly: "
        "\"I couldn't find that in the documents.\" "
        "Do not use outside knowledge. "
        "At the end, mention the source file and page number(s) you used."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    if SHOW_CONTEXT:
        print("\n === Context Sent to LLAMA ===")
        print(context)
        print("=========")
    
    return [
        {"role":"system", "content": system_message},
        {"role":"user", "content": user_message}
    ]

def ask(question, k=TOP_K):
    hits = retrieve(question, k)
    if not hits:
        return "No documents found. Run: python rag.py ingest yourfile.pdf"
    
    messages = build_prompt(question, hits)

    try:
        response = llm.chat.completions.create(
            model = MODEL,
            messages = messages,
            temperature = 0.2
        )
    except Exception as error:
        return f"API error: {error}"
    
    return response.choices[0].message.content

def debug(question, k=TOP_K):
    hits = retrieve(question, k)
    print(f"\n Top {len(hits)} chunks for: {question!r}")
    for h in hits:
        print(f"\n--- {h['source']} | page {h['page']} | similarity {h['similarity']:.2f}")
        print(h["text"][:300] + ("..." if len(h["text"]) > 300 else ""))


if __name__ == "__main__":
    args = sys.argv[1:]
 
    if len(args) >= 2 and args[0] == "ingest":
       
        ingest(args[1])
 
    elif len(args) >= 2 and args[0] == "debug":
      
        debug(" ".join(args[1:]))
 
    elif args and args[0] == "reset":
        db.delete_collection("docs")
        print("All stored documents deleted.")
 
    else:
        print(f"\nStored chunks: {collection.count()}")
        if collection.count() == 0:
            print("Tip: add a document first -> python rag.py ingest yourfile.pdf")
        print("Ask questions about your documents (type 'exit' to quit)")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not question:
                continue  
            print("\nAssistant:", ask(question))
