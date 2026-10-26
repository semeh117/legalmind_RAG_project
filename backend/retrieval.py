import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

"Initialize ChromaDB client and collection"
client = chromadb.Client()
collection = client.get_or_create_collection(name="legalmind")

def store(chuncks :list[dict]) -> None:
    """Store the chunks in ChromaDB."""
    collection.add(
        embeddings=[chunk["embedding"] for chunk in chuncks],
        documents=[chunk["text"] for chunk in chuncks],
        metadatas=[{"page_number": chunk["page_number"], "chunk_number": chunk["chunk_number"]} for chunk in chuncks],
        ids=[f"{chunk['page_number']}_{chunk['chunk_number']}" for chunk in chuncks]
    )

def retrieve(quest: str, top_k: int = 4) -> list[dict]:
    """Retrieve relevant chunks based on the query."""
    query_embedding = model.encode(quest).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
