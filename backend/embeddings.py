from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks : list[dict])-> list[dict]:
    """Generate embeddings for each text chunk."""
    for chunk in chunks:
        chunk["embedding"] = model.encode(chunk["text"]).tolist()  # Convert to list for JSON serialization
    return chunks
