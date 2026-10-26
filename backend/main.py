import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel 
from ingestion import extract_text_from_pdf, split_into_chunks
from embeddings import embed_chunks
from retrieval import store, retrieve
from llm import get_answer

app = FastAPI(title="LegalMind API")

# Pydantic model for the question request
class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    """Check if server is running."""
    return {"status": "LegalMind is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Receive PDF, process it, store in ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Full pipeline
    pages = extract_text_from_pdf(temp_path)
    chunks = split_into_chunks(pages)
    embedded_chunks = embed_chunks(chunks)
    store(embedded_chunks)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return {"message": f"Successfully processed {len(chunks)} chunks from {file.filename}"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Receive question, retrieve from ChromaDB, return answer."""
    results = retrieve(request.question)
    answer = get_answer(request.question, results)
    return {"answer": answer}