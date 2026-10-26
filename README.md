# ⚖️ LegalMind — AI Legal Document Assistant

> Stop drowning in legal documents. Ask questions, get answers — with the page they came from.

LegalMind is a **Retrieval-Augmented Generation (RAG)** application: upload a legal PDF (contract, NDA, terms of service), ask questions in plain language, and get concise answers grounded in the document, each citing the page it came from.

The pipeline is measured, not just demoed: a golden dataset and an LLM-as-a-judge evaluation track retrieval and answer quality — and caught a real retrieval bug along the way ([see Evaluation](#-evaluation)).

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
                                                         ↓
                                  User Question → Embed Question
                                                         ↓
                                                 Semantic Search (top 4)
                                                         ↓
                                     gpt-oss-120b (Groq) → Answer + Page Citation
```

1. **Ingest** — the PDF is extracted page by page and split into ~1000-character chunks; near-empty fragments (signature blocks, footers) are dropped
2. **Embed** — each chunk becomes a 384-dimensional vector with `all-MiniLM-L6-v2`
3. **Store** — vectors, text and page numbers are stored in ChromaDB
4. **Retrieve** — the question is embedded and the 4 most similar chunks are retrieved
5. **Generate** — the LLM answers **only** from those chunks, cites the page, and says *"I cannot find this in the provided document"* when the answer isn't there

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector database | ChromaDB |
| LLM | `openai/gpt-oss-120b` via Groq |
| PDF processing | PyMuPDF (fitz) |
| Chunking | LangChain Text Splitters |
| Evaluation | DeepEval, judged by `qwen/qwen3.8-27b` via Groq |
| Containerization | Docker + Docker Compose |

---

## 🚀 Getting Started

### Prerequisites
- Docker Desktop installed and running
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repository
```bash
git clone https://github.com/semah117/legalmind.git
cd legalmind
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Then put your key in `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Open the app
| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

Upload a PDF, type a question, click **Ask LegalMind**. A sample contract is in [`sample_docs/`](sample_docs/).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server status |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question about the indexed document |

Interactive documentation: `http://localhost:8000/docs`

---

## 📊 Evaluation

### Method
- **Golden dataset** — 10 hand-written questions about [`Data_Engineer_Employment_Contract.pdf`](sample_docs/Data_Engineer_Employment_Contract.pdf), each with an expected answer and the expected source passage. They cover direct lookups, multi-step reasoning (total compensation, severance after 3 years) and a question the contract does *not* answer (non-compete).
- **LLM-as-a-judge** — [DeepEval](https://github.com/confident-ai/deepeval) metrics scored by `qwen/qwen3.8-27b`, a different model family from the answering LLM so the judge isn't grading its own style.

| Metric | What it checks | Component |
|---|---|---|
| Contextual Recall | Did retrieval find everything the expected answer needs? | Retrieval |
| Contextual Precision | Are the retrieved chunks relevant, and ranked first? | Retrieval |
| Faithfulness | Is every claim in the answer supported by the retrieved text? | Generation |
| Answer Relevancy | Does the answer address the question? | Generation |

### Results

| Metric | top-K = 2 | **top-K = 4 + short-chunk filter** |
|---|---|---|
| Contextual Recall | 0.80 | **0.90** |
| Contextual Precision | 0.90 | **0.93** |
| Faithfulness | 0.97 | **0.94** |
| Answer Relevancy | 0.83 | **0.95** |

### What the evaluation caught
With top-K = 2, *"Who owns the code and data pipelines the employee builds?"* got *"I cannot find this in the provided document"* — even though §5.2 Intellectual Property answers it. Ranking the chunks showed the clause in **4th place**, behind a 104-character signature block whose embedding sits close to any question about "the Company" and "the Employee". Raising top-K to 4 and dropping chunks under 200 characters fixed it: the answer now cites §5.2 correctly.

The small Faithfulness drop is the judge seeing *more* context and flagging that the total-compensation answer omits the stipend and RSUs — a real limitation, previously hidden.

**Known limitations**
- The correct chunk is sometimes retrieved but ranked low (a reranker would help)
- Multi-step questions whose facts live in different sections can miss one of them
- The answer occasionally over-states a clause (e.g. bonus paid "only if" metrics are met vs. "based on")

### Run it yourself
Needs `GROQ_API_KEY` in `.env` (used for both the answers and the judge):
```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r backend/requirements.txt deepeval
cd backend
python evaluation/test_pipeline.py
```
Per-question scores and the judge's reasons are written to `backend/evaluation/eval_results.json`.

---

## 📁 Project Structure

```
legalmind/
├── backend/
│   ├── main.py              # FastAPI endpoints (/upload, /ask, /health)
│   ├── ingestion.py         # PDF extraction and chunking
│   ├── embeddings.py        # Sentence Transformer embeddings
│   ├── retrieval.py         # ChromaDB store and search
│   ├── llm.py               # Prompt and answer generation (Groq)
│   ├── evaluation/
│   │   ├── golden_data.py   # Questions, expected answers, expected context
│   │   └── test_pipeline.py # Runs the pipeline and scores it with DeepEval
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── sample_docs/             # Sample contract used by the evaluation
├── docker-compose.yaml
├── .env.example
└── README.md
```

---

## 👤 Author

**Semah** — [@semah117](https://github.com/semah117)

Data Science & AI Engineering Student — École Polytechnique de Sousse
