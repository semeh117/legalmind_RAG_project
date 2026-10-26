import sys
import os
import time

# ────────────────────────────────────────────────────────────────
# Path setup
# ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
)
load_dotenv(ENV_PATH)

assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not found in .env"
print("✅ GROQ_API_KEY loaded (pipeline answers + judge)")

# ────────────────────────────────────────────────────────────────
# DeepEval
# ────────────────────────────────────────────────────────────────
from groq import Groq
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

# ────────────────────────────────────────────────────────────────
# Your pipeline
# ────────────────────────────────────────────────────────────────
from ingestion import extract_text_from_pdf, split_into_chunks
from embeddings import embed_chunks
from retrieval import store, retrieve
from llm import get_answer
from evaluation.golden_data import GOLDEN_DATASET


# ────────────────────────────────────────────────────────────────
# Groq judge — Qwen scores the answers. Different model family from
# the pipeline LLM (gpt-oss) so the judge isn't grading its own style.
# ────────────────────────────────────────────────────────────────
class GroqEvaluator(DeepEvalBaseLLM):
    """Judge using Qwen on Groq. JSON mode + hidden reasoning give
    clean structured output, which DeepEval's metrics must parse.
    max_retries lets the client wait out the free tier's
    tokens-per-minute limit (429) instead of failing the metric."""

    def __init__(self):
        self.model_id = "qwen/qwen3.8-27b"
        self.client = Groq(max_retries=10)

    def get_model_name(self) -> str:
        return self.model_id

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            reasoning_format="hidden",
        )
        elapsed = time.perf_counter() - start
        print(f"    ⚡ Judge response in {elapsed:.2f}s")
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


# ===============================================================
# Step 1 — Ingest document
# ===============================================================
PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sample_docs",
    "Data_Engineer_Employment_Contract.pdf",
)

print(f"\n📄 Ingesting contract from: {PDF_PATH}")
pages = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(pages)
embedded = embed_chunks(chunks)
store(embedded)
print(f"✅ Stored {len(chunks)} chunks")


# ===============================================================
# Step 2 — Build test cases (uses Groq via get_answer)
# ===============================================================
print("\n🔨 Building test cases...")
test_cases = []

for i, entry in enumerate(GOLDEN_DATASET):
    print(f"  [{i+1}/{len(GOLDEN_DATASET)}] {entry['question'][:60]}...")

    retrieved_chunks = retrieve(entry["question"])
    answer = get_answer(entry["question"], retrieved_chunks)

    test_case = LLMTestCase(
        input=entry["question"],
        actual_output=answer,
        expected_output=entry["expected_answer"],
        retrieval_context=retrieved_chunks["documents"][0],
        expected_context=[entry["expected_context"]],
    )
    test_cases.append(test_case)

print(f"✅ Built {len(test_cases)} test cases")


# ===============================================================
# DEBUG MODE — run only 2 cases first. Change to full list once stable.
# ===============================================================
DEBUG_LIMIT = None  # None = full golden dataset
test_cases = test_cases[:DEBUG_LIMIT]
print(f"⚠️  Debug mode: running {len(test_cases)} of {len(GOLDEN_DATASET)} cases")


# ===============================================================
# Step 3 — Define metrics (Qwen on Groq scores them)
# ===============================================================
evaluator = GroqEvaluator()

metrics = [
    ContextualRecallMetric(threshold=0.5, model=evaluator),
    ContextualPrecisionMetric(threshold=0.5, model=evaluator),
    FaithfulnessMetric(threshold=0.5, model=evaluator),
    AnswerRelevancyMetric(threshold=0.5, model=evaluator),
]


# ===============================================================
# Step 4 — Run evaluation
# One metric at a time, sequentially — keeps under the judge's
# tokens-per-minute rate limit on the Groq free tier.
# metric.measure() is called directly (instead of evaluate()) so the
# score/reason can be read back per test case. Results are saved
# after every case so a crash doesn't lose finished work.
# ===============================================================
import json

results_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_results.json"
)

print("\n🚀 Starting evaluation (rate limits may add pauses — that's expected)\n")

output = []
for i, test_case in enumerate(test_cases):
    print("=" * 70)
    print(f"TEST CASE {i+1}/{len(test_cases)}: {test_case.input[:60]}")
    print("=" * 70)

    scores = {}
    for metric in metrics:
        name = metric.__class__.__name__
        print(f"\nRunning {name}...")
        start = time.perf_counter()
        try:
            metric.measure(test_case)
            elapsed = time.perf_counter() - start
            print(f"✅ {name}: {metric.score:.2f} ({elapsed:.2f}s)")
            scores[name] = {
                "score": metric.score,
                "success": metric.success,
                "reason": metric.reason,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"❌ Failed after {elapsed:.2f}s — {type(e).__name__}: {e}")
            scores[name] = {"score": None, "error": f"{type(e).__name__}: {e}"}

    output.append({
        "question": test_case.input,
        "actual_output": test_case.actual_output,
        "expected_output": test_case.expected_output,
        "scores": scores,
    })
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ===============================================================
# Step 5 — Summary (average score per metric)
# ===============================================================
print("\n📊 Average scores:")
for metric in metrics:
    name = metric.__class__.__name__
    vals = [r["scores"][name]["score"] for r in output
            if r["scores"][name].get("score") is not None]
    avg = sum(vals) / len(vals) if vals else float("nan")
    print(f"  {name:28s} {avg:.2f}  ({len(vals)}/{len(output)} scored)")

print(f"\n💾 Results saved to {results_path}")
print("\n🎉 Evaluation complete.")
