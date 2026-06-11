<div align="center">

# 📚 Hybrid RAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-ff6b35?style=flat-square)](https://python.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-0F6FFF?style=flat-square)](https://www.pinecone.io/)
[![Cohere](https://img.shields.io/badge/Cohere-Reranker-39594D?style=flat-square)](https://cohere.com/)
[![Redis](https://img.shields.io/badge/Redis-Semantic_Cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-f55036?style=flat-square)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

A production-grade, domain-specific Retrieval-Augmented Generation (RAG) system built with LangChain. Query a fixed set of documents using a hybrid search pipeline — combining BM25 sparse retrieval and dense vector search in a single Pinecone index — with Cohere cross-encoder reranking, citation-enforced generation, and a Streamlit chat UI.

---
</div>

## Architecture Overview

```
Documents
    │
    ▼
Ingestion Pipeline
    ├─ Load (PDF, DOCX, HTML, Markdown)
    ├─ Chunk (recursive / token / markdown-header)
    ├─ Deduplicate (SimHash)
    ├─ Embed (HuggingFace)
    ├─ Fit BM25 encoder → bm25_encoder.json
    └─ Upsert → Pinecone sparse-dense index
                         │
                         ▼
              Query (read-only at runtime)
                         │
              RetrievalPipeline
    ├─ Query Expansion (HyDE / multi-query)
    ├─ Pinecone Hybrid Query (dense + BM25 in one call, alpha=0.5)
    ├─ Cohere Reranker (rerank-english-v3.0, top-5)
    └─ Context Compression (keyword-overlap scoring)
                         │
              RAGChain (Generation)
    ├─ Prompt Builder ([doc-N] citation rules)
    ├─ LLM (OpenAI / Anthropic)
    ├─ Citation Enforcer (strip hallucinated refs)
    └─ Answer Validator (grounding check)
                         │
              FastAPI  ──►  Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | LangChain |
| Vector Store | Pinecone (sparse-dense hybrid) |
| Sparse Retrieval | BM25Encoder (`pinecone-text`) |
| Embeddings | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| Reranker | Cohere `rerank-english-v3.0` |
| LLM | OpenAI GPT-4o / Anthropic Claude |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Session Store | Redis |
| Semantic Cache | Redis (cosine similarity, threshold 0.92) |
| Tracing | LangSmith |
| Evaluation | RAGAS |

---

## Project Structure

```
Hybrid-RAG/
│
├── config.py                        # All settings via .env (pydantic-settings)
├── ingest.py                        # One-time CLI ingestion script
├── streamlit_app.py                 # Streamlit frontend
├── requirements.txt                 # dependencies
├── docker-compose.yml               # Local development stack for Hybrid RAG.
├── bm25_encoder.json                # Auto-generated after first ingest
src/
  │
  ├── ingestion/
  │   ├── loaders.py                     # PDF, HTML, DOCX, Markdown loaders
  │   ├── chunkers.py                    # Recursive, token, markdown-header splitting
  │   ├── embedders.py                   # OpenAI, Cohere, HuggingFace embeddings
  │   ├── metadata_extractor.py          # Title, word count, SHA-256 hash, SimHash dedup
  │   └── ingestion_pipeline.py          # Orchestrates all steps + BM25 fitting
  │
  ├── retrieval/
  │   ├── Hybrid_retriever.py            # Pinecone hybrid (dense + BM25) query
  │   ├── Reranker.py                    # Cohere Rerank API
  │   ├── Query_expander.py              # HyDE + multi-query expansion
  │   └── Retrieval_pipeline.py          # Full retrieval orchestrator   
  │
  ├── generation/
  │   ├── rag_chain.py                   # Main RAG orchestrator
  │   ├── prompt_builder.py              # System prompt + [doc-N] citation rules
  │   ├── citation_enforcer.py           # Validate + strip hallucinated citations
  │   ├── answer_validator.py            # Hallucination guard (lexical / LLM)
  │   ├── llm_client.py                  # Cached LLM factory (OpenAI / Anthropic)
  │   ├── streaming_handler.py           # SSE token streaming
  │   └── response_schema.py             # Pydantic request/response models
  │
  ├── api/
  │   ├── app.py                         # FastAPI factory + startup hooks
  │   ├── routes/
  │   │   ├── query.py                   # POST /query, /query/stream, /session/*
  │   │   └── health.py                  # GET /health (liveness probe)
  │   ├── middleware/
  │   │   └── __init__.py                # Tracing, rate-limiting, API key auth
  │   ├── session_manager.py             # Redis-backed chat history (per session_id)
  │   └── cache.py                       # Two-layer semantic cache (SHA + cosine)
  │
  ├── evaluation/
  │   ├── ragas_runner.py                # Faithfulness, recall, relevancy, precision
  │   ├── latency_bench.py               # p50/p95/p99 retrieval + generation latency
  │   ├── chunk_quality.py               # Keyword coverage + chunk coherence scoring
  │   └── eval_dataset.json              # Q&A pairs for evaluation
  │
  └── observability/
      ├── metrics.py                     # Prometheus metrics for the RAG system.
      └── tracer.py                      # LangSmith run tracing + feedback
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Docker (for Redis)
- Pinecone account (serverless index)
- Cohere API key
- Groq API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
# LLM
GROQ_API_KEY=sk-...
LLM_MODEL= "choose any model available in groq"

# Embeddings
HUGGINGFACEHUB_API_TOKEN=hf_...
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=768

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX=hybrid-rag
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Cohere (reranker)
COHERE_API_KEY=...
RERANKER_MODEL=rerank-english-v3.0

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_SECRET_KEY=your-secret-key

# LangSmith (optional)
LANGSMITH_API_KEY=...
LANGCHAIN_PROJECT=hybrid-rag
LANGCHAIN_TRACING_V2=true
```

### 4. Start Redis

```bash
docker compose up -d
```

### 5. Ingest your documents

Place your documents in a folder (PDF, DOCX, HTML, Markdown supported) and run:

```bash
python ingest.py
```

This will:
- Load and chunk all documents
- Deduplicate via SimHash
- Embed using HuggingFace
- Upsert to Pinecone sparse-dense index
- Fit and save `bm25_encoder.json`

### 6. Start the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Start the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | RAG query with session history + semantic cache |
| `POST` | `/api/v1/query/stream` | SSE streaming RAG query |
| `GET` | `/api/v1/session/{id}` | Retrieve chat history |
| `DELETE` | `/api/v1/session/{id}` | Clear chat session |
| `GET` | `/health` | Liveness + readiness probe |

### Example query

```bash
curl -X POST http://localhost:8000/api/v1/query \
     -H "Authorization: Bearer your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the refund policy?"}'
```

```json
{
  "answer": "Refunds are processed within 7 business days of the request. [doc-2]",
  "sources": [
    {
      "doc_index": 2,
      "title": "policy.pdf",
      "content": "Refund requests are reviewed and processed...",
      "score": 0.923
    }
  ],
  "is_fully_grounded": true,
  "latency_ms": 1240,
  "model": "openai/gpt-oss-120b"
}
```

---

## Retrieval Pipeline

The system uses **Pinecone native hybrid search** 

```
query
  │
  ├─ HyDE expansion        → generate hypothetical answer for better dense recall
  │
  ├─ Pinecone hybrid query → single API call combining:
  │     dense vector  (alpha = 0.5)   semantic similarity
  │     sparse vector (1 - alpha)     BM25 keyword matching
  │
  ├─ Cohere reranker       → re-score top-20 candidates → keep top-5
  │
  └─ Context compressor    → prune low-signal sentences before generation
```

Adjust `HYBRID_ALPHA` in `.env` to tune the dense/sparse blend:
- `0.7` → more semantic (better for conceptual questions)
- `0.3` → more keyword (better for exact term lookup)

---

## Citation Enforcement

Every answer is grounded with `[doc-N]` bracket citations. The system:

1. Numbers each retrieved chunk as `[doc-1]`, `[doc-2]`, etc. in the prompt
2. Instructs the LLM to cite every factual claim
3. Strips any `[doc-N]` references to non-existent documents
4. Validates that every sentence with >6 words carries at least one citation
5. Returns `is_fully_grounded: true/false` in the response

---

## Evaluation

Run the full local evaluation suite:

```bash
python -m src.evaluation
```

Or individually:

```bash
python -m src.evaluation.chunk_quality      # no LLM calls, fastest
python -m src.evaluation.latency_bench --runs 30
python -m src.evaluation.ragas_runner
```

Results are saved to `results/`:

| File | Contents |
|---|---|
| `ragas_latest.json` | Faithfulness, answer relevancy, context recall, context precision |
| `latency.json` | p50/p95/p99 for retrieval, generation, e2e |
| `chunk_quality.json` | Keyword coverage + coherence per golden question |
| `summary.json` | Combined pass/fail across all metrics |

### Evaluation thresholds (configurable in `.env`)

| Metric | Default threshold |
|---|---|
| Faithfulness | 0.80 |
| Answer Relevancy | 0.80 |
| Context Recall | 0.75 |

---

## Configuration Reference

All settings live in `config.py` and are loaded from `.env`.

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-oss-120b` | LLM model name |
| `EMBEDDING_DIM` | `768` | Embedding dimension |
| `RETRIEVER_TOP_K` | `20` | Candidates before reranking |
| `RERANKER_TOP_N` | `5` | Docs passed to generator |
| `HYBRID_ALPHA` | `0.5` | Dense/sparse blend (1.0 = pure dense) |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity for cache hit |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit per key |

---

## Key Design Decisions

**No ingestion at runtime** — documents are indexed once via `ingest.py`. The API exposes only query endpoints, preventing any document mutation at runtime.

**Single Pinecone index for hybrid search** — BM25 sparse vectors and dense embedding vectors live in the same Pinecone index. One API call returns a fused result, replacing the previous Elasticsearch + Qdrant + RRF architecture.

**BM25 encoder fitted on your corpus** — after ingestion, `bm25_encoder.json` is saved to disk. `VectorRetriever` loads it on startup so sparse vectors at query time use your domain vocabulary, not generic MS MARCO weights.

**Citation enforcement is structural** — the prompt template hard-codes citation rules, `citation_enforcer.py` validates every sentence post-generation, and `answer_validator.py` runs a secondary hallucination check. `is_fully_grounded` in the response tells you the result.

**Semantic cache has two layers** — exact SHA-256 match (zero embedding cost) and cosine-similarity match (catches paraphrased repeats). Cache is flushed after every ingestion run.

---

### Author

