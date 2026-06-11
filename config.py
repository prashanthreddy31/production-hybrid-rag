"""Central configuration loaded from environment / .env file."""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: SecretStr = os.getenv("groq_api_key")
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_dim: int = 768
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    HF_token: SecretStr = os.getenv("HF_token")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_top_k: int = 20          # candidates before rerank
    cohere_api_key: SecretStr = os.getenv("cohere_api_key")
    reranker_top_n: int = 5            # docs passed to generator
    reranker_model: str = "rerank-english-v3.0"
    hybrid_alpha: float = 0.6
    bm25_encoder_path: str = "bm25_encoder.json"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 40

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: SecretStr = os.getenv("pc_api_key")
    pinecone_index: str = "research-papers"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"       
    pinecone_region: str = "us-east-1"  


    # ── Redis (session / semantic cache) ─────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── API ───────────────────────────────────────────────────────────────────
    api_secret_key: SecretStr = "hybrid-rag-secret"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    rate_limit_per_minute: int = 60

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_api_key: SecretStr = os.getenv("langsmith_api_key")
    langsmith_project: str = "hybrid-rag"
    langchain_tracing_v2: bool = True
    log_level: str = "INFO"

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_faithfulness_threshold: float = 0.80
    eval_context_recall_threshold: float = 0.75
    eval_answer_relevancy_threshold: float = 0.80
    evaluation_dataset_path: str = "src/evaluation/evaluation_dataset.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()