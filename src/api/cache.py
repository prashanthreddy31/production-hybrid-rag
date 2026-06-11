"""
Semantic query cache — Redis-backed, embedding-similarity keyed.

On a cache HIT  → return the stored QueryResponse immediately (no LLM call).
On a cache MISS → caller runs the RAG chain, then stores the result.

Similarity is computed as cosine distance between the incoming query
embedding and every cached query embedding for this namespace.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import redis
import structlog

from config import get_settings
from src.ingestion.embedders import get_embeddings

log = structlog.get_logger(__name__)
settings = get_settings()

_CACHE_NS = "semcache"          # Redis key namespace
_EMB_NS   = "semcache:emb"     # stores raw embeddings
_INDEX_KEY = "semcache:index"  # sorted set: score → cache_key


@dataclass
class CacheEntry:
    question: str
    response_json: str       # serialised QueryResponse
    created_at: float


class SemanticCache:
    """
    Two-layer cache:
      1. Exact SHA-256 match  — zero-cost lookup, handles identical queries.
      2. Embedding similarity — cosine distance, catches paraphrased queries.

    Both layers share the same TTL from settings.cache_ttl_seconds.
    """

    def __init__(self) -> None:
        self._redis = redis.from_url(
            settings.redis_url,
            decode_responses=False,      # we store binary embeddings
            socket_connect_timeout=2,
        )
        self._ttl = settings.cache_ttl_seconds
        self._threshold = settings.semantic_cache_threshold
        self._embeddings = get_embeddings()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, question: str) -> dict | None:
        """
        Return cached response dict or None.
        Checks exact hash first, then embedding similarity.
        """
        try:
            # 1. Exact match
            exact_key = self._exact_key(question)
            raw = self._redis.get(exact_key)
            if raw:
                log.debug("cache_exact_hit", question=question[:60])
                return json.loads(raw)

            # 2. Semantic match
            return self._semantic_get(question)

        except (redis.RedisError, Exception) as exc:
            log.warning("cache_get_error", error=str(exc))
            return None

    def set(self, question: str, response: dict) -> None:
        """Store *response* keyed by both exact hash and embedding."""
        try:
            payload = json.dumps(response).encode()
            ttl = self._ttl

            # Exact key
            self._redis.set(self._exact_key(question), payload, ex=ttl)

            # Embedding key
            emb = self._embed(question)
            emb_key = f"{_EMB_NS}:{self._exact_key(question)}"
            entry = json.dumps({
                "question": question,
                "response": response,
                "created_at": time.time(),
                "embedding": emb.tolist(),
            }).encode()
            self._redis.set(emb_key, entry, ex=ttl)

            log.debug("cache_set", question=question[:60])
        except (redis.RedisError, Exception) as exc:
            log.warning("cache_set_error", error=str(exc))

    def invalidate(self, question: str) -> None:
        try:
            self._redis.delete(self._exact_key(question))
            log.info("cache_invalidated", question=question[:60])
        except redis.RedisError:
            pass

    def flush(self) -> int:
        """Delete all cache keys — use with care."""
        try:
            keys = list(self._redis.scan_iter(f"{_CACHE_NS}:*")) + \
                   list(self._redis.scan_iter(f"{_EMB_NS}:*"))
            if keys:
                self._redis.delete(*keys)
            log.info("cache_flushed", deleted=len(keys))
            return len(keys)
        except redis.RedisError:
            return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _semantic_get(self, question: str) -> dict | None:
        """Scan embedding keys and return best cosine-similar match."""
        query_emb = self._embed(question)

        best_score = -1.0
        best_response: dict | None = None

        for key in self._redis.scan_iter(f"{_EMB_NS}:*"):
            raw = self._redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                stored_emb = np.array(entry["embedding"], dtype=np.float32)
                score = float(self._cosine(query_emb, stored_emb))
                if score > best_score:
                    best_score = score
                    best_response = entry["response"]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        if best_score >= self._threshold and best_response is not None:
            log.info("cache_semantic_hit", score=round(best_score, 4), question=question[:60])
            return best_response

        log.debug("cache_miss", question=question[:60], best_score=round(best_score, 4))
        return None

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embeddings.embed_query(text)
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _exact_key(question: str) -> str:
        h = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
        return f"{_CACHE_NS}:{h}"