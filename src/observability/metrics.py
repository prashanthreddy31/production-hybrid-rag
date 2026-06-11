"""
observability/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~
Prometheus metrics for the RAG system.

Metrics exposed at GET /metrics (added to FastAPI by calling setup_metrics(app)).

Tracked:
    rag_query_total          — total queries (labelled by status: success/error)
    rag_query_latency_ms     — e2e latency histogram
    rag_retrieval_latency_ms — retrieval-only latency histogram
    rag_generation_latency_ms— generation-only latency histogram
    rag_reranker_score       — top reranker score distribution
    rag_cache_hits_total     — semantic cache hit counter
    rag_grounded_total       — grounded vs ungrounded answer counter
    rag_tokens_used_total    — estimated token spend (prompt + completion)

Usage::

    from observability.metrics import (
        record_query, record_cache_hit, record_grounding, setup_metrics
    )

    setup_metrics(app)        # call once in app.py

    record_query(
        latency_ms=320,
        retrieval_ms=180,
        generation_ms=140,
        reranker_score=0.87,
        status="success",
    )
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI

log = structlog.get_logger(__name__)

# ── Metric definitions ────────────────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        make_asgi_app, REGISTRY,
    )

    _query_total = Counter(
        "rag_query_total",
        "Total RAG queries",
        ["status"],           # success | error
    )
    _query_latency = Histogram(
        "rag_query_latency_ms",
        "End-to-end query latency in ms",
        buckets=[50, 100, 200, 400, 800, 1500, 3000, 6000],
    )
    _retrieval_latency = Histogram(
        "rag_retrieval_latency_ms",
        "Retrieval phase latency in ms",
        buckets=[25, 50, 100, 200, 500, 1000, 2000],
    )
    _generation_latency = Histogram(
        "rag_generation_latency_ms",
        "Generation phase latency in ms",
        buckets=[50, 100, 250, 500, 1000, 2000, 4000],
    )
    _reranker_score = Histogram(
        "rag_reranker_score",
        "Top reranker score distribution",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    _cache_hits = Counter(
        "rag_cache_hits_total",
        "Semantic cache hits",
        ["layer"],            # exact | semantic
    )
    _grounded = Counter(
        "rag_grounded_total",
        "Grounded vs ungrounded answers",
        ["grounded"],         # true | false
    )
    _tokens = Counter(
        "rag_tokens_used_total",
        "Estimated tokens consumed",
        ["type"],             # prompt | completion
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_not_installed", hint="pip install prometheus-client")


# ── Public API ────────────────────────────────────────────────────────────────

def setup_metrics(app: FastAPI) -> None:
    """Mount the /metrics endpoint on *app*."""
    if not _PROMETHEUS_AVAILABLE:
        return
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    log.info("prometheus_metrics_mounted", path="/metrics")


def record_query(
    latency_ms: int,
    retrieval_ms: int = 0,
    generation_ms: int = 0,
    reranker_score: float | None = None,
    status: str = "success",
) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _query_total.labels(status=status).inc()
    _query_latency.observe(latency_ms)
    if retrieval_ms:
        _retrieval_latency.observe(retrieval_ms)
    if generation_ms:
        _generation_latency.observe(generation_ms)
    if reranker_score is not None:
        _reranker_score.observe(reranker_score)


def record_cache_hit(layer: str = "semantic") -> None:
    """layer: 'exact' or 'semantic'"""
    if not _PROMETHEUS_AVAILABLE:
        return
    _cache_hits.labels(layer=layer).inc()


def record_grounding(is_grounded: bool) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _grounded.labels(grounded=str(is_grounded).lower()).inc()


def record_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _tokens.labels(type="prompt").inc(prompt_tokens)
    _tokens.labels(type="completion").inc(completion_tokens)