"""
Health check route — /health

Returns liveness + readiness status for all critical dependencies:
  • Pinecone vector store
  • Redis (session store + cache)
  • LLM provider reachability
"""
from __future__ import annotations

import time

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import get_settings
from src.generation.response_schema import HealthResponse

log = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter()


# ── Dependency probes ─────────────────────────────────────────────────────────

def _check_pinecone() -> str:
    try:
        from pinecone import Pinecone
        client = Pinecone(
            api_key=settings.pinecone_api_key.get_secret_value(),
        )
        client.list_indexes()
        return "ok"
    except Exception as exc:
        log.warning("health_pinecone_fail", error=str(exc))
        return f"error: {exc}"


def _check_redis() -> str:
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        return "ok"
    except Exception as exc:
        log.warning("health_redis_fail", error=str(exc))
        return f"error: {exc}"


def _check_llm() -> str:
    """Lightweight model reachability check — no actual LLM call made."""
    try:
        from groq import Groq
        Groq(
            api_key=settings.groq_api_key.get_secret_value()
        ).models.list()
        return "ok"       
    except Exception as exc:
        log.warning("health_llm_fail", error=str(exc))
        return f"error: {exc}"


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> JSONResponse:
    """Liveness + readiness probe for all critical dependencies."""
    t0 = time.perf_counter()

    checks = {
        "pinecone":        _check_pinecone(),
        "redis":         _check_redis(),
        "llm":           _check_llm(),
    }

    all_ok = all(v == "ok" for v in checks.values())
    status = "healthy" if all_ok else "degraded"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    log.info("health_check", status=status, elapsed_ms=elapsed_ms, **checks)

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content=HealthResponse(
            status=status,
            checks=checks,
        ).model_dump(),
    )