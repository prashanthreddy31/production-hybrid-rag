"""
api/
~~~~
FastAPI service layer for the Ask-My-Docs RAG system.

Structure:
    app.py               — FastAPI application factory + uvicorn entrypoint
    routes/
        query.py         — POST /api/v1/query, /query/stream, /ingest/*, /session/*
        health.py        — GET /health (liveness + readiness)
    middleware/
        __init__.py      — RequestTracingMiddleware, RateLimitMiddleware, APIKeyAuthMiddleware
    session_manager.py   — Redis-backed chat history per session_id
    cache.py             — Two-layer semantic cache (exact SHA + embedding cosine)

Start the server::

    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from src.api.app import app, create_app
from src.api.session_manager import SessionManager
from src.api.cache import SemanticCache

__all__ = ["app", "create_app", "SessionManager", "SemanticCache"]