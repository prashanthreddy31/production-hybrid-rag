"""
FastAPI route handlers:
  POST /query           — single-turn or multi-turn RAG query (with semantic cache)
  POST /query/stream    — SSE streaming RAG query
  DELETE /session/{id}  — clear a chat session
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document

import structlog

from config import get_settings
from src.generation.Rag_chain import RAGChain
from src.generation.response_schema import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
)
from src.generation.prompt_builder import build_rag_messages
from src.generation.streaming_handler import stream_rag_response
from src.retrieval.Retrieval_pipeline import RetrievalPipeline
from src.api.session_manager import SessionManager
from src.api.cache import SemanticCache

log = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter()


# ── Dependency singletons (created once at first request) ─────────────────────

_rag_chain:        RAGChain | None          = None
_retrieval:        RetrievalPipeline | None = None
_session_manager:  SessionManager | None    = None
_cache:            SemanticCache | None     = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def get_retrieval() -> RetrievalPipeline:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalPipeline()
    return _retrieval


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(
    req: QueryRequest,
    chain: RAGChain = Depends(get_rag_chain),
    sessions: SessionManager = Depends(get_session_manager),
    cache: SemanticCache = Depends(get_cache),
) -> QueryResponse:
    """
    Run a RAG query with optional session history and semantic caching.

    - If the query is in the semantic cache (cosine similarity ≥ threshold),
      the cached response is returned immediately.
    - Otherwise the full retrieval → rerank → generate → validate pipeline runs.
    - The response is written to cache and chat history is updated.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # 1. Semantic cache lookup (skip for session queries — history changes meaning)
    if not req.session_id:
        cached = cache.get(req.question)
        if cached:
            cached["session_id"] = session_id
            return QueryResponse(**cached)

    # 2. Load session history
    history = sessions.get_history(session_id) if req.session_id else None

    # 3. Run RAG chain
    try:
        response = chain.query(
            question=req.question,
            session_id=session_id,
            chat_history=history,
            filter_metadata=req.filter_metadata,
        )
    except Exception as exc:
        log.error("query_error", error=str(exc), question=req.question[:60])
        raise HTTPException(status_code=500, detail="RAG query failed") from exc

    # 4. Persist to session history
    sessions.append(session_id, role="user",      content=req.question)
    sessions.append(session_id, role="assistant", content=response.answer)
    response.session_id = session_id

    # 5. Cache (only stateless queries)
    if not req.session_id:
        cache.set(req.question, response.model_dump())

    return response


# ── Streaming query ───────────────────────────────────────────────────────────

@router.post("/query/stream", tags=["rag"])
async def query_stream(
    req: QueryRequest,
    retrieval: RetrievalPipeline = Depends(get_retrieval),
) -> StreamingResponse:
    """
    SSE streaming RAG query.

    Yields ``data: {"type":"token","content":"..."}`` events during generation,
    then a final ``data: {"type":"meta", ...}`` event with citation/grounding info.
    """
    try:
        docs = retrieval.retrieve(
            question=req.question,
            filter_metadata=req.filter_metadata,
        )
    except Exception as exc:
        log.error("stream_retrieval_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc

    messages = build_rag_messages(req.question, docs)

    return StreamingResponse(
        stream_rag_response(messages, docs),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Session management ────────────────────────────────────────────────────────

@router.delete("/session/{session_id}", tags=["session"])
async def clear_session(
    session_id: str,
    sessions: SessionManager = Depends(get_session_manager),
) -> dict:
    """Clear the chat history for *session_id*."""
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    sessions.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/session/{session_id}", tags=["session"])
async def get_session(
    session_id: str,
    sessions: SessionManager = Depends(get_session_manager),
) -> dict:
    """Retrieve the full chat history for *session_id*."""
    history = sessions.get_history(session_id)
    return {"session_id": session_id, "turns": len(history), "history": history}