"""
Streaming handler — SSE chunk forwarding with citation-aware post-processing.

Provides:
  StreamingHandler   — wraps the LLM streaming call, buffers the full
                       response, then runs citation validation on completion.
  stream_rag_response — convenience async generator for FastAPI SSE routes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

import structlog

from config import get_settings
from src.generation.llm_client import get_llm
from src.generation.citation_enforcer import validate_citations, strip_hallucinated_citations

log = structlog.get_logger(__name__)
settings = get_settings()


# ── SSE event helpers ─────────────────────────────────────────────────────────

def _text_event(data: str) -> str:
    """Format a plain text chunk as an SSE event."""
    return f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"


def _meta_event(payload: dict) -> str:
    """Format a metadata event (sent after streaming completes)."""
    return f"data: {json.dumps({'type': 'meta', **payload})}\n\n"


def _done_event() -> str:
    return "data: [DONE]\n\n"


# ── Streaming handler ─────────────────────────────────────────────────────────

@dataclass
class StreamingResult:
    full_text: str
    is_fully_grounded: bool
    cited_indices: list[int]
    elapsed_ms: int
    tokens_streamed: int


class StreamingHandler:
    """
    Streams LLM tokens to the caller while buffering for post-stream validation.

    Usage::

        handler = StreamingHandler()
        for chunk in handler.stream_sync(messages, docs):
            yield chunk   # SSE-formatted string
    """

    def __init__(self) -> None:
        self._llm = get_llm(streaming=True)

    # ── Sync streaming (for sync FastAPI endpoints / tests) ───────────────────

    def stream_sync(
        self,
        messages: list[BaseMessage],
        docs: list[Document],
    ) -> Iterator[str]:
        """
        Yield SSE-formatted strings.

        Yields token events during generation, then a single meta event
        with citation/grounding info once the stream is complete.
        """
        t0 = time.perf_counter()
        buffer: list[str] = []

        for chunk in self._llm.stream(messages):
            content = getattr(chunk, "content", "") or ""
            if content:
                buffer.append(content)
                yield _text_event(content)

        full_text = "".join(buffer)
        cleaned = strip_hallucinated_citations(full_text, len(docs))
        citation_result = validate_citations(cleaned, docs)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "stream_complete",
            tokens=len(buffer),
            grounded=citation_result.is_fully_grounded,
            elapsed_ms=elapsed_ms,
        )

        yield _meta_event(
            {
                "is_fully_grounded": citation_result.is_fully_grounded,
                "cited_indices": citation_result.cited_indices,
                "uncited_count": len(citation_result.uncited_sentences),
                "elapsed_ms": elapsed_ms,
            }
        )
        yield _done_event()

    # ── Async streaming (for async FastAPI endpoints) ─────────────────────────

    async def stream_async(
        self,
        messages: list[BaseMessage],
        docs: list[Document],
    ) -> AsyncIterator[str]:
        """Async version of stream_sync for use with FastAPI async routes."""
        t0 = time.perf_counter()
        buffer: list[str] = []

        async for chunk in self._llm.astream(messages):
            content = getattr(chunk, "content", "") or ""
            if content:
                buffer.append(content)
                yield _text_event(content)

        full_text = "".join(buffer)
        cleaned = strip_hallucinated_citations(full_text, len(docs))
        citation_result = validate_citations(cleaned, docs)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "async_stream_complete",
            tokens=len(buffer),
            grounded=citation_result.is_fully_grounded,
            elapsed_ms=elapsed_ms,
        )

        yield _meta_event(
            {
                "is_fully_grounded": citation_result.is_fully_grounded,
                "cited_indices": citation_result.cited_indices,
                "uncited_count": len(citation_result.uncited_sentences),
                "elapsed_ms": elapsed_ms,
            }
        )
        yield _done_event()


# ── Convenience wrapper used by API routes ────────────────────────────────────

def stream_rag_response(
    messages: list[BaseMessage],
    docs: list[Document],
) -> Iterator[str]:
    """
    Thin convenience wrapper — creates a handler and starts a sync stream.

    Use in FastAPI with::

        return StreamingResponse(
            stream_rag_response(messages, docs),
            media_type="text/event-stream",
        )
    """
    handler = StreamingHandler()
    yield from handler.stream_sync(messages, docs)