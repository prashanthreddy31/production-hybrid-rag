"""
observability/tracer.py
~~~~~~~~~~~~~~~~~~~~~~~~
LangSmith tracing integration.

Tracing is enabled automatically at API startup via app.py when
LANGSMITH_API_KEY is set in .env. This module provides:

    - get_tracer()         — returns a configured LangSmith tracer
    - trace_rag_query()    — context manager to wrap a full RAG query trace
    - log_feedback()       — record thumbs-up/down feedback against a run

Usage::

    from observability.tracer import trace_rag_query, log_feedback

    with trace_rag_query(question, session_id) as run_id:
        response = chain.query(question)

    log_feedback(run_id, score=1, comment="Great answer")
"""
from __future__ import annotations

import contextlib
from typing import Generator

import structlog

from config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


def is_tracing_enabled() -> bool:
    return bool(settings.langsmith_api_key.get_secret_value())


@contextlib.contextmanager
def trace_rag_query(
    question: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> Generator[str | None, None, None]:
    """
    Context manager that wraps a RAG query in a LangSmith trace.

    Yields the run_id so you can attach feedback after the query completes.
    Falls back to a no-op if tracing is not configured.

    Example::

        with trace_rag_query(question, session_id) as run_id:
            response = chain.query(question)
        log_feedback(run_id, score=1)
    """
    if not is_tracing_enabled():
        yield None
        return

    try:
        from langsmith import Client, traceable
        client = Client(api_key=settings.langsmith_api_key.get_secret_value())

        extra_meta = {
            "question":   question,
            "session_id": session_id or "none",
            "llm_model":  settings.llm_model,
            **(metadata or {}),
        }

        run_id = None
        try:
            with client.trace(
                name="rag_query",
                project_name=settings.langsmith_project,
                metadata=extra_meta,
            ) as run:
                run_id = str(run.id) if run else None
                log.debug("trace_started", run_id=run_id, question=question[:60])
                yield run_id
        except Exception as exc:
            log.warning("trace_error", error=str(exc))
            yield run_id

    except ImportError:
        log.warning("langsmith_not_installed", hint="pip install langsmith")
        yield None


def log_feedback(
    run_id: str | None,
    score: int,                  # 1 = thumbs up, 0 = thumbs down
    comment: str | None = None,
    key: str = "user_feedback",
) -> None:
    """
    Attach user feedback to a LangSmith run.

    Args:
        run_id:  The run ID returned by trace_rag_query.
        score:   1 for positive, 0 for negative.
        comment: Optional free-text comment.
        key:     Feedback key (default: "user_feedback").
    """
    if not run_id or not is_tracing_enabled():
        return

    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        log.info("feedback_logged", run_id=run_id, score=score)
    except Exception as exc:
        log.warning("feedback_error", run_id=run_id, error=str(exc))