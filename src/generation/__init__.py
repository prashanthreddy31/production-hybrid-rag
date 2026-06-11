"""
generation/
~~~~~~~~~~~
Generation package for the Ask-My-Docs RAG system.

Public surface:
    RAGChain            — full generation pipeline (prompt → LLM → validate → respond)
    get_llm             — cached LLM instance factory (OpenAI / Anthropic)
    build_rag_messages  — assemble system + context + chat-history messages
    build_context_block — format retrieved docs into numbered [doc-N] context
    validate_citations  — check every sentence carries a [doc-N] bracket
    strip_hallucinated_citations — remove out-of-range [doc-N] references
    validate_answer     — hallucination guard (lexical or LLM mode)
    StreamingHandler    — SSE token streaming with post-stream citation check
    stream_rag_response — convenience wrapper for FastAPI streaming routes

Schemas:
    QueryRequest, QueryResponse, SourceDocument
    IngestRequest, IngestResponse, HealthResponse

Typical usage::

    from generation import RAGChain

    chain = RAGChain()
    response = chain.query("What is the refund policy?")
    print(response.answer)
    # "Refunds are processed within 7 business days. [doc-2]"

For streaming::

    from generation import stream_rag_response
    from generation.prompt_builder import build_rag_messages

    messages = build_rag_messages(question, docs)
    return StreamingResponse(
        stream_rag_response(messages, docs),
        media_type="text/event-stream",
    )
"""

from src.generation.Rag_chain import RAGChain
from src.generation.llm_client import get_llm
from src.generation.prompt_builder import build_rag_messages
from src.generation.citation_enforcer import (
    build_context_block,
    validate_citations,
    strip_hallucinated_citations,
    CitationResult,
)
from src.generation.answer_validator import validate_answer, ValidationResult
from src.generation.streaming_handler import StreamingHandler, stream_rag_response
from src.generation.response_schema import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)

__all__ = [
    # Core
    "RAGChain",
    "get_llm",
    # Prompt
    "build_rag_messages",
    "build_context_block",
    # Citations
    "validate_citations",
    "strip_hallucinated_citations",
    "CitationResult",
    # Validation
    "validate_answer",
    "ValidationResult",
    # Streaming
    "StreamingHandler",
    "stream_rag_response",
    # Schemas
    "QueryRequest",
    "QueryResponse",
    "SourceDocument",
    "IngestRequest",
    "IngestResponse",
    "HealthResponse",
]