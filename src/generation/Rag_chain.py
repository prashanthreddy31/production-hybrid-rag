"""
Core RAG chain.
 
Query flow:
  expand query
    → Hybrid retrieve 
        → cross-encoder rerank
          → build prompt
            → LLM generate
              → validate citations
                → return response
"""
from __future__ import annotations
 
import time
from typing import AsyncIterator, Iterator
 
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
 
import structlog
from config import get_settings
from src.retrieval.Retrieval_pipeline import RetrievalPipeline
from src.generation.llm_client import get_llm
from src.generation.prompt_builder import build_rag_messages
from src.generation.citation_enforcer import validate_citations, strip_hallucinated_citations, CitationResult
from src.generation.response_schema import QueryResponse, SourceDocument

log = structlog.get_logger(__name__)
settings = get_settings()

class RAGChain:
    """Production RAG chain: hybrid retrieval + reranking + citation enforcement."""

    def __init__(self) -> None:
        self.retrieval = RetrievalPipeline()
        self.llm = get_llm()
        self.streaming_llm = get_llm(streaming=True)

    def query(
            self,
            question: str,
            session_id: str | None = None,
            chat_history: list[dict] | None = None,
            filter_metadata: dict | None = None,
    ) -> QueryResponse:
        t0 = time.perf_counter()

        docs = self.retrieval.retrieve(
            question=question, 
            filter_metadata=filter_metadata
        )
        answer, citation_result = self._generate(question, docs, chat_history)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info("rag_query_done", question=question[:60], latency_ms=latency_ms)

        return self._build_response(
            answer, citation_result, docs, session_id, latency_ms
        )
    
    def stream(
            self,
            question: str,
            chat_history: list[dict] | None = None,
            filter_metadata: dict | None = None,
    ) -> Iterator[str]:
        docs = self.retrieval.retrieve(
            question = question,
            filter_metadata = filter_metadata,
        )
        messages = build_rag_messages(question, docs, chat_history)

        for chunk in self.streaming_llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield str(chunk.content)


    def _generate(
            self,
            question: str,
            docs,
            chat_history: list[dict] | None,
    ) -> tuple[str, CitationResult]:
        messages = build_rag_messages(question, docs, chat_history)
        raw_answer: str = (self.llm | StrOutputParser()).invoke(messages)
        cleaned = strip_hallucinated_citations(raw_answer, len(docs))
        citation_result = validate_citations(cleaned, docs)
        return cleaned, citation_result
    
    def _build_response(
        self,
        answer: str,
        citation_result: CitationResult,
        docs,
        session_id: str | None,
        latency_ms: int,
    ) -> QueryResponse:
        sources = [
            SourceDocument(
                content=doc.page_content[:400],
                source=doc.metadata.get("source", ""),
                title=doc.metadata.get("title"),
                page=doc.metadata.get("page"),
                score=doc.metadata.get("reranker_score"),
                doc_index=i + 1,
            )
            for i, doc in enumerate(docs)
        ]
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            is_fully_grounded=citation_result.is_fully_grounded,
            latency_ms=latency_ms,
            model=settings.llm_model,
        )