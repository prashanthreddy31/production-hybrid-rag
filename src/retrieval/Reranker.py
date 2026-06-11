"""Cohere reranker"""
from __future__ import annotations

from functools import lru_cache

import cohere
from langchain_core.documents import Document

import structlog
from config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

@lru_cache(maxsize=1)
def _get_cohere_client() -> cohere.Client:
    log.info("cohere_client_init", model=settings.reranker_model)
    return cohere.Client(api_key=settings.cohere_api_key.get_secret_value())

class CrossEncoderReranker:
    """Rerank retrieved documents using the Cohere Rerank API."""

    def __init__(
            self,
            model: str = settings.reranker_model,
            top_n: int = settings.reranker_top_n
    ) -> None:
        self.model = model
        self.top_n = top_n
        self._client = _get_cohere_client()

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        
        # Cohere expects a flat list of strings
        passages = [d.page_content for d in docs]

        response = self._client.rerank(
            model = self.model,
            query= query,
            documents= passages,
            top_n= self.top_n,
            return_documents=False,
        )

        results = []
        for rank, hit in enumerate(response.results, start=1):
            doc = docs[hit.index]
            doc.metadata["reranker_score"] = round(hit.relevance_score, 4)
            doc.metadata["reranker_rank"] = rank
            results.append(doc)

        log.debug(
            "reranker_done",
            query=query[:60],
            input_docs=len(docs),
            output_docs=len(results),
            top_score=results[0].metadata["reranker_score"] if results else None,
        )
        return results