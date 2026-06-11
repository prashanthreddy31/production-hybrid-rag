"""
Retrieval orchestrator — Pinecone native hybrid (dense + sparse BM25).
"""

from __future__ import annotations

import time
from typing import Literal

from transformers import AutoTokenizer
from langchain_core.documents import Document

import structlog

from config import get_settings
from src.retrieval.Hybrid_retriever import HybridRetriever      
from src.retrieval.Reranker import CrossEncoderReranker
from src.retrieval.Query_expander import hyde_expander, multi_query_expand

log = structlog.get_logger(__name__)
settings = get_settings()
 
ExpandStrategy = Literal["none", "hyde", "multi_query"]

class RetrievalPipeline:
    """
    End-to-end retrieval using Pinecone native hybrid search.
 
        query
          ├─ expand (HyDE / multi-query / none)
          ├─ Pinecone hybrid query  ←  dense + sparse BM25 fused server-side
          └─ Cohere rerank
    """
    def __init__(
        self,
        expand_strategy: ExpandStrategy = "hyde"
    ) -> None:
        self.expand_strategy = expand_strategy
        self._retriever = HybridRetriever()
        self._reranker = CrossEncoderReranker()
        self._tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        
    def retrieve(
        self,
        question: str,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """
        Run the full hybrid retrieval pipeline.
 
        Args:
            question:        Raw user question.
            filter_metadata: Pinecone metadata filter dict (supports $eq/$in/$gte etc).
            expand_strategy: Override the instance-level expansion strategy.
 
        Returns:
            Reranked (and optionally compressed) list of Documents.
        """
        t0 = time.perf_counter()
        strategy = self.expand_strategy

        # 1. Query expansion
        queries = self._expand(question, strategy)
        log.debug("queries_expanded", strategy=strategy, queries=queries)

        # 2. Retrieve for all query variants and merge with dedup
        seen_ids: set[str] = set() 
        hybrid_docs: list[Document] = []

        for q in queries:
            # Critical for HyDE where hypothetical doc can be very long
            q = self._truncate_to_tokens(q, max_tokens=400)

            for doc in self._retriever.retrieve(q, filter_metadata):
                key = doc.metadata.get("chunk_id", doc.page_content[:64])
                if key not in seen_ids:
                    hybrid_docs.append(doc)
                    seen_ids.add(key)
        log.info(
            "hybrid_docs_merged",
            strategy=strategy,
            total_queries=len(queries),
            unique_docs=len(hybrid_docs),
        )

        # 3. Cohere rerank
        reranked = self._reranker.rerank(question, hybrid_docs)

        elapsed_ms = int((time.perf_counter()- t0) * 1000)
        log.info(
            "retrieval_pipeline_done",
            question=question[:80],
            expand=strategy,
            hybrid_retrieved=len(hybrid_docs),
            reranked=len(reranked),
            final=len(hybrid_docs),
            elapsed_ms=elapsed_ms,
        )
        return reranked
    
    def _expand(self, question: str, strategy: ExpandStrategy) -> list[str]:
        if strategy == "hyde":
            hypothetical = hyde_expander(question)
            return [question, hypothetical]
        
        if strategy == "multi_query":
            return multi_query_expand(question, n=3)
        return [question]

    def _truncate_to_tokens(self, text: str, max_tokens: int = 400) -> str:
        """
        Truncate text to max_tokens using the embedding model tokenizer.
        Critical for HyDE hypothetical docs which can exceed 512 token limit.
        """
        tokens = self._tokenizer.encode(
            text,
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False,
        )
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

        
