"""
Pinecone hybrid retriever — dense + sparse (BM25)

    score = alpha * dense_score + (1 - alpha) * sparse_score
 
alpha = 1.0  →  pure dense (semantic)
alpha = 0.0  →  pure sparse (BM25 keyword)
alpha = 0.5  →  equal blend  (good default)
"""
from __future__ import annotations

from functools import cached_property

from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

import structlog

from config import get_settings
from src.ingestion.embedders import get_embeddings, EmbeddingProvider

log = structlog.get_logger(__name__)
settings = get_settings()

class HybridRetriever:
    """Hybrid retriever over a Pinecone sparse-dense index."""

    def __init__(
            self,
            embedding_provider: EmbeddingProvider = "huggingface",
            top_k: int= settings.retriever_top_k,
            alpha: float = settings.hybrid_alpha,
    ) -> None:
        self.top_k = top_k
        self.alpha = alpha
        self._embeddings = get_embeddings(embedding_provider)
        self.pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self._index = self.pc.Index(settings.pinecone_index)
        encoder_path = Path(settings.bm25_encoder_path)
        if encoder_path.exists():
            self._bm25 = BM25Encoder().load(str(encoder_path))
            log.info("bm25_encoder_loaded", path=str(encoder_path))
        else:
            self._bm25 = BM25Encoder.default()
            log.info("bm25_encoder_default", note="run ingestion first to fit on your corpus")

    @cached_property
    def _store(self) -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=settings.pinecone_index,
            embedding=self._embeddings,
            namespace=settings.pinecone_namespace,
        )
    
    def retrieve(
            self,
            query: str,
            filter_metadata: dict | None = None,
    ) -> list[Document]:
        """
        Run a hybrid dense + sparse query against Pinecone.
 
        Args:
            query:           Raw query string.
            filter_metadata: Optional Pinecone metadata filter dict.
                             Supports $eq, $in, $gte, $lte operators.
                             e.g. {"category": "legal", "year": {"$gte": 2023}}
        """
        # 1. Compute dense embedding vector
        dense_vector = self._embeddings.embed_query(query)

        # 2. Compute sparse BM25 vector
        sparse_vector = self._bm25.encode_queries(query)

        # 3. Apply alpha weighting 
        scaled_dense  = [v * self.alpha for v in dense_vector]
        scaled_sparse = {
            "indices": sparse_vector["indices"],
            "values": [v * (1 - self.alpha) for v in sparse_vector["values"]],
        }
        # 3. Build query kwargs
        query_kwargs: dict = {
            "vector": scaled_dense,
            "sparse_vector": scaled_sparse,
            "top_k": self.top_k,
            "namespace": settings.pinecone_namespace,
            "include_metadata": True,
        }
        if filter_metadata:
            query_kwargs["filter"] = filter_metadata

        # 4. Single Pinecone hybrid query
        response = self._index.query(**query_kwargs)

        # 5. Convert Pinecone matches → LangChain Documents
        docs: list[Document] = []
        for match in response.matches:
            meta = dict(match.metadata or {})
            content = meta.pop("text", "")
            meta["vector_store"] = round(float(match.score), 4)
            meta["pinecone_id"] = match.id
            docs.append(Document(page_content=content, metadata=meta))

        log.debug("hybrid_retriever", query=query[:60], alpha=self.alpha, results=len(docs))
        return docs
    

    def fit_bm25(self, corpus: list[str]) -> None:
        """
        Fit the BM25 encoder on your domain corpus and save it to disk.
        """
        self._bm25.fit(corpus)
        self._bm25.dump(settings.bm25_encoder_path)
        log.info("bm25_encoder_fitted", path=settings.bm25_encoder_path)
