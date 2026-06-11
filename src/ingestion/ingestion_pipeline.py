"""Ingestion pipeline: load → chunk → deduplicate → embed → index."""
from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore, PineconeSparseVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

import structlog

from config import get_settings
from src.ingestion.loaders import load_document, load_directory
from src.ingestion.chunkers import chunk_documents, Strategy
from src.ingestion.embedders import get_embeddings, EmbeddingProvider
from src.ingestion.metadata_extractor import extract_metadata_batch, deduplicate
 
log = structlog.get_logger(__name__)
settings = get_settings()

class IngestionPipeline:
    """End to end ingestion: ffiles/directories → Pinecone + BM25 index."""

    def __init__(
            self, 
            chunking_strategy: Strategy = "recursive",
            embedding_provider: EmbeddingProvider = "huggingface",
            batch_size: int = 64,
    ) -> None:
        self.chunking_strategy = chunking_strategy
        self.embeddings = get_embeddings(embedding_provider)
        self.batch_size = batch_size
        self.pc = Pinecone(api_key= settings.pinecone_api_key.get_secret_value())
        self._ensure_index()
        self.index = self.pc.Index(settings.pinecone_index)
        
        # Load existing BM25 encoder if already fitted, else fall back to default
        encoder_path = Path(settings.bm25_encoder_path)
        if encoder_path.exists():
            self._bm25 = BM25Encoder().load(str(encoder_path))
            log.info("bm25_encoder_loaded", path=str(encoder_path))
        else:
            self._bm25 = BM25Encoder.default()   
            log.info("bm25_encoder_default", note="will be refitted after first ingest")


    def ingest_directory(self, directory: str | Path) -> dict:
        docs = load_directory(directory)
        return self._run_pipeline(docs, source= str(directory))
    
    def ingest_documents(self, docs: list[Document]) -> dict:
        return self._run_pipeline(docs, source="api_upload")
    
    # Internal steps

    def _run_pipeline(self, docs: list[Document], source: str) -> dict:
        t0 = time.perf_counter()
        log.info("pipeline_start", source=source, raw_docs=len(docs))

        # 1. Metadata enrichment
        docs = extract_metadata_batch(docs)

        # 2. Chunk
        chunks = chunk_documents(docs, strategy=self.chunking_strategy)

        # 3. Chunk dedup
        chunks = deduplicate(chunks, threshold=3)

        # 4. Fit Bm25 so sparse vectore use fitted stats
        self._fit_and_save_bm25(chunks)

        # 5. Index dense + sparse into Pinecone in batches
        total_indexed = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.page_content for c in batch]

            # Compute both vectors locally
            dense_vectors = self.embeddings.embed_documents(texts)
            sparse_vectors = self._bm25.encode_documents(texts)

            # Build upsert payload with both vectors
            vectors = []
            for j, chunk in enumerate(batch):
                chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i+j}")
                vectors.append({
                    "id": chunk_id,
                    "values": dense_vectors[j],
                    "sparse_values": sparse_vectors[j],
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.page_content,
                    },
                })

            self.index.upsert(
                vectors= vectors,
                namespace= settings.pinecone_namespace
            )

            total_indexed += len(batch)
            log.info("batch_indexed", batch=i // self.batch_size + 1, indexed=total_indexed, total= len(chunks))


        elapsed = time.perf_counter() - t0
        result = {
            "source": source,
            "raw_docs": len(docs),
            "chunks_indexed": total_indexed,
            "elapsed_seconds": round(elapsed, 2),
        }
        log.info("pipeline_complete", **result)
        return result
    
    def _fit_and_save_bm25(self, chunks: list[Document]) -> None:
        """Fit BM25Encoder on chunks and persist params to disk."""
        corpus = [c.page_content for c in chunks]
        self._bm25.fit(corpus)
        self._bm25.dump(settings.bm25_encoder_path)
        log.info(
            "bm25_encoder_fitted",
            corpus_size=len(corpus),
            path=settings.bm25_encoder_path,
        )
    
    def _ensure_index(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing = {idx.name for idx in self.pc.list_indexes()}
        if settings.pinecone_index not in existing:
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.embedding_dim,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            log.info(
                "pinecone_index_created",
                name=settings.pinecone_index,
                dim=settings.embedding_dim,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            )
        else:
            log.info("pinecone_index_exists", name=settings.pinecone_index)