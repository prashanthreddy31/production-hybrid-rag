"""Embedding providers: Cohere and HuggingFace."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal
import torch 

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings

import structlog

from config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

EmbeddingProvider = Literal["cohere", "huggingface"]

@lru_cache(maxsize=1)
def get_embeddings(provider: EmbeddingProvider ="huggingface") -> Embeddings:
    """Return a cached embeddings instance for *provider*."""
    if provider == "huggingface":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Embeddings_provider", provider="HuggingFace", model=settings.embedding_model)
        return HuggingFaceEmbeddings(
            model_name = settings.embedding_model,
            model_kwargs={
                "device": device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
                },
        )
    
    if provider == "cohere":
        log.info("Embeddings_provider", provider="cohere")
        return CohereEmbeddings(input_type = "search_document")
    
    raise ValueError(f"Unkown embedding provider : {provider!r}")