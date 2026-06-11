"""
ingestion/
~~~~~~~~~~
Document ingestion package for the Ask-My-Docs RAG system.
 
Public surface:
    IngestionPipeline   — orchestrate the full load→chunk→embed→index flow
    load_document       — load a single file into LangChain Documents
    load_directory      — recursively load all supported files in a folder
    chunk_documents     — split Documents with a chosen strategy
    get_embeddings      — return a cached embedding model instance
    extract_metadata    — enrich a single Document with inferred metadata
    deduplicate         — SimHash-based near-duplicate removal
"""
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.loaders import load_document, load_directory
from src.ingestion.chunkers import chunk_documents
from src.ingestion.embedders import get_embeddings
from .metadata_extractor import extract_metadata_batch, deduplicate
 
__all__ = [
    "IngestionPipeline",
    "load_document",
    "load_directory",
    "chunk_documents",
    "get_embeddings",
    "extract_metadata",
    "deduplicate",
]