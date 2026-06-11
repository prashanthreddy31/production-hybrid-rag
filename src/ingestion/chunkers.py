"""Chunking strategies: recursive character, semantic, and fixed-size."""
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

import structlog
from config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

# Recursive chunking

def recursive_chunk(
        docs: list[Document],
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap
) -> list[Document]:
    """General purpose splitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators= ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        add_start_index = True
    )
    chunks = splitter.split_documents(docs)
    _tag_chunks(chunks, "recursive")
    log.info("Chunked_recursive", input=len(docs), output=len(chunks))
    return chunks

# Markdown-header-aware

def markdown_chunk(docs: list[Document]) -> list[Document]:
    """Split Markdown docs on headers; fall back to recursive for large sections."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: list[Document]= []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata.update(doc.metadata)
            if len(section.page_content) > settings.chunk_size:
                chunks.extend(char_splitter.split_documents([section]))
            else:
                chunks.append(section)
    _tag_chunks(chunks, "markdown")
    log.info("Chunked_markdown", input=len(docs), output=len(chunks))
    return chunks

# ── Dispatcher ────────────────────────────────────────────────────────────────
 
Strategy = Literal["recursive", "markdown"]

def chunk_documents(
        docs: list[Document],
        strategy: Strategy = "recursive",
        **kwargs,
) -> list[Document]:
    if strategy == "recursive":
        return recursive_chunk(docs)
    if strategy == "markdown":
        return markdown_chunk(docs)
    raise ValueError(f"Unknown Chunking strategy: {strategy}")

 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _tag_chunks(chunks: list[Document], strategy: str) -> None:
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata.setdefault("chunk_id", f"{chunk.metadata.get('source', 'doc')}::{i}")
