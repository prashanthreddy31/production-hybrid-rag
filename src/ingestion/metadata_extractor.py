"""Metadata extraction and near duplicate detection."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

import structlog

log = structlog.get_logger(__name__)

# ── Metadata extractor

def extract_metadata(doc: Document) -> Document:
    """Enrich document metadata with inferred title, word count, checksum, etc."""
    text = doc.page_content
    meta = doc.metadata

    # Infer title from first non-empty line
    if 'title' not in meta:
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        meta["title"] = first_line[:120] or meta.get("source", "untitled")

    # Word and character counts
    meta["word_count"] = len(text.split())
    meta["char_count"] = len(text)

    # Language hint (very cheap heuristic – replace with langdetect if needed)
    meta.setdefault("language", "en")
 
    # Stable content hash
    meta["content_hash"] = hashlib.sha256(text.encode()).hexdigest()
 
    # Infer domain/category from source path
    source = meta.get("source", "")
    if source:
        parts = Path(source).parts
        meta.setdefault("category", parts[-2] if len(parts) >= 2 else "general")
 
    return doc

def extract_metadata_batch(docs: list[Document]) -> list[Document]:
    return [extract_metadata(d) for d in docs]

# ── SimHash deduplicator ──────────────────────────────────────────────────────
 
def _simhash(text: str, bits: int = 64) -> int:
    """Minimal SimHash implementation — no external dep required."""
    tokens = re.findall(r"\w+", text.lower())
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint
 
 
def _hamming(a: int, b: int, bits: int = 64) -> int:
    return bin(a ^ b).count("1")
 
 
def deduplicate(
    docs: list[Document],
    threshold: int = 4,   # max Hamming distance to consider duplicate
) -> list[Document]:
    """Remove near-duplicate documents using SimHash."""
    seen: list[tuple[int, int]] = []  # (fingerprint, original_index)
    unique: list[Document] = []
 
    for doc in docs:
        fp = _simhash(doc.page_content)
        if all(_hamming(fp, s) > threshold for s, _ in seen):
            seen.append((fp, len(unique)))
            unique.append(doc)
 
    removed = len(docs) - len(unique)
    if removed:
        log.info("deduplication", removed=removed, kept=len(unique))
    return unique
 