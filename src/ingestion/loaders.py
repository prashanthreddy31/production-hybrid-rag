"""Document loaders — normalise every source format into LangChain Documents."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_core.documents import Document

import structlog

log = structlog.get_logger(__name__)

class BaseLoader:
    """Thin wrapper that adds source metadata and basic cleanup"""

    def load(self, source: str | Path) -> list[Document]:
        raise NotImplementedError
    
    @staticmethod
    def _clean(docs: list[Document], source: str) -> list[Document]:
        for doc in docs:
            doc.page_content = doc.page_content.strip()
            doc.metadata.setdefault("source", source)
            doc.metadata.setdefault("loader", "base")
        return [d for d in docs if d.page_content]
    


class PDFLoader(BaseLoader):
    def load(self, source: str | Path) -> list[Document]:
        path = str(source)
        log.info("Loading_pdf...", path = path)
        raw = PyPDFLoader(path).load()
        for i, doc in enumerate(raw):
            doc.metadata.update({"page": i, "file_type": "pdf", "loader": "pdf"})
        return self._clean(raw, path)
    
class MarkdownLoader(BaseLoader):
    def load(self, source: str | Path) -> list[Document]:
        path = str(source)
        log.info("Loading_markdown...", path=path)
        raw = TextLoader(path, encoding="utf-8").load()
        for doc in raw:
            doc.metadata.update({"file_type": "markdown", "loader": "markdown"})
        return self._clean(raw, path)
    

# ── Auto-dispatch loader ──────────────────────────────────────────────────────

_EXT_MAP: dict[str, BaseLoader] = {
    ".pdf" : PDFLoader(),
    ".md" : MarkdownLoader(),
    ".txt" : MarkdownLoader(),
}

def load_document(source: str | Path) -> list[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = Path(source).suffix.lower()
    loader = _EXT_MAP.get(ext)
    if loader is None:
        mime, _ = mimetypes.guess_type(str(source))
        if mime and "pdf" in mime:
            loader = PDFLoader()
        elif mime and "md" in mime:
            loader = TextLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext!r} ({source})")        
    return loader.load(source)

def load_directory(directory: str | Path, glob: str = "**/*") -> list[Document]:
    """Recursively load all supported documents from *directory*."""
    base = Path(directory)
    docs: list[Document] = []
    for path in sorted(base.glob(glob)):
        if path.suffix.lower() in _EXT_MAP and path.is_file():
            try:
                docs.extend(load_document(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("loader_error", path=str(path), error=str(exc))
    log.info("directory_loaded", directory=str(directory), total_docs=len(docs))
    return docs
                



