"""Citation enforcement: inject [doc-N] markers and validate grounded responses."""
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
 
from langchain_core.documents import Document
 
import structlog
 
log = structlog.get_logger(__name__)
 
_CITATION_PATTERN = re.compile(r"\[doc-(\d+)\]")

@dataclass
class CitationResult:
    answer: str
    cited_indices: list[int]
    uncited_sentences: list[str]
    is_fully_grounded: bool
    context_block: str


def build_context_block(docs: list[Document]) -> str:
    """Format retrieved docs into a numbered context block for the prompt."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unkown")
        title = doc.metadata.get("title", source)
        lines.append(f"[doc-{i}] (source: {title})\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(lines)

def validate_citations(answer: str, docs: list[Document]) -> CitationResult:
    """
    Check every sentence in *answer* contains at least one [doc-N] citation.
    Returns a CitationResult with grounding analysis.
    """
    context_block = build_context_block(docs)
    cited_indices = [int(m) for m in _CITATION_PATTERN.findall(answer)]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    uncited = [s for s in sentences if not _CITATION_PATTERN.search(s) and len(s.split()) > 6]

    result = CitationResult(
        answer=answer,
        cited_indices= sorted(set(cited_indices)),
        uncited_sentences= uncited,
        is_fully_grounded= len(uncited) == 0,
        context_block= context_block,
    )

    if uncited:
        log.warning(
            "uncited_sentences_detected",
            count=len(uncited),
            examples=[s[:80] for s in uncited[:2]],
        )
 
    return result

def strip_hallucinated_citations(answer: str, num_docs: int) -> str:
    """Remove any [doc-N] citations that reference non-existent documents."""

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= num_docs else ""
    
    cleaned = _CITATION_PATTERN.sub(_replace, answer)
    return re.sub(r"\s{2,}", " ", cleaned).strip()