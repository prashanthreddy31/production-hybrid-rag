"""
Answer validator — hallucination guard.
 
Two validation modes:
  • "llm"   — ask the LLM to verify each sentence is grounded in context
              (authoritative, costs ~200 tokens per answer)
  • "lexical"— lightweight token-overlap check, zero LLM calls
              (fast, catches obvious hallucinations)
"""
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
from typing import Literal
 
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
 
import structlog
 
log = structlog.get_logger(__name__)
 
ValidateMode = Literal["llm", "lexical"]

# ── Data model ────────────────────────────────────────────────────────────────
 
@dataclass
class ValidationResult:
    is_grounded: bool
    hallucinated_sentences: list[str] = field(default_factory=list)
    grounded_sentences: list[str] = field(default_factory=list)
    confidence: float = 1.0
    mode: str = "lexical"

# ── LLM-based validation ──────────────────────────────────────────────────────

_VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict fact-checker. "
        "Given a CONTEXT and an ANSWER, identify any sentences in the ANSWER "
        "that contain claims NOT supported by the CONTEXT. "
        "Return a JSON object with two keys:\n"
        '  "grounded": [list of supported sentences]\n'
        '  "hallucinated": [list of unsupported sentences]\n'
        "Return ONLY the JSON object, no explanation.",
    ),
    (
        "human",
        "CONTEXT:\n{context}\n\n"
        "ANSWER:\n{answer}",
    ),
])

def _llm_validate(answer: str, context: str) -> ValidationResult:
    import json
    from src.generation.llm_client import get_llm

    chain = _VALIDATE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"context": context, "answer": answer})

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.load(raw)
        hallucinated = data.get("hallucinated", [])
        grounded = data.get("grounded", [])
    except json.JSONDecodeError:
        log.warning("validation_parse_error", raw=raw[:120])
        hallucinated = []
        grounded = _split_sentences(answer)

    is_grounded = len(hallucinated) ==0
    confidence = len(grounded) / max(len(grounded) + len(hallucinated), 1)

    return ValidationResult(
        is_grounded= is_grounded,
        hallucinated_sentences= hallucinated,
        grounded_sentences= grounded,
        confidence= confidence,
        mode="llm",
    )

# ── Lexical validation ────────────────────────────────────────────────────────
 
def _lexical_validate(
    answer: str,
    context: str,
    overlap_threshold: float = 0.25,
) -> ValidationResult:
    """
    Flag a sentence as hallucinated if its key-noun overlap with the
    full context is below *overlap_threshold*.
    """
    context_tokens = set(re.findall(r"\b[a-z]{4,}\b", context.lower()))
    sentences = _split_sentences(answer)
 
    grounded, hallucinated = [], []
    for sent in sentences:
        # Skip short connective sentences
        if len(sent.split()) < 6:
            grounded.append(sent)
            continue
        sent_tokens = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
        if not sent_tokens:
            grounded.append(sent)
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap >= overlap_threshold:
            grounded.append(sent)
        else:
            hallucinated.append(sent)
 
    is_grounded = len(hallucinated) == 0
    confidence = len(grounded) / max(len(sentences), 1)
 
    return ValidationResult(
        is_grounded=is_grounded,
        hallucinated_sentences=hallucinated,
        grounded_sentences=grounded,
        confidence=round(confidence, 3),
        mode="lexical",
    )
 
 
# ── Public API ────────────────────────────────────────────────────────────────
 
def validate_answer(
    answer: str,
    docs: list[Document],
    mode: ValidateMode = "lexical",
) -> ValidationResult:
    """
    Validate that *answer* is grounded in *docs*.
 
    Args:
        answer: The generated answer string.
        docs:   The source documents used to generate the answer.
        mode:   "lexical" (fast) or "llm" (accurate).
 
    Returns:
        ValidationResult with is_grounded flag and per-sentence breakdown.
    """
    context = "\n\n".join(d.page_content for d in docs)
 
    if mode == "llm":
        result = _llm_validate(answer, context)
    else:
        result = _lexical_validate(answer, context)
 
    if not result.is_grounded:
        log.warning(
            "hallucination_detected",
            mode=mode,
            count=len(result.hallucinated_sentences),
            sentences=result.hallucinated_sentences[:2],
        )
    else:
        log.debug("answer_fully_grounded", mode=mode, confidence=result.confidence)
 
    return result
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]