"""Query expansion: HyDE (hypothetical document embedding) + multi-query."""
from __future__ import annotations
 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
 
import structlog
 
from config import get_settings
from src.generation.llm_client import get_llm
 
log = structlog.get_logger(__name__)
settings = get_settings()

# HyDE

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Write a short, factual passage (3-5 sentences) "
        "that would directly answer the user's question. "
        "Do not hedge or say you don't know — write as if you are certain.",
    ),
    ("human", "{question}"),
])

def hyde_expander(question: str) -> str:
    """Generate a hypothetical answer to improve dense retrieval."""
    chain = _HYDE_PROMPT | get_llm() | StrOutputParser()
    hypothetical = chain.invoke({"question": question})
    log.debug(
        "hyde_expanded",
        original=question[:60], 
        hypothetical=hypothetical[:80],
        hyp_length= len(hypothetical),
    )
    return hypothetical

# Multi-query

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate {n} different phrasings of the user's question to improve retrieval coverage. "
        "Return only the questions, one per line, no numbering.",
    ),
    ("human", "{question}"),
])
 
 
def multi_query_expand(question: str, n: int = 3) -> list[str]:
    """Return *n* query variations including the original."""
    chain = _MULTI_QUERY_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": question, "n": n})
    variants = [line.strip() for line in raw.strip().splitlines() if line.strip() and line.strip() != question]
    queries = [question] + variants[:n]

    log.debug(
        "multi_query_expanded",
        original_question= question[:60],
        n_variants = len(variants),
        total_queries= len(queries),
    )
    return queries