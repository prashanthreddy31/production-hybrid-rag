"""LLM client factory — Groq, with retry."""
from __future__ import annotations
 
from functools import lru_cache
 
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential
 
import structlog
 
from config import get_settings
 
log = structlog.get_logger(__name__)
settings = get_settings()

@lru_cache(maxsize=1)
def get_llm(streaming: bool = False) -> BaseChatModel:
    """Return a cached LLM instance."""
    return ChatGroq(
        model = settings.llm_model,
        temperature= settings.llm_temperature,
        api_key= settings.groq_api_key.get_secret_value(),
        streaming=streaming,
        max_tokens=settings.llm_max_tokens,
    )