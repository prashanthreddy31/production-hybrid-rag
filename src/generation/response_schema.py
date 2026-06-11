"""Pydantic models for API request/response contracts."""
from __future__ import annotations
 
from typing import Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str | None = None
    filter_metadata: dict[str, Any] | None = None
    top_k: int | None = None
    stream: bool = False


class SourceDocument(BaseModel):
    content: str
    source: str
    title: str | None = None
    page: int | None = None
    score: float | None = None
    doc_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    session_id: str | None = None
    is_fully_grounded: bool
    latency_ms: int
    model: str

class IngestRequest(BaseModel):
    texts: list[str] | None = None
    metadatas: list[dict] | None = None
 
 
class IngestResponse(BaseModel):
    chunks_indexed: int
    elapsed_seconds: float
    source: str
 
 
class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    checks: dict[str, str]