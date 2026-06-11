"""
FastAPI application entrypoint.

"""
from __future__ import annotations
 
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from config import get_settings
from src.api.middleware import (
    RequestTracingMiddleware,
    RateLimitMiddleware,
    APIKeyAuthMiddleware,
)
from src.api.routes.routes_query import router as query_router
from src.api.routes.routes_health import router as health_router
from src.observability.metrics import setup_metrics
from src.observability.tracer import is_tracing_enabled
 
log = structlog.get_logger(__name__)
settings = get_settings()

# App factory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _configure_logging()
    _configure_langsmith()

    log.info(
        "app_started",
        llm_model=settings.llm_model,
        pinecone_index=settings.pinecone_index,  
        langsmith_tracing=is_tracing_enabled(),
    )

    yield

    # Shutdown
    log.info("app_shutdown")

def create_app() -> FastAPI:
    app = FastAPI(
        title= "Hybrid RAG System",
        description=(
            "Domain-specific RAG system with hybrid BM25 + vector retrieval, "
            "cross-encoder reranking, citation enforcement, and semantic caching."
        ),
        version= "0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # ── Middleware (outermost → innermost) ────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyAuthMiddleware)

    # Routers
    app.include_router(health_router)
    app.include_router(query_router, prefix="/api/v1")

    # Observability 
    setup_metrics(app)   # mounts GET /metrics

    return app

# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _configure_logging() -> None:
    import logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
 
 
def _configure_langsmith() -> None:
    import os
    key = settings.langsmith_api_key.get_secret_value()
    if key:
        os.environ["LANGCHAIN_API_KEY"]      = key
        os.environ["LANGCHAIN_PROJECT"]      = settings.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"]   = "true"
        log.info("langsmith_tracing_enabled", project=settings.langsmith_project)
 
 
# ── Module-level app instance (used by uvicorn) ───────────────────────────────
app = create_app()