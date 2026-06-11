"""
Middleware stack:
  1. RequestTracingMiddleware  — injects trace_id, logs every request/response
  2. RateLimitMiddleware       — sliding-window rate limit per API key / IP
  3. APIKeyAuthMiddleware      — Bearer token validation (optional, toggled by config)
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

# ── 1. Request tracing ────────────────────────────────────────────────────────

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique ``X-Trace-Id`` header to every request/response.
    Logs method, path, status, and latency in structured JSON.
    """

    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        request.state.started_at = time.perf_counter()

        log.info(
            "request_start",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        response: Response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - request.state.started_at) * 1000)

        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Latency-Ms"] = str(elapsed_ms)

        log.info(
            "request_end",
            trace_id=trace_id,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        return response


# ── 2. Rate limiting ──────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter keyed on API key (from Authorization header)
    or client IP as fallback.

    Limit: settings.rate_limit_per_minute requests per 60-second window.
    Exempt paths: /metrics, /health, /docs, /openapi.json
    """

    _EXEMPT = {"/metrics", "/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._limit = settings.rate_limit_per_minute
        self._window = 60.0  # seconds

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._EXEMPT:
            return await call_next(request)

        client_key = self._client_key(request)
        now = time.time()
        window = self._windows[client_key]

        # Evict timestamps outside the sliding window
        while window and window[0] < now - self._window:
            window.popleft()

        if len(window) >= self._limit:
            log.warning("rate_limit_exceeded", client=client_key)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": self._limit,
                    "window_seconds": int(self._window),
                },
                headers={"Retry-After": str(int(self._window))},
            )

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(self._limit - len(window))
        return response

    @staticmethod
    def _client_key(request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return f"key:{auth[7:20]}"   # first 13 chars of token as key
        return f"ip:{request.client.host if request.client else 'unknown'}"


# ── 3. API key auth ───────────────────────────────────────────────────────────

_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Simple Bearer-token guard.

    - Paths in _PUBLIC_PATHS are unauthenticated.
    - All other paths require ``Authorization: Bearer <api_secret_key>``.

    In production replace with a proper JWT / OAuth2 implementation.
    """

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in _PUBLIC_PATHS):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {settings.api_secret_key.get_secret_value()}"

        if auth != expected:
            log.warning(
                "auth_failed",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)