"""
Session manager — Redis-backed chat history store.
 
Each session holds an ordered list of {role, content} turns.
History is capped at MAX_TURNS and expires after TTL seconds.
"""
from __future__ import annotations
 
import json
from typing import Literal
 
import redis
import structlog
 
from config import get_settings
 
log = structlog.get_logger(__name__)
settings = get_settings()
 
MAX_TURNS = 20          # total messages stored per session
Role = Literal["user", "assistant", "system"]

class SessionManager:
    """
    Store and retrieve per-session chat history in Redis.
 
    Keys: ``session:{session_id}``  →  JSON list of {role, content} dicts.
    """

    def __init__(self) -> None:
        self._redis = redis.from_url(
            settings.redis_url,
            decode_responses= True,
            socket_connect_timeout= 2,
        )
        self._ttl = settings.cache_ttl_seconds

    
    def get_history(self, session_id: str) -> list[dict]:
        """Return the full chat history for *session_id* (newest last)."""
        try:
            raw = self._redis.get(self._key(session_id))
            if raw is None:
                return []
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            log.warning("session_get_error", session_id=session_id, error=str(exc))
            return []
        

    def append(self, session_id: str, role: Role, content: str) -> None:
        """Append one turn and reset the TTL"""
        try:
            history = self.get_history(session_id)
            history.append({"role": role, "content": content})
            # Cap at MAX_TURNS
            if len(history) > MAX_TURNS:
                history = history[-MAX_TURNS:]
            self._redis.set(
                self._key(session_id),
                json.dumps(history),
                ex=self._ttl,
            )
        except redis.RedisError as exc:
            log.warning("session_append_error", session_id=session_id, error=str(exc))
 
    def clear(self, session_id: str) -> None:
        """Delete the session history."""
        try:
            self._redis.delete(self._key(session_id))
            log.info("session_cleared", session_id=session_id)
        except redis.RedisError as exc:
            log.warning("session_clear_error", session_id=session_id, error=str(exc))
 
    def exists(self, session_id: str) -> bool:
        try:
            return bool(self._redis.exists(self._key(session_id)))
        except redis.RedisError:
            return False
 
    # ── Internal ──────────────────────────────────────────────────────────────
 
    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"
