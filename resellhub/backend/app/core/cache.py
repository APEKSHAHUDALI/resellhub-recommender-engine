"""
Thin Redis wrapper for caching recommendation responses.

Recommendation results for a given user rarely need to be computed twice
within a few minutes of each other, so we cache the serialized API response
by (user_id, surface) for cache_ttl_seconds. If Redis is down, we log and
fall back to computing live rather than 500ing - caching should never be a
single point of failure for a read path.
"""
from __future__ import annotations
import json
import logging

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    _client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=1)
    _client.ping()
    _available = True
except Exception:
    _client = None
    _available = False
    logger.warning("Redis unavailable at startup - recommendation caching disabled, serving live.")


def get_cached(key: str):
    if not _available:
        return None
    try:
        raw = _client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("Redis GET failed (%s) - falling back to live compute", e)
        return None


def set_cached(key: str, value, ttl: int | None = None):
    if not _available:
        return
    try:
        _client.set(key, json.dumps(value), ex=ttl or settings.cache_ttl_seconds)
    except Exception as e:
        logger.warning("Redis SET failed (%s) - continuing without cache write", e)


def invalidate(prefix: str):
    """Called after retraining so stale recommendations aren't served from cache."""
    if not _available:
        return
    try:
        for key in _client.scan_iter(f"{prefix}*"):
            _client.delete(key)
    except Exception as e:
        logger.warning("Redis invalidate failed (%s)", e)
