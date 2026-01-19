"""Redis caching layer for RAG pipeline."""

from typing import Optional, Any
import json
import os


class RedisCache:
    """Redis-based caching for query results."""

    def __init__(
        self,
        url: Optional[str] = None,
        ttl: int = 3600,
        prefix: str = "rag:"
    ):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = int(os.getenv("REDIS_CACHE_TTL", ttl))
        self.prefix = prefix
        self._client = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.url, decode_responses=True)
            except ImportError:
                raise RuntimeError("redis package is required for caching")
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            value = await client.get(full_key)

            if value is None:
                return None

            return json.loads(value)
        except Exception:
            # Cache miss on error - don't fail the request
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            serialized = json.dumps(value)

            await client.setex(
                full_key,
                ttl or self.ttl,
                serialized
            )
            return True
        except Exception:
            # Don't fail on cache errors
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            await client.delete(full_key)
            return True
        except Exception:
            return False

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern."""
        try:
            client = await self._get_client()
            full_pattern = self._make_key(pattern)

            # Get all matching keys
            keys = []
            async for key in client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)

            return len(keys)
        except Exception:
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            return await client.exists(full_key) > 0
        except Exception:
            return False

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self._get_client()
            info = await client.info("stats")

            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "connected": True
            }
        except Exception as e:
            return {
                "hits": 0,
                "misses": 0,
                "connected": False,
                "error": str(e)
            }
