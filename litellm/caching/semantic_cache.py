"""
Semantic Cache implementation for LiteLLM

Provides semantic caching using embedding-based similarity matching with Redis
as the storage backend. Unlike exact-match caching, this finds cached responses
for prompts that are semantically similar (not just identical).

Has 4 methods (BaseCache interface):
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache

Plus semantic-specific methods:
    - get_semantic_match
    - store_semantic_entry
    - get_stats
"""

import ast
import asyncio
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import print_verbose, verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.types.utils import EmbeddingResponse

from .base_cache import BaseCache

try:
    from pydantic import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors. Pure Python, no numpy dependency.

    Returns a float in [-1, 1] where 1 means identical direction,
    0 means orthogonal, and -1 means opposite direction.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have equal length, got {len(a)} and {len(b)}"
        )
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot_product += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SemanticCacheConfig(BaseModel):
    """Configuration model for the SemanticCache."""

    enabled: bool = False
    similarity_threshold: float = 0.90
    embedding_model: str = "text-embedding-3-small"
    redis_url: Optional[str] = None
    ttl: int = 3600
    max_cache_size: int = 10000


class SemanticCache(BaseCache):
    """
    Semantic cache backed by Redis with embedding-based similarity search.

    Core idea: instead of hashing queries by exact string match, this cache
    generates embeddings via `litellm.embedding()` and finds cached responses
    whose prompt embeddings have cosine similarity >= the configured threshold.

    Storage format (Redis):
        - Index key: ``semantic_cache:index`` — a Redis SET of all entry keys
        - Entry key: ``semantic_cache:entry:<hash>`` — a Redis STRING holding JSON:
          ``{"embedding": [...], "response": "...", "prompt": "...", "timestamp": ..., "hit_count": 0}``

    Similarity search strategy (MVP):
        Brute-force scan of all stored embeddings. This works well for up to
        ~10 000 entries. For larger deployments consider:
        - Redis Vector Search (RediSearch / redis-stack with FT.SEARCH)
        - Azure AI Search
        - A dedicated vector database (Qdrant, Pinecone, etc.)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        embedding_model: str = "text-embedding-3-small",
        redis_url: Optional[str] = None,
        ttl: int = 3600,
        max_cache_size: int = 10000,
        **kwargs,
    ):
        """
        Initialize the Semantic Cache.

        Args:
            similarity_threshold: Minimum cosine similarity to consider a cache hit (0.0–1.0).
            embedding_model: Model name passed to ``litellm.embedding()``.
            redis_url: Full Redis URL. Falls back to ``REDIS_URL`` env var.
            ttl: Default time-to-live for cache entries in seconds.
            max_cache_size: Maximum number of entries to store / scan.
        """
        import redis.asyncio as aioredis

        super().__init__(default_ttl=ttl)

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.max_cache_size = max_cache_size

        # Resolve Redis URL
        _redis_url = redis_url or os.environ.get("REDIS_URL")
        if _redis_url is None:
            raise ValueError(
                "redis_url must be provided or set via the REDIS_URL environment variable."
            )

        self.redis_client = aioredis.from_url(
            _redis_url, decode_responses=True
        )

        # Stats counters
        self._total_lookups: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        verbose_proxy_logger.info(
            "SemanticCache initialised — model=%s, threshold=%.2f, ttl=%d, max_size=%d",
            self.embedding_model,
            self.similarity_threshold,
            self.default_ttl,
            self.max_cache_size,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_text(messages: List[Dict]) -> str:
        """Convert a messages list to a canonical string for embedding."""
        try:
            return get_str_from_messages(messages)
        except Exception:
            # Fallback: concatenate content fields
            parts: List[str] = []
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                parts.append(str(content))
            return " ".join(parts)

    @staticmethod
    def _hash_text(text: str) -> str:
        """Produce a deterministic short hash of text (used as part of the Redis key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _tenant_prefix(self, tenant_id: str = "default") -> str:
        """Return a tenant-scoped key prefix to isolate cache data per tenant."""
        return f"semantic_cache:{tenant_id}"

    def _entry_key(self, text_hash: str, tenant_id: str = "default") -> str:
        return f"{self._tenant_prefix(tenant_id)}:entry:{text_hash}"

    def _index_key(self, tenant_id: str = "default") -> str:
        return f"{self._tenant_prefix(tenant_id)}:index"

    async def _generate_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding vector using litellm.aembedding().

        Passes ``cache={"no-store": True, "no-cache": True}`` so the embedding
        call itself is never cached (avoids recursion).
        """
        try:
            from litellm.proxy.proxy_server import llm_model_list, llm_router
        except ImportError:
            llm_router = None
            llm_model_list = None

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )

        try:
            if llm_router is not None and self.embedding_model in router_model_names:
                user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
                embedding_response = await llm_router.aembedding(
                    model=self.embedding_model,
                    input=[text],
                    cache={"no-store": True, "no-cache": True},
                    metadata={
                        "user_api_key": user_api_key,
                        "semantic-cache-embedding": True,
                        "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                    },
                )
            else:
                embedding_response = cast(
                    EmbeddingResponse,
                    await litellm.aembedding(
                        model=self.embedding_model,
                        input=[text],
                        cache={"no-store": True, "no-cache": True},
                    ),
                )
            return embedding_response["data"][0]["embedding"]
        except Exception as e:
            verbose_proxy_logger.error(
                "SemanticCache: embedding generation failed — %s", str(e)
            )
            raise

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation."""
        embedding_response = cast(
            EmbeddingResponse,
            litellm.embedding(
                model=self.embedding_model,
                input=[text],
                cache={"no-store": True, "no-cache": True},
            ),
        )
        return embedding_response["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # Semantic-specific public API
    # ------------------------------------------------------------------

    async def get_semantic_match(
        self, messages: List[Dict], tenant_id: str = "default", **kwargs
    ) -> Optional[Dict]:
        """
        Search the cache for a semantically similar prompt.

        Args:
            messages: The chat messages to match against.
            tenant_id: Tenant identifier for cache isolation.

        Returns:
            The cached response dict if similarity >= threshold, else ``None``.
        """
        self._total_lookups += 1

        try:
            text = self._messages_to_text(messages)
            if not text.strip():
                self._cache_misses += 1
                return None

            query_embedding = await self._generate_embedding(text, **kwargs)

            index_key = self._index_key(tenant_id)

            # Retrieve all entry keys from the index set
            entry_keys = await self.redis_client.smembers(index_key)
            if not entry_keys:
                self._cache_misses += 1
                return None

            # Brute-force scan for the best match.
            # NOTE: For production workloads with >10 000 entries, migrate to
            # Redis Vector Search (RediSearch FT.SEARCH) or a dedicated vector
            # database for sub-linear search time.
            best_similarity: float = -1.0
            best_entry: Optional[Dict] = None
            best_key: Optional[str] = None

            # Fetch entries in batches via pipeline for efficiency
            pipe = self.redis_client.pipeline(transaction=False)
            key_list = list(entry_keys)
            for k in key_list:
                pipe.get(k)
            raw_values = await pipe.execute()

            for key, raw in zip(key_list, raw_values):
                if raw is None:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                stored_embedding = entry.get("embedding")
                if stored_embedding is None:
                    continue

                similarity = cosine_similarity(query_embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
                    best_key = key

            if best_entry is not None and best_similarity >= self.similarity_threshold:
                self._cache_hits += 1

                # Increment hit count (fire-and-forget)
                try:
                    best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
                    if best_key is not None:
                        remaining_ttl = await self.redis_client.ttl(best_key)
                        ttl_to_set = remaining_ttl if remaining_ttl > 0 else self.default_ttl
                        await self.redis_client.set(
                            best_key,
                            json.dumps(best_entry),
                            ex=ttl_to_set,
                        )
                except Exception:
                    pass  # Non-critical — don't block the response

                verbose_proxy_logger.debug(
                    "SemanticCache HIT — similarity=%.4f, threshold=%.2f, prompt=%s",
                    best_similarity,
                    self.similarity_threshold,
                    text[:80],
                )

                # Update metadata with similarity score
                kwargs.setdefault("metadata", {})["semantic-similarity"] = best_similarity

                return self._parse_cached_response(best_entry.get("response"))
            else:
                self._cache_misses += 1
                verbose_proxy_logger.debug(
                    "SemanticCache MISS — best_similarity=%.4f, threshold=%.2f",
                    best_similarity,
                    self.similarity_threshold,
                )
                kwargs.setdefault("metadata", {})["semantic-similarity"] = max(
                    best_similarity, 0.0
                )
                return None

        except Exception as e:
            self._cache_misses += 1
            verbose_proxy_logger.error(
                "SemanticCache: error during lookup — %s", str(e)
            )
            return None

    async def store_semantic_entry(
        self,
        messages: List[Dict],
        response: Dict,
        ttl: Optional[int] = None,
        tenant_id: str = "default",
        **kwargs,
    ) -> None:
        """
        Store a prompt/response pair with its embedding in the cache.

        Args:
            messages: The chat messages (used to generate the embedding).
            response: The LLM response to cache.
            ttl: Optional TTL override (seconds). Defaults to ``self.default_ttl``.
            tenant_id: Tenant identifier for cache isolation.
        """
        try:
            text = self._messages_to_text(messages)
            if not text.strip():
                return

            embedding = await self._generate_embedding(text, **kwargs)
            text_hash = self._hash_text(text)
            key = self._entry_key(text_hash, tenant_id)
            index_key = self._index_key(tenant_id)
            effective_ttl = ttl if ttl is not None else self.default_ttl

            entry = {
                "embedding": embedding,
                "response": str(response),
                "prompt": text[:500],  # store truncated prompt for debugging
                "timestamp": time.time(),
                "hit_count": 0,
            }

            pipe = self.redis_client.pipeline(transaction=False)
            pipe.set(key, json.dumps(entry), ex=effective_ttl)
            pipe.sadd(index_key, key)
            await pipe.execute()

            # Enforce max_cache_size by evicting oldest entries
            current_size = await self.redis_client.scard(index_key)
            if current_size > self.max_cache_size:
                await self._evict_oldest(current_size - self.max_cache_size, tenant_id=tenant_id)

            verbose_proxy_logger.debug(
                "SemanticCache STORE — key=%s, ttl=%d", key, effective_ttl
            )

        except Exception as e:
            verbose_proxy_logger.error(
                "SemanticCache: error storing entry — %s", str(e)
            )

    def get_stats(self) -> Dict:
        """
        Return cache performance statistics.

        Returns:
            Dict with: total_lookups, cache_hits, cache_misses, hit_rate,
            estimated_savings (rough estimate of avoided LLM calls).
        """
        hit_rate = (
            self._cache_hits / self._total_lookups
            if self._total_lookups > 0
            else 0.0
        )
        return {
            "total_lookups": self._total_lookups,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "estimated_savings": self._cache_hits,  # each hit = 1 avoided LLM call
        }

    # ------------------------------------------------------------------
    # BaseCache interface — async
    # ------------------------------------------------------------------

    async def async_get_cache(self, key: str, **kwargs) -> Any:
        """
        Retrieve a cached response via semantic similarity.

        The ``key`` argument is part of the BaseCache interface but is not used
        for the similarity lookup — ``messages`` from ``kwargs`` are used instead.
        """
        messages = kwargs.get("messages", [])
        if not messages:
            print_verbose("SemanticCache: no messages in async_get_cache kwargs")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None
        tenant_id = kwargs.get("metadata", {}).get("tenant_id", "default")
        return await self.get_semantic_match(messages, tenant_id=tenant_id, **kwargs)

    async def async_set_cache(self, key: str, value: Any, **kwargs) -> None:
        """Store a response in the semantic cache."""
        messages = kwargs.get("messages", [])
        if not messages:
            print_verbose("SemanticCache: no messages in async_set_cache kwargs")
            return
        tenant_id = kwargs.get("metadata", {}).get("tenant_id", "default")
        await self.store_semantic_entry(messages, value, tenant_id=tenant_id, **kwargs)

    async def async_set_cache_pipeline(
        self, cache_list: List[Tuple[str, Any]], **kwargs
    ) -> None:
        """Store multiple entries. Executes concurrently."""
        try:
            tasks = [
                self.async_set_cache(k, v, **kwargs) for k, v in cache_list
            ]
            await asyncio.gather(*tasks)
        except Exception as e:
            verbose_proxy_logger.error(
                "SemanticCache: error in async_set_cache_pipeline — %s", str(e)
            )

    # ------------------------------------------------------------------
    # BaseCache interface — sync
    # ------------------------------------------------------------------

    def set_cache(self, key: str, value: Any, **kwargs) -> None:
        """
        Synchronous cache store. Generates embedding synchronously and writes
        to Redis via a new event loop if necessary.
        """
        print_verbose(f"SemanticCache set_cache, kwargs keys: {list(kwargs.keys())}")
        try:
            messages = kwargs.get("messages", [])
            if not messages:
                return

            text = self._messages_to_text(messages)
            if not text.strip():
                return

            embedding = self._generate_embedding_sync(text)
            text_hash = self._hash_text(text)
            tenant_id = kwargs.get("metadata", {}).get("tenant_id", "default") if kwargs else "default"
            redis_key = self._entry_key(text_hash, tenant_id)
            index_key = self._index_key(tenant_id)
            ttl = self.get_ttl(**kwargs) or self.default_ttl

            entry = {
                "embedding": embedding,
                "response": str(value),
                "prompt": text[:500],
                "timestamp": time.time(),
                "hit_count": 0,
            }

            # For synchronous Redis writes we need a sync client or run async
            # in an event loop. Use asyncio.run in a thread-safe way.
            async def _write():
                pipe = self.redis_client.pipeline(transaction=False)
                pipe.set(redis_key, json.dumps(entry), ex=ttl)
                pipe.sadd(index_key, redis_key)
                await pipe.execute()

            try:
                loop = asyncio.get_running_loop()
                # We're inside an existing loop — schedule as a task
                loop.create_task(_write())
            except RuntimeError:
                asyncio.run(_write())

        except Exception as e:
            verbose_proxy_logger.error(
                "SemanticCache: sync set_cache error — %s", str(e)
            )

    def get_cache(self, key: str, **kwargs) -> Any:
        """
        Synchronous cache retrieval. Generates embedding synchronously and
        performs a brute-force scan.

        For high-throughput use the async methods are strongly recommended.
        """
        print_verbose(f"SemanticCache get_cache, kwargs keys: {list(kwargs.keys())}")
        self._total_lookups += 1

        try:
            messages = kwargs.get("messages", [])
            if not messages:
                self._cache_misses += 1
                return None

            text = self._messages_to_text(messages)
            if not text.strip():
                self._cache_misses += 1
                return None

            query_embedding = self._generate_embedding_sync(text)

            tenant_id = kwargs.get("metadata", {}).get("tenant_id", "default") if kwargs else "default"
            index_key = self._index_key(tenant_id)

            # Synchronous Redis read — run async code
            async def _read():
                entry_keys = await self.redis_client.smembers(index_key)
                if not entry_keys:
                    return None

                pipe = self.redis_client.pipeline(transaction=False)
                key_list = list(entry_keys)
                for k in key_list:
                    pipe.get(k)
                raw_values = await pipe.execute()

                best_sim = -1.0
                best_entry = None
                for raw in raw_values:
                    if raw is None:
                        continue
                    try:
                        entry = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    stored_emb = entry.get("embedding")
                    if stored_emb is None:
                        continue
                    sim = cosine_similarity(query_embedding, stored_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

                if best_entry is not None and best_sim >= self.similarity_threshold:
                    return best_entry.get("response")
                return None

            try:
                loop = asyncio.get_running_loop()
                # Cannot block inside a running loop; return None gracefully
                self._cache_misses += 1
                verbose_proxy_logger.debug(
                    "SemanticCache: sync get_cache called inside async context, returning None"
                )
                return None
            except RuntimeError:
                result = asyncio.run(_read())

            if result is not None:
                self._cache_hits += 1
                return self._parse_cached_response(result)
            else:
                self._cache_misses += 1
                return None

        except Exception as e:
            self._cache_misses += 1
            verbose_proxy_logger.error(
                "SemanticCache: sync get_cache error — %s", str(e)
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cached_response(cached_response: Any) -> Any:
        """Deserialize a cached response string back to a Python object."""
        if cached_response is None:
            return None

        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8")

        try:
            return json.loads(cached_response)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(cached_response)
            except (ValueError, SyntaxError):
                return cached_response

    async def _evict_oldest(self, count: int, tenant_id: str = "default") -> None:
        """
        Remove the *count* oldest entries from the cache.

        Uses sampling to avoid loading ALL entries into memory when the
        cache is large.
        """
        import random

        try:
            index_key = self._index_key(tenant_id)
            entry_keys = list(await self.redis_client.smembers(index_key))
            if not entry_keys or count <= 0:
                return

            # Sample a subset to find oldest (avoid loading ALL entries)
            sample_size = min(len(entry_keys), count * 3)
            sample_keys = (
                random.sample(entry_keys, sample_size)
                if len(entry_keys) > sample_size
                else entry_keys
            )

            # Get values only for sampled entries
            pipe = self.redis_client.pipeline(transaction=False)
            for k in sample_keys:
                pipe.get(k)
            raw_values = await pipe.execute()

            entries_with_ts: List[Tuple[str, float]] = []
            for key, raw in zip(sample_keys, raw_values):
                if raw is None:
                    continue
                try:
                    entry = json.loads(raw)
                    entries_with_ts.append((key, entry.get("timestamp", 0.0)))
                except (json.JSONDecodeError, TypeError):
                    entries_with_ts.append((key, 0.0))  # Remove corrupt entries

            # Sort ascending by timestamp (oldest first)
            entries_with_ts.sort(key=lambda x: x[1])
            keys_to_remove = [k for k, _ in entries_with_ts[:count]]

            if keys_to_remove:
                pipe = self.redis_client.pipeline(transaction=False)
                for k in keys_to_remove:
                    pipe.delete(k)
                    pipe.srem(index_key, k)
                await pipe.execute()

                verbose_proxy_logger.debug(
                    "SemanticCache: evicted %d oldest entries", len(keys_to_remove)
                )
        except Exception as e:
            verbose_proxy_logger.error(
                "SemanticCache: eviction error — %s", str(e)
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        try:
            await self.redis_client.aclose()
        except Exception:
            pass

    async def test_connection(self) -> dict:
        """Test Redis connectivity."""
        try:
            pong = await self.redis_client.ping()
            if pong:
                return {"status": "success", "message": "SemanticCache Redis connection OK"}
            return {"status": "failed", "message": "Redis ping returned False"}
        except Exception as e:
            return {"status": "failed", "message": str(e), "error": str(e)}
