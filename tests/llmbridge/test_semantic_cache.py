"""Tests for litellm.caching.semantic_cache – pure-logic unit tests."""

import math

import pytest

from litellm.caching.semantic_cache import SemanticCacheConfig, cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Pure math tests – no I/O."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors_high_similarity(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.05, 2.95]
        sim = cosine_similarity(a, b)
        assert sim > 0.99, f"Expected similarity > 0.99, got {sim}"

    def test_zero_vector_a(self):
        """Zero vector should return 0.0, not raise."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        z = [0.0, 0.0]
        assert cosine_similarity(z, z) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_unit_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_negative_dot_product(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_single_element_vectors(self):
        assert cosine_similarity([5.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([5.0], [-5.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# SemanticCacheConfig
# ---------------------------------------------------------------------------


class TestSemanticCacheConfig:
    def test_default_values(self):
        cfg = SemanticCacheConfig()
        assert cfg.enabled is False
        assert cfg.similarity_threshold == 0.90
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.redis_url is None
        assert cfg.ttl == 3600
        assert cfg.max_cache_size == 10000

    def test_custom_values(self):
        cfg = SemanticCacheConfig(
            enabled=True,
            similarity_threshold=0.75,
            embedding_model="text-embedding-ada-002",
            redis_url="redis://localhost:6379",
            ttl=7200,
            max_cache_size=5000,
        )
        assert cfg.enabled is True
        assert cfg.similarity_threshold == 0.75
        assert cfg.redis_url == "redis://localhost:6379"

    def test_threshold_bounds_low(self):
        cfg = SemanticCacheConfig(similarity_threshold=0.0)
        assert cfg.similarity_threshold == 0.0

    def test_threshold_bounds_high(self):
        cfg = SemanticCacheConfig(similarity_threshold=1.0)
        assert cfg.similarity_threshold == 1.0


# ---------------------------------------------------------------------------
# SemanticCache init (no Redis)
# ---------------------------------------------------------------------------


class TestSemanticCacheInitNoRedis:
    def test_init_without_redis_url_raises(self):
        """Without redis_url and no REDIS_URL env var, init should raise."""
        import os

        old = os.environ.pop("REDIS_URL", None)
        try:
            from litellm.caching.semantic_cache import SemanticCache

            with pytest.raises((ValueError, Exception)):
                SemanticCache()
        finally:
            if old is not None:
                os.environ["REDIS_URL"] = old


# ---------------------------------------------------------------------------
# get_stats structure (tested via mock-free path)
# ---------------------------------------------------------------------------


class TestGetStatsStructure:
    def test_stats_keys_present(self):
        """Verify get_stats returns expected keys even without Redis."""
        import os

        old = os.environ.pop("REDIS_URL", None)
        try:
            from litellm.caching.semantic_cache import SemanticCache

            # Create with a dummy redis_url so __init__ doesn't error on the url
            # but the actual connection is never used for stats.
            try:
                cache = SemanticCache(redis_url="redis://localhost:9999")
            except Exception:
                pytest.skip("Cannot construct SemanticCache without redis libs")
                return

            stats = cache.get_stats()
            assert "total_lookups" in stats
            assert "cache_hits" in stats
            assert "cache_misses" in stats
            assert "hit_rate" in stats
            assert "estimated_savings" in stats
            assert stats["total_lookups"] == 0
            assert stats["hit_rate"] == 0.0
        finally:
            if old is not None:
                os.environ["REDIS_URL"] = old
