"""Tests for litellm.proxy.health_endpoints.info_endpoint – pure-logic unit tests."""

from unittest.mock import patch

import pytest

from litellm.proxy.health_endpoints.info_endpoint import (
    _get_memory_rss_mb,
    info_endpoint,
)


# ---------------------------------------------------------------------------
# _get_memory_rss_mb
# ---------------------------------------------------------------------------


class TestGetMemoryRssMb:
    def test_returns_float_or_none(self):
        result = _get_memory_rss_mb()
        assert result is None or isinstance(result, float)

    def test_positive_if_available(self):
        result = _get_memory_rss_mb()
        if result is not None:
            assert result > 0


# ---------------------------------------------------------------------------
# info_endpoint
# ---------------------------------------------------------------------------


class TestInfoEndpoint:
    @pytest.mark.asyncio
    async def test_returns_correct_structure(self):
        with patch("litellm.version", "0.0.0-test", create=True):
            data = await info_endpoint()
        assert isinstance(data, dict)
        assert "version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_version_is_string(self):
        with patch("litellm.version", "1.2.3", create=True):
            data = await info_endpoint()
        assert isinstance(data["version"], str)

    @pytest.mark.asyncio
    async def test_uptime_is_positive(self):
        with patch("litellm.version", "0.0.0", create=True):
            data = await info_endpoint()
        assert data["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_timestamp_is_iso_format(self):
        from datetime import datetime

        with patch("litellm.version", "0.0.0", create=True):
            data = await info_endpoint()
        dt = datetime.fromisoformat(data["timestamp"])
        assert dt is not None

    @pytest.mark.asyncio
    async def test_python_version_present(self):
        with patch("litellm.version", "0.0.0", create=True):
            data = await info_endpoint()
        assert "python_version" in data
        assert isinstance(data["python_version"], str)

    @pytest.mark.asyncio
    async def test_memory_field_present(self):
        with patch("litellm.version", "0.0.0", create=True):
            data = await info_endpoint()
        assert "memory_rss_mb" in data
