import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter

import litellm
from litellm._logging import verbose_proxy_logger

router = APIRouter()

_process_start_time = time.monotonic()


def _get_memory_rss_mb() -> Optional[float]:
    """Return current process RSS memory in MB, or None if unavailable."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        pass

    try:
        import resource

        # resource.getrusage returns RSS in KB on Linux
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return round(rusage.ru_maxrss / 1024, 2)
    except (ImportError, AttributeError):
        pass

    return None


def _get_active_models_count() -> int:
    """Return the number of models configured in the proxy."""
    try:
        from litellm.proxy.proxy_server import llm_model_list

        if llm_model_list is not None:
            return len(llm_model_list)
    except ImportError:
        verbose_proxy_logger.debug("info_endpoint: could not import llm_model_list")
    return 0


def _get_cache_stats() -> Dict[str, Any]:
    """Return in-memory cache statistics if available."""
    stats: Dict[str, Any] = {}
    try:
        from litellm.proxy.proxy_server import user_api_key_cache

        in_memory_cache = getattr(user_api_key_cache, "in_memory_cache", None)
        if in_memory_cache is not None:
            cache_dict = getattr(in_memory_cache, "cache_dict", None)
            if cache_dict is not None:
                stats["in_memory_cache_size"] = len(cache_dict)
    except (ImportError, AttributeError):
        verbose_proxy_logger.debug(
            "info_endpoint: could not retrieve cache stats"
        )
    return stats


@router.get(
    "/info",
    tags=["health"],
)
async def info_endpoint():
    """
    Unprotected endpoint exposing system health metrics.

    Returns basic runtime information useful for Kubernetes probes,
    monitoring dashboards, and operational visibility.

    No authentication required.
    """
    uptime_seconds = round(time.monotonic() - _process_start_time, 2)

    return {
        "version": litellm.version,
        "uptime_seconds": uptime_seconds,
        "memory_rss_mb": _get_memory_rss_mb(),
        "python_version": sys.version,
        "active_models_count": _get_active_models_count(),
        "cache_stats": _get_cache_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
