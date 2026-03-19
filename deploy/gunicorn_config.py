# =============================================================================
# Gunicorn Configuration for LiteLLM Proxy (Production)
# =============================================================================
#
# Usage:
#   gunicorn -c deploy/gunicorn_config.py litellm.proxy.proxy_server:app
#
# All settings can be overridden via environment variables for container
# deployments without rebuilding images.
# =============================================================================

import multiprocessing
import os

# -----------------------------------------------------------------------------
# Server Socket
# -----------------------------------------------------------------------------

# Bind address and port.
# PORT env var allows container orchestrators (Kubernetes, ECS) to control the
# listening port without modifying the config.
bind = f"0.0.0.0:{os.getenv('PORT', '4000')}"

# Number of seconds to wait for requests on a keep-alive connection.
# Balances connection reuse vs resource consumption.
keepalive = 5

# -----------------------------------------------------------------------------
# Worker Processes
# -----------------------------------------------------------------------------

# Number of worker processes.
# Default formula (2 * CPU + 1) is the Gunicorn recommendation for a mix of
# CPU-bound and I/O-bound workloads. Override with GUNICORN_WORKERS when
# running in containers with cgroup CPU limits, since multiprocessing.cpu_count()
# may report the host CPU count rather than the container's allocated CPUs.
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Use Uvicorn's worker class so each Gunicorn worker runs the FastAPI/ASGI app
# via uvicorn's high-performance event loop.
worker_class = "uvicorn.workers.UvicornWorker"

# -----------------------------------------------------------------------------
# Timeouts
# -----------------------------------------------------------------------------

# Workers silent for more than this many seconds are killed and restarted.
# LLM API calls (especially streaming completions) can take a while, so 120s
# accommodates long-running inference requests.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Timeout for graceful worker shutdown. After receiving a restart/stop signal,
# workers have this many seconds to finish serving in-flight requests before
# being forcefully killed.
graceful_timeout = 30

# -----------------------------------------------------------------------------
# Worker Recycling (Memory Leak Prevention)
# -----------------------------------------------------------------------------

# Restart a worker after it has processed this many requests. Prevents gradual
# memory leaks from accumulating in long-running workers.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))

# Add random jitter (0 to max_requests_jitter) to max_requests per worker.
# This staggers worker restarts so they don't all recycle simultaneously,
# which would cause a brief capacity dip.
max_requests_jitter = 50

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

# "-" sends logs to stdout/stderr, which is the standard practice for
# containerized applications (collected by Docker/Kubernetes log drivers).
accesslog = "-"
errorlog = "-"

# Log level for Gunicorn's internal messages. "info" captures startup,
# worker lifecycle, and request-level events without being too noisy.
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Log format including response time for monitoring and alerting.
# %(t)s       - timestamp
# %(h)s       - remote address
# %(r)s       - request line (method + path + protocol)
# %(s)s       - HTTP status code
# %(b)s       - response length
# %(D)s       - request duration in microseconds
access_log_format = '%(t)s %(h)s "%(r)s" %(s)s %(b)s %(D)sμs'

# -----------------------------------------------------------------------------
# Server Hooks
# -----------------------------------------------------------------------------


def on_starting(server):
    """Called just before the master process is initialized.

    Useful for logging deployment configuration at startup for debugging.
    """
    server.log.info("LiteLLM Gunicorn server starting")
    server.log.info(f"  Workers: {server.app.cfg.workers}")
    server.log.info(f"  Worker class: {server.app.cfg.worker_class}")
    server.log.info(f"  Bind: {server.app.cfg.bind}")
    server.log.info(f"  Timeout: {server.app.cfg.timeout}s")
    server.log.info(f"  Max requests: {server.app.cfg.max_requests}")


def pre_fork(server, worker):
    """Called just before a worker is forked.

    Runs in the master process. Useful for pre-fork setup such as closing
    shared database connections that shouldn't be inherited by children.
    """
    server.log.info(f"Pre-fork: spawning worker (pid will be assigned)")


def post_fork(server, worker):
    """Called just after a worker has been forked.

    Runs in the child (worker) process. Each worker gets its own event loop
    and database connections, so this is the right place for per-worker
    initialization logging.
    """
    server.log.info(f"Post-fork: worker spawned (pid: {worker.pid})")


def worker_exit(server, worker):
    """Called when a worker process exits.

    Useful for logging worker lifecycle events and detecting unexpected exits.
    """
    server.log.info(f"Worker exited (pid: {worker.pid})")
