"""
Vectorless RAG — Structured Logging
======================================
Configures application-wide logging with structured JSON output
for production and readable console output for development.
"""

import os
import sys
import logging
import time
import functools
from flask import request, g


def setup_logging(app):
    """
    Configure logging for the Flask application.

    In development: Readable console output with colors.
    In production: JSON-structured output for log aggregation.

    Args:
        app: The Flask application instance.
    """
    log_level = logging.DEBUG if app.debug else logging.INFO

    # Clear existing handlers
    app.logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if app.debug:
        # Development: Human-readable format
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        # Production: Structured format (key=value for easy parsing)
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S"
        )

    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)

    # Also configure the root logger for third-party libraries
    logging.getLogger().setLevel(logging.WARNING)

    # Register request logging middleware
    _register_request_logging(app)


def _register_request_logging(app):
    """
    Add before/after request hooks for automatic request logging.
    Logs method, path, status code, and response time for every request.
    """

    @app.before_request
    def log_request_start():
        """Record request start time."""
        g.request_start_time = time.time()

    @app.after_request
    def log_request_end(response):
        """Log request completion with timing."""
        duration = time.time() - getattr(g, "request_start_time", time.time())
        duration_ms = round(duration * 1000, 2)

        # Skip logging for static files in development
        if request.path.startswith("/static/"):
            return response

        app.logger.info(
            f"{request.method} {request.path} -> {response.status_code} ({duration_ms}ms)"
        )
        return response


def get_logger(name):
    """
    Get a named logger for use in service modules.

    Args:
        name: The logger name (typically __name__ of the calling module).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def log_performance(func):
    """
    Decorator that logs the execution time of a function.

    Usage:
        @log_performance
        def my_slow_function():
            ...
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f">> {func.__name__}() started")
        try:
            result = func(*args, **kwargs)
            elapsed = round(time.time() - start, 2)
            logger.info(f"OK {func.__name__}() completed in {elapsed}s")
            return result
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"FAIL {func.__name__}() failed after {elapsed}s: {e}")
            raise

    return wrapper
