import logging
import structlog
import structlog.contextvars
from structlog.stdlib import add_log_level
from structlog.processors import JSONRenderer, TimeStamper
from structlog.contextvars import merge_contextvars


def configure_logging() -> None:
    """Configure structlog with correlation ID support and JSON output."""
    structlog.configure(
        processors=[
            merge_contextvars,
            add_log_level,
            TimeStamper(fmt="iso"),
            JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging_level=20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
