"""Optional Langfuse tracing: active only when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set."""

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe
else:
    get_client = None
    observe = None


def traced(name: str, as_type: str = "span", capture_input: bool = False) -> Callable:
    """Decorator that records the call as a Langfuse observation (no-op when disabled)."""
    if not ENABLED:
        return lambda func: func
    return observe(name=name, as_type=as_type, capture_input=capture_input, capture_output=False)


def update(**fields: Any) -> None:
    """Set input/output/metadata on the current observation."""
    if ENABLED:
        get_client().update_current_span(**fields)


def flush() -> None:
    """Send buffered traces; short CLI runs exit before the background sender does."""
    if ENABLED:
        get_client().flush()


def current_trace_id() -> str | None:
    return get_client().get_current_trace_id() if ENABLED else None


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach an evaluation score (e.g. a RAGAS metric) to a trace."""
    if ENABLED and trace_id:
        get_client().create_score(trace_id=trace_id, name=name, value=value, comment=comment)
