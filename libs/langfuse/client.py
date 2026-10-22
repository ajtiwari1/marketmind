"""Client helpers backed by the official Langfuse Python SDK."""

from langfuse import Langfuse, get_client


def get_langfuse_client() -> Langfuse:
    """Return the SDK-managed Langfuse client."""
    return get_client()


def flush_langfuse() -> None:
    """Flush queued observations before an application exits."""
    get_langfuse_client().flush()
