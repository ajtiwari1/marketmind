"""Decorator helpers for instrumenting functions with Langfuse."""

import inspect
import logging
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, cast

from langfuse import observe

from ..client import get_langfuse_client

P = ParamSpec("P")
R = TypeVar("R")


class _LangfuseLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        client = get_langfuse_client()
        with client.start_as_current_observation(
            as_type="span",
            name=f"log.{record.levelname.lower()}",
            input={"message": self.format(record)},
            metadata={
                "logger": record.name,
                "level": record.levelname,
                "module": record.module,
            },
        ):
            pass


def trace_logs(
    *,
    logger_name: str,
    name: str | None = None,
    level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Trace a function and record the selected logger's messages as child spans."""
    if not logger_name:
        raise ValueError("logger_name must be a non-empty logger name")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        traced_function = observe(
            name=name or function.__name__,
            as_type="span",
        )(function)
        logger = logging.getLogger(logger_name)

        def add_handler() -> _LangfuseLogHandler:
            handler = _LangfuseLogHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            return handler

        def remove_handler(handler: _LangfuseLogHandler) -> None:
            logger.removeHandler(handler)
            handler.close()

        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                handler = add_handler()
                try:
                    return await traced_function(*args, **kwargs)  # type: ignore[misc]
                finally:
                    remove_handler(handler)

            return cast(Callable[P, R], async_wrapper)

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            handler = add_handler()
            try:
                return traced_function(*args, **kwargs)
            finally:
                remove_handler(handler)

        return wrapper

    return decorate


__all__ = ["observe", "trace_logs"]
