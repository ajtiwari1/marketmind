import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch

from libs.langfuse.utils.decorators import trace_logs


class TraceLogsDecoratorTests(unittest.TestCase):
    @patch("libs.langfuse.utils.decorators.observe", side_effect=lambda **_: lambda fn: fn)
    @patch("libs.langfuse.utils.decorators.get_langfuse_client")
    def test_decorator_records_logs_and_removes_handler(
        self, get_client: MagicMock, _observe: MagicMock
    ) -> None:
        client = MagicMock()
        get_client.return_value = client
        logger = logging.getLogger("test.trace_logs.decorator")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        original_handlers = list(logger.handlers)

        @trace_logs(logger_name=logger.name, name="test-operation")
        def work() -> str:
            logger.info("work started")
            return "done"

        try:
            self.assertEqual(work(), "done")
            client.start_as_current_observation.assert_called_once_with(
                as_type="span",
                name="log.info",
                input={"message": "work started"},
                metadata={
                    "logger": logger.name,
                    "level": "INFO",
                    "module": "test_trace_logs_decorator",
                },
            )
            _observe.assert_called_once_with(name="test-operation", as_type="span")
        finally:
            self.assertEqual(logger.handlers, original_handlers)
            logger.setLevel(original_level)

    @patch("libs.langfuse.utils.decorators.observe", side_effect=lambda **_: lambda fn: fn)
    @patch("libs.langfuse.utils.decorators.get_langfuse_client")
    def test_handler_is_removed_when_decorated_function_raises(
        self, _get_client: MagicMock, _observe: MagicMock
    ) -> None:
        logger = logging.getLogger("test.trace_logs.exception")
        original_handlers = list(logger.handlers)

        @trace_logs(logger_name=logger.name)
        def fail() -> None:
            raise ValueError("expected failure")

        with self.assertRaisesRegex(ValueError, "expected failure"):
            fail()

        self.assertEqual(logger.handlers, original_handlers)
        _observe.assert_called_once()
        _get_client.assert_not_called()

    @patch("libs.langfuse.utils.decorators.observe", side_effect=lambda **_: lambda fn: fn)
    @patch("libs.langfuse.utils.decorators.get_langfuse_client")
    def test_async_function_is_traced(
        self, _get_client: MagicMock, _observe: MagicMock
    ) -> None:
        logger = logging.getLogger("test.trace_logs.async")

        @trace_logs(logger_name=logger.name)
        async def work() -> str:
            return "async done"

        self.assertEqual(asyncio.run(work()), "async done")
        _observe.assert_called_once()
        _get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
