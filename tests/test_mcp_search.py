import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from search_tool.client import (
    SearchToolError,
    TavilyMCPClient,
    _parse_tool_result,
    normalize_results,
)
from search_tool.config import TavilyMCPConfig
from search_tool.models import WebSearchResult


class TavilyMCPClientTests(unittest.TestCase):
    def test_normalizes_tavily_records_and_applies_limit(self) -> None:
        records = [
            {
                "title": "First result",
                "url": "https://example.com/one",
                "content": "First snippet",
                "score": 0.8,
                "published_date": "2026-10-20",
            },
            {
                "title": "Second result",
                "url": "https://example.com/two",
                "snippet": "Second snippet",
            },
        ]

        results = normalize_results({"results": records}, limit=1)

        self.assertEqual(
            results,
            [
                WebSearchResult(
                    title="First result",
                    url="https://example.com/one",
                    snippet="First snippet",
                    score=0.8,
                    published_date="2026-10-20",
                )
            ],
        )

    def test_rejects_malformed_records_and_urls(self) -> None:
        with self.assertRaisesRegex(SearchToolError, "results list"):
            normalize_results({"items": []}, limit=5)
        with self.assertRaisesRegex(SearchToolError, "invalid result URL"):
            normalize_results(
                [{"title": "Bad", "url": "javascript:alert(1)", "content": "text"}],
                limit=5,
            )
        with self.assertRaisesRegex(SearchToolError, "without a title"):
            normalize_results(
                [{"url": "https://example.com", "content": "text"}],
                limit=5,
            )

    def test_rejects_mcp_tool_error_result(self) -> None:
        result = SimpleNamespace(is_error=True, structured_content=None, content=[])

        with self.assertRaisesRegex(SearchToolError, "returned an error"):
            _parse_tool_result(result, limit=5)

    def test_parses_formatted_text_from_tavily_mcp_server(self) -> None:
        result = SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[
                SimpleNamespace(
                    text=(
                        "Answer: A short summary\n"
                        "Detailed Results:\n\n"
                        "Title: First result\n"
                        "ID: result-1\n"
                        "URL: https://example.com/one\n"
                        "Content: First snippet.\n\n"
                        "Title: Second result\n"
                        "ID: result-2\n"
                        "URL: https://example.com/two\n"
                        "Content: Second snippet.\n"
                        "Raw Content: Longer page text."
                    )
                )
            ],
        )

        results = _parse_tool_result(result, limit=1)

        self.assertEqual(
            results,
            [
                WebSearchResult(
                    title="First result",
                    url="https://example.com/one",
                    snippet="First snippet.",
                )
            ],
        )

    def test_opens_official_server_and_invokes_only_search_tool(self) -> None:
        server_calls: list[object] = []

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            initialize = AsyncMock()
            list_tools = AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="tavily_search",
                            input_schema={
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_results": {"type": "integer"},
                                    "search_depth": {"type": "string"},
                                }
                            },
                        )
                    ]
                )
            )
            call_tool = AsyncMock(
                return_value=SimpleNamespace(
                    is_error=False,
                    structured_content={
                        "results": [
                            {
                                "title": "Result",
                                "url": "https://example.com",
                                "content": "Snippet",
                            }
                        ]
                    },
                    content=[],
                )
            )

        session = FakeSession()

        @asynccontextmanager
        async def fake_stdio(server: object):
            server_calls.append(server)
            yield (object(), object())

        client = TavilyMCPClient(
            TavilyMCPConfig(
                api_key="test-key",
                timeout_seconds=2,
            )
        )
        with (
            patch("search_tool.client.stdio_client", side_effect=fake_stdio),
            patch("search_tool.client.ClientSession", return_value=session),
        ):
            results = asyncio.run(client.search(" latest changes ", max_results=3))

        self.assertEqual(results[0].title, "Result")
        server = server_calls[0]
        self.assertEqual(server.command, "npx")
        self.assertEqual(server.args, ["-y", "tavily-mcp@latest"])
        self.assertEqual(server.env["TAVILY_API_KEY"], "test-key")
        self.assertNotIn("LANGFUSE_SECRET_KEY", server.env)
        session.call_tool.assert_awaited_once_with(
            "tavily_search",
            arguments={"query": "latest changes", "max_results": 3},
            read_timeout_seconds=2,
        )

    def test_missing_key_and_invalid_result_count_fail_before_spawn(self) -> None:
        client = TavilyMCPClient(TavilyMCPConfig(api_key=None))
        with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
            asyncio.run(client.search("query"))
        client_with_key = TavilyMCPClient(TavilyMCPConfig(api_key="test-key"))
        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            asyncio.run(client_with_key.search("query", max_results=0))


if __name__ == "__main__":
    unittest.main()
