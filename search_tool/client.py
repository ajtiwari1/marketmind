"""MCP client adapter for the official Tavily local search server."""

import asyncio
import json
import math
import os
import re
from typing import Any
from urllib.parse import urlparse

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

from .config import TavilyMCPConfig
from .models import WebSearchResult


class SearchToolError(RuntimeError):
    """Raised when the Tavily MCP server or its response is unusable."""


class SearchToolConfigurationError(ValueError):
    """Raised when the Tavily MCP client is not configured."""


def normalize_results(payload: Any, *, limit: int) -> list[WebSearchResult]:
    """Validate Tavily search output and normalize its result records."""
    if isinstance(payload, dict):
        records = payload.get("results")
    else:
        records = payload
    if not isinstance(records, list):
        raise SearchToolError("Tavily search returned no results list")

    normalized: list[WebSearchResult] = []
    for record in records[:limit]:
        if not isinstance(record, dict):
            raise SearchToolError("Tavily returned a malformed search result")
        title = record.get("title")
        url = record.get("url")
        snippet = record.get("content") or record.get("snippet")
        if not isinstance(title, str) or not title.strip():
            raise SearchToolError("Tavily returned a result without a title")
        if not isinstance(url, str):
            raise SearchToolError("Tavily returned a result without a URL")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise SearchToolError("Tavily returned an invalid result URL")
        if not isinstance(snippet, str) or not snippet.strip():
            raise SearchToolError("Tavily returned a result without a content snippet")
        score = record.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise SearchToolError("Tavily returned an invalid result score")
        published_date = record.get("published_date")
        if published_date is not None and not isinstance(published_date, str):
            raise SearchToolError("Tavily returned an invalid published date")
        normalized.append(
            WebSearchResult(
                title=title.strip(),
                url=url,
                snippet=snippet.strip(),
                score=float(score) if score is not None else None,
                published_date=published_date,
            )
        )
    return normalized


def _parse_tool_result(result: Any, *, limit: int) -> list[WebSearchResult]:
    if result.is_error:
        raise SearchToolError("Tavily search tool returned an error")
    structured = result.structured_content
    if structured is not None:
        return normalize_results(structured, limit=limit)
    for item in result.content:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = _parse_formatted_results(text)
            if payload is None:
                continue
        return normalize_results(payload, limit=limit)
    raise SearchToolError("Tavily search returned no parseable result content")


def _parse_formatted_results(text: str) -> list[dict[str, str]] | None:
    """Parse the human-readable result format returned by Tavily's MCP server."""
    pattern = re.compile(
        r"(?:^|\n\n)Title: (?P<title>[^\n]*)\n"
        r"(?:ID: [^\n]*\n)?"
        r"URL: (?P<url>[^\n]*)\n"
        r"Content: (?P<content>.*?)"
        r"(?=\n(?:Raw Content|Favicon): |\n\n(?:Title: |Images:)|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return [
        {
            "title": match.group("title"),
            "url": match.group("url"),
            "content": match.group("content").strip(),
        }
        for match in matches
    ]


class TavilyMCPClient:
    def __init__(self, config: TavilyMCPConfig | None = None) -> None:
        self.config = config or TavilyMCPConfig.from_env()
        self.config.validate()

    async def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if not self.config.api_key:
            raise SearchToolConfigurationError(
                "Tavily API key is missing. Set TAVILY_API_KEY for web or hybrid search."
            )
        self.config.validate()

        environment = {
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "TAVILY_API_KEY": self.config.api_key,
        }
        server = StdioServerParameters(
            command=self.config.command,
            args=["-y", self.config.package],
            env=environment,
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with stdio_client(server) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tool_list = await session.list_tools()
                        tool = next(
                            (
                                item
                                for item in tool_list.tools
                                if item.name == self.config.tool_name
                            ),
                            None,
                        )
                        if tool is None:
                            raise SearchToolError(
                                f"Configured Tavily tool '{self.config.tool_name}' "
                                "was not found"
                            )
                        properties = tool.input_schema.get("properties", {})
                        if "query" not in properties:
                            raise SearchToolError(
                                f"Tavily tool '{self.config.tool_name}' does not "
                                "accept a query argument"
                            )
                        arguments: dict[str, Any] = {"query": query.strip()}
                        if "max_results" in properties:
                            arguments["max_results"] = max_results
                        result = await session.call_tool(
                            self.config.tool_name,
                            arguments=arguments,
                            read_timeout_seconds=self.config.timeout_seconds,
                        )
                        return _parse_tool_result(result, limit=max_results)
        except TimeoutError as error:
            raise SearchToolError(
                f"Tavily MCP search exceeded {self.config.timeout_seconds:g} seconds"
            ) from error
        except MCPError as error:
            raise SearchToolError("Tavily MCP request failed") from error
        except (anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.EndOfStream) as error:
            raise SearchToolError("Tavily MCP server closed the connection") from error
        except FileNotFoundError as error:
            raise SearchToolConfigurationError(
                f"Could not find '{self.config.command}'. Install Node.js and ensure "
                "npx is available on PATH to run the Tavily MCP server."
            ) from error
        except (OSError, RuntimeError) as error:
            if isinstance(error, SearchToolError):
                raise
            raise SearchToolError(
                f"Could not run Tavily MCP server ({type(error).__name__})"
            ) from error
