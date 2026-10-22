"""Typed, normalized web search results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float | None = None
    published_date: str | None = None
    source_type: str = "web"

    @property
    def text(self) -> str:
        return self.snippet

    @property
    def source(self) -> str:
        return self.url

    @property
    def file_name(self) -> str:
        return self.title

