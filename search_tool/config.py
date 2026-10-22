"""Environment-backed settings for the Tavily MCP client."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TavilyMCPConfig:
    api_key: str | None = None
    command: str = "npx"
    package: str = "tavily-mcp@latest"
    tool_name: str = "tavily_search"
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "TavilyMCPConfig":
        config = cls(
            api_key=os.getenv("TAVILY_API_KEY") or None,
            command=os.getenv("TAVILY_MCP_COMMAND", "npx"),
            package=os.getenv("TAVILY_MCP_PACKAGE", "tavily-mcp@latest"),
            tool_name=os.getenv("TAVILY_MCP_TOOL_NAME", "tavily_search"),
            timeout_seconds=float(os.getenv("TAVILY_MCP_TIMEOUT_SECONDS", "20")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.command.strip():
            raise ValueError("TAVILY_MCP_COMMAND must not be empty")
        if not self.package.strip():
            raise ValueError("TAVILY_MCP_PACKAGE must not be empty")
        if not self.tool_name.strip():
            raise ValueError("TAVILY_MCP_TOOL_NAME must not be empty")
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError(
                "TAVILY_MCP_TIMEOUT_SECONDS must be between 0 and 120 seconds"
            )
