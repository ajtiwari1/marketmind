# Plan: External search through MCP

## Goal

Add Tavily external search to the question-answering API through Tavily's
official local Model Context Protocol (MCP) server. Keep the MCP client in the
top-level `search_tool/` package, outside `chat/`, so the RAG core remains
reusable and independent from the search provider.

The existing local Qdrant hybrid search remains the primary knowledge source.
The API caller selects local, web, or hybrid search per request. Web search
never silently replaces local retrieval.

## Proposed layout

```text
search_tool/
  __init__.py
  config.py
  client.py
  models.py
middleware/
  endpoints.py
tests/
  test_mcp_search.py
  test_mcp_search_api.py
```

`search_tool/` owns MCP connection lifecycle, tool discovery/calling, and
normalization of provider responses. `middleware/` calls its typed interface;
`chat/` stays independent of MCP and web-search provider details.

## Implementation stages

1. **Confirm the search service contract** — implemented
   - Use Tavily's official `tavily-mcp@latest` server through local stdio.
   - Authenticate using the `TAVILY_API_KEY` environment variable.
   - Discover and invoke only Tavily's `tavily_search` tool.
   - Keep provider-specific names and credentials out of the RAG pipeline.

2. **Define a typed external-search interface** — implemented
   - Accept a non-empty query, bounded result count, and optional domain or
     recency filters only if the selected provider supports them.
   - Return normalized results with title, URL, snippet, and optional published
     date/provider metadata.
   - Validate tool output and reject malformed or empty results explicitly.

3. **Implement MCP connection and tool invocation** — implemented
   - Use the official Python MCP client SDK and the selected server transport.
   - Discover available tools, verify the configured search tool exists, and
     pass only supported arguments.
   - Manage connection and session lifetime with async context managers;
     do not leave subprocesses or network sessions running after a request.
   - Add bounded timeouts and clear errors for unavailable servers, tool
     failures, and invalid responses. Do not silently return fake results.

4. **Integrate external results with question answering** — implemented
   - Add an explicit request option such as `search_mode: "local" | "web" |
     "hybrid"`; retain `"local"` as the default.
   - `"local"` uses existing Qdrant retrieval only.
   - `"web"` uses the MCP search results as grounding context.
   - `"hybrid"` combines local chunks and external results while preserving
     source type and attribution.
   - Pass external snippets to Gemini as untrusted reference data, just like
     locally retrieved chunks; preserve citations in the API response.
   - Keep the existing request shape and response behavior compatible when the
     new option is omitted.

5. **Configuration, observability, and safety** — implemented
   - Store MCP transport, endpoint/command, tool name, and timeouts in
     environment-backed settings; never commit credentials or print secrets.
   - Do not put raw questions, full snippets, or provider credentials in
     Langfuse traces or routine logs.
   - Treat returned pages and snippets as untrusted content; do not allow
     retrieved instructions to trigger tools or change application policy.
   - Configure outbound network access deliberately and document data sent to
     the external provider.

6. **Test and document** — implemented; live smoke test pending prerequisites
   - Unit-test tool discovery, request construction, result normalization,
     invalid outputs, timeouts, and cleanup using a fake MCP session/server.
   - API-test each search mode, backward-compatible defaults, provider errors,
     and source/citation formatting without live external calls.
   - Keep ordinary tests independent from the network and credentials. Mocked
     client and endpoint tests are implemented.
   - Run an opt-in smoke test against the real MCP server once Node.js and a
     Tavily API key are available in the runtime environment.
   - Document provider setup, environment variables, privacy implications,
     startup requirements, and example API requests.

## Proposed API behavior

The exact schema should be settled during implementation. A possible request:

```json
{
  "question": "What changed in the latest release?",
  "top_k": 5,
  "search_mode": "hybrid"
}
```

Responses should identify the origin of every citation (`local` or `web`) and
include a URL for external citations. The server should return explicit errors
when a requested mode cannot run, rather than silently omitting that source.

## Non-goals for the first implementation

- Building or hosting a general-purpose MCP server unless the selected provider
  requires a project-owned adapter.
- Crawling arbitrary URLs, bypassing site restrictions, or scraping without
  provider-approved access.
- Automatically using external search for every request.
- Persisting web results in Qdrant without a separate ingestion and freshness
  policy.
- Browser automation, autonomous actions, or arbitrary MCP tool execution.
- Query rewriting, web-page fetching beyond what the selected search tool
  returns, and learned ranking/fusion.

## Configuration and use

- Install Node.js 20 or newer and ensure `npx` is on the API process `PATH`.
- Set `TAVILY_API_KEY` in the environment or ignored `.env` file.
- `POST /api/ask` accepts `search_mode`: `local` (default), `web`, or `hybrid`.
- `top_k` controls the local result count and Tavily's maximum web-result count.
- Web-only mode does not accept local `document_ids` or `file_types` filters.
- The MCP subprocess receives only `PATH`, `HOME`, and `TAVILY_API_KEY`.
- Each web search starts and closes a local MCP subprocess/session. The request
  is bounded by `TAVILY_MCP_TIMEOUT_SECONDS` (default 20 seconds).

The code and mocked test coverage are implemented. The live Tavily integration
has not been run in this environment because Node.js/npx is unavailable; it
also requires a Tavily API key and network access. Normal unit tests use mocked
MCP sessions and do not require these external prerequisites.
