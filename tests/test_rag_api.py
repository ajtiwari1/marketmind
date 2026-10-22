import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from chat.generation import GeneratedAnswer, GeminiConfigurationError
from middleware.app import app
from chat.retrieval import RetrievedChunk
from search_tool.client import SearchToolConfigurationError, SearchToolError
from search_tool.models import WebSearchResult


def make_result() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer="Grounded response [1]",
        model="gemini-test",
        sources=(
            RetrievedChunk(
                chunk_id="chunk-id",
                document_id="document-id",
                text="Supporting document text",
                source="file:///guide.md",
                file_name="guide.md",
                file_type="text/markdown",
                chunk_index=2,
                score=0.82,
                metadata={"heading_path": ["Guide"]},
            ),
        ),
    )


class AskEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_returns_answer_and_source_metadata(self) -> None:
        with patch("middleware.endpoints.answer_question", return_value=make_result()) as ask:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "How do I configure it?",
                    "top_k": 3,
                    "document_ids": ["doc-a"],
                    "file_types": ["text/markdown"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Grounded response [1]")
        self.assertEqual(response.json()["model"], "gemini-test")
        self.assertEqual(response.json()["sources"][0]["chunk_id"], "chunk-id")
        self.assertEqual(
            response.json()["sources"][0]["metadata"]["heading_path"],
            ["Guide"],
        )
        args = ask.call_args.kwargs
        self.assertEqual(ask.call_args.args[0], "How do I configure it?")
        self.assertEqual(args["top_k"], 3)
        self.assertEqual(args["filters"].document_ids, ("doc-a",))
        self.assertEqual(args["filters"].file_types, ("text/markdown",))

    def test_search_mode_defaults_to_local(self) -> None:
        with patch(
            "middleware.endpoints.answer_question",
            return_value=make_result(),
        ) as ask:
            response = self.client.post("/api/ask", json={"question": "Question?"})

        self.assertEqual(response.status_code, 200)
        ask.assert_called_once()

    def test_web_mode_returns_web_citations(self) -> None:
        web_source = WebSearchResult(
            title="Web result",
            url="https://example.com/article",
            snippet="Web snippet",
            published_date="2026-10-20",
        )
        result = GeneratedAnswer(
            answer="Web-grounded answer [1]",
            model="gemini-test",
            sources=(web_source,),
        )
        with (
            patch("middleware.endpoints.TavilyMCPClient") as client_class,
            patch("middleware.endpoints.answer_with_sources", return_value=result) as answer,
        ):
            client_class.return_value.search = AsyncMock(return_value=[web_source])
            response = self.client.post(
                "/api/ask",
                json={"question": "Question?", "search_mode": "web"},
            )

        self.assertEqual(response.status_code, 200)
        source = response.json()["sources"][0]
        self.assertEqual(source["source_type"], "web")
        self.assertEqual(source["url"], "https://example.com/article")
        self.assertEqual(source["published_date"], "2026-10-20")
        answer.assert_called_once()

    def test_hybrid_mode_combines_local_and_web_sources(self) -> None:
        local_source = make_result().sources[0]
        web_source = WebSearchResult(
            title="Web result",
            url="https://example.com",
            snippet="Web snippet",
        )
        result = GeneratedAnswer(
            answer="Combined answer [1][2]",
            model="gemini-test",
            sources=(local_source, web_source),
        )
        with (
            patch("middleware.endpoints.retrieve", return_value=[local_source]),
            patch("middleware.endpoints.TavilyMCPClient") as client_class,
            patch("middleware.endpoints.answer_with_sources", return_value=result) as answer,
        ):
            client_class.return_value.search = AsyncMock(return_value=[web_source])
            response = self.client.post(
                "/api/ask",
                json={
                    "question": "Question?",
                    "search_mode": "hybrid",
                    "document_ids": ["doc-a"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["source_type"] for item in response.json()["sources"]],
            ["local", "web"],
        )
        self.assertEqual(len(answer.call_args.args[1]), 2)

    def test_rejects_blank_question_and_out_of_range_top_k(self) -> None:
        blank = self.client.post("/api/ask", json={"question": "  "})
        bad_top_k = self.client.post(
            "/api/ask",
            json={"question": "Question?", "top_k": 0},
        )

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(bad_top_k.status_code, 422)

    def test_returns_service_unavailable_when_gemini_key_is_missing(self) -> None:
        with patch(
            "middleware.endpoints.answer_question",
            side_effect=GeminiConfigurationError("Gemini API key is missing"),
        ):
            response = self.client.post("/api/ask", json={"question": "Question?"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("Gemini API key is missing", response.json()["detail"])

    def test_returns_gateway_error_for_provider_failures(self) -> None:
        with patch(
            "middleware.endpoints.answer_question",
            side_effect=RuntimeError("private provider detail"),
        ):
            response = self.client.post("/api/ask", json={"question": "Question?"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Retrieval or answer generation service failed",
        )
        self.assertNotIn("private provider detail", response.text)

    def test_returns_service_unavailable_when_tavily_key_is_missing(self) -> None:
        with patch(
            "middleware.endpoints.TavilyMCPClient",
            side_effect=SearchToolConfigurationError("TAVILY_API_KEY is missing"),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "Question?", "search_mode": "web"},
            )

        self.assertEqual(response.status_code, 503)

    def test_returns_gateway_error_for_mcp_failures(self) -> None:
        with patch(
            "middleware.endpoints.TavilyMCPClient"
        ) as client_class:
            client_class.return_value.search = AsyncMock(
                side_effect=SearchToolError("private MCP detail")
            )
            response = self.client.post(
                "/api/ask",
                json={"question": "Question?", "search_mode": "web"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private MCP detail", response.text)

    def test_local_filters_are_rejected_in_web_only_mode(self) -> None:
        response = self.client.post(
            "/api/ask",
            json={
                "question": "Question?",
                "search_mode": "web",
                "document_ids": ["doc-a"],
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
