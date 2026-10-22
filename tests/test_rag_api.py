import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from chat.generation import GeneratedAnswer, GeminiConfigurationError
from middleware.app import app
from chat.retrieval import RetrievedChunk


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


if __name__ == "__main__":
    unittest.main()
