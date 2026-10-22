import unittest
from unittest.mock import MagicMock, patch

from qdrant_client import models

from chat.config import RagConfig
from chat.embeddings.sparse import SparseVector
from chat.retrieval import RetrievalFilters, retrieve


class FakeDenseQueryModel:
    model_name = "fake-dense"
    dimension = 3

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector


class FakeSparseQueryModel:
    model_name = "fake-sparse"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        self.queries.extend(texts)
        return [SparseVector(indices=[1, 4], values=[0.6, 0.4]) for _ in texts]


class FakeSearchStore:
    def __init__(self, points: list[models.ScoredPoint] | None = None) -> None:
        self.points = points or []
        self.dimension: int | None = None
        self.model_validation_called = False
        self.search_arguments: dict[str, object] = {}

    def ensure_collection(self, vector_size: int) -> None:
        self.dimension = vector_size

    def validate_embedding_models(self) -> None:
        self.model_validation_called = True

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        *,
        limit: int,
        prefetch_limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        self.search_arguments = {
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
            "limit": limit,
            "prefetch_limit": prefetch_limit,
            "query_filter": query_filter,
        }
        return self.points


def make_point() -> models.ScoredPoint:
    return models.ScoredPoint(
        id="point-id",
        version=1,
        score=0.75,
        payload={
            "chunk_id": "chunk-id",
            "document_id": "document-id",
            "text": "Retrieved text",
            "source": "file:///guide.md",
            "file_name": "guide.md",
            "file_type": "text/markdown",
            "chunk_index": 2,
            "heading_path": ["Guide"],
        },
    )


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.langfuse_client = MagicMock()
        patcher = patch(
            "chat.retrieval.get_langfuse_client",
            return_value=self.langfuse_client,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_retrieve_embeds_query_fuses_results_and_maps_payload(self) -> None:
        dense = FakeDenseQueryModel()
        sparse = FakeSparseQueryModel()
        store = FakeSearchStore([make_point()])
        config = RagConfig(
            embedding_model="fake-dense",
            sparse_embedding_model="fake-sparse",
        )

        results = retrieve(
            "  deployment guide ",
            top_k=4,
            config=config,
            dense_embedder=dense,
            sparse_embedder=sparse,
            store=store,
        )

        self.assertEqual(dense.queries, ["deployment guide"])
        self.assertEqual(sparse.queries, ["deployment guide"])
        self.assertEqual(store.dimension, 3)
        self.assertTrue(store.model_validation_called)
        self.assertEqual(store.search_arguments["limit"], 4)
        self.assertEqual(store.search_arguments["prefetch_limit"], 20)
        self.assertIsNone(store.search_arguments["query_filter"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-id")
        self.assertEqual(results[0].text, "Retrieved text")
        self.assertEqual(results[0].score, 0.75)
        self.assertEqual(results[0].metadata, {"heading_path": ["Guide"]})
        trace_inputs = [
            call.kwargs.get("input")
            for call in self.langfuse_client.start_as_current_observation.call_args_list
        ]
        self.assertTrue(all("deployment guide" not in str(value) for value in trace_inputs))

    def test_metadata_filters_apply_document_and_file_type_conditions(self) -> None:
        config = RagConfig(
            embedding_model="fake-dense",
            sparse_embedding_model="fake-sparse",
        )
        store = FakeSearchStore()

        retrieve(
            "search",
            config=config,
            dense_embedder=FakeDenseQueryModel(),
            sparse_embedder=FakeSparseQueryModel(),
            store=store,
            filters=RetrievalFilters(
                document_ids=("doc-a", "doc-b"),
                file_types=("text/markdown",),
            ),
        )

        query_filter = store.search_arguments["query_filter"]
        self.assertIsInstance(query_filter, models.Filter)
        self.assertEqual(len(query_filter.must), 2)
        self.assertEqual(query_filter.must[0].key, "document_id")
        self.assertEqual(query_filter.must[0].match.any, ["doc-a", "doc-b"])
        self.assertEqual(query_filter.must[1].key, "file_type")
        self.assertEqual(query_filter.must[1].match.any, ["text/markdown"])

    def test_rejects_invalid_query_and_result_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            retrieve("  ")
        with self.assertRaisesRegex(ValueError, "top_k"):
            retrieve("query", top_k=0)

    def test_rejects_invalid_dense_embedding_before_search(self) -> None:
        store = FakeSearchStore()
        config = RagConfig(
            embedding_model="fake-dense",
            sparse_embedding_model="fake-sparse",
        )

        with self.assertRaisesRegex(ValueError, "invalid vector"):
            retrieve(
                "query",
                config=config,
                dense_embedder=FakeDenseQueryModel([0.1, float("nan"), 0.3]),
                sparse_embedder=FakeSparseQueryModel(),
                store=store,
            )

        self.assertEqual(store.search_arguments, {})


if __name__ == "__main__":
    unittest.main()
