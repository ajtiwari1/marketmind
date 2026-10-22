import unittest
from unittest.mock import MagicMock

from qdrant_client import models

from chat.config import RagConfig
from chat.embeddings.sparse import SparseVector
from chat.vector_store.qdrant import QdrantChunkStore


class QdrantCollectionSafetyTests(unittest.TestCase):
    def test_missing_collection_is_not_created_implicitly(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = False
        store = QdrantChunkStore(RagConfig(), client=client)

        with self.assertRaisesRegex(RuntimeError, "create_collection=True"):
            store.ensure_collection(384)

        client.create_collection.assert_not_called()

    def test_collection_creation_requires_explicit_permission(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = False
        store = QdrantChunkStore(RagConfig(), client=client)

        store.ensure_collection(384, create_if_missing=True)

        call = client.create_collection.call_args.kwargs
        self.assertEqual(call["collection_name"], "marketmind_documents")
        self.assertEqual(
            call["vectors_config"]["dense"],
            models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        self.assertIn("sparse", call["sparse_vectors_config"])

    def test_existing_collection_dimension_must_match(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors = {
            "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
        }
        client.get_collection.return_value.config.params.sparse_vectors = {
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams())
        }
        store = QdrantChunkStore(RagConfig(), client=client)

        with self.assertRaisesRegex(ValueError, "has dense vector size 768"):
            store.ensure_collection(384)

        client.create_collection.assert_not_called()

    def test_hybrid_search_uses_dense_sparse_prefetch_and_rrf(self) -> None:
        client = MagicMock()
        client.query_points.return_value.points = []
        store = QdrantChunkStore(RagConfig(), client=client)

        results = store.search_hybrid(
            [0.1, 0.2, 0.3],
            SparseVector(indices=[2, 5], values=[0.8, 0.4]),
            limit=5,
            prefetch_limit=20,
        )

        self.assertEqual(results, [])
        call = client.query_points.call_args.kwargs
        self.assertEqual(call["collection_name"], "marketmind_documents")
        self.assertEqual(call["query"].fusion, models.Fusion.RRF)
        self.assertEqual(len(call["prefetch"]), 2)
        self.assertEqual(call["prefetch"][0].using, "dense")
        self.assertEqual(call["prefetch"][0].query, [0.1, 0.2, 0.3])
        self.assertEqual(call["prefetch"][1].using, "sparse")
        self.assertEqual(call["prefetch"][1].query.indices, [2, 5])
        self.assertEqual(call["limit"], 5)


if __name__ == "__main__":
    unittest.main()
