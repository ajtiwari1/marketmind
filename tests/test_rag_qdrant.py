import os
import unittest
from hashlib import sha256
from uuid import uuid4

from chat.config import RagConfig
from chat.documents.models import Chunk
from chat.embeddings.sparse import SparseVector
from chat.vector_store.qdrant import QdrantChunkStore


@unittest.skipUnless(
    os.getenv("RUN_QDRANT_INTEGRATION") == "1",
    "Set RUN_QDRANT_INTEGRATION=1 to test against local Qdrant",
)
class LocalQdrantIntegrationTests(unittest.TestCase):
    def test_create_upsert_replace_and_hybrid_search(self) -> None:
        from qdrant_client import QdrantClient

        collection_name = f"marketmind_test_{uuid4().hex}"
        config = RagConfig(collection_name=collection_name)
        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        store = QdrantChunkStore(config, client=client)
        document_id = str(uuid4())

        def make_chunk(index: int, text: str) -> Chunk:
            content_hash = sha256(text.encode()).hexdigest()
            chunk_id = sha256(f"{document_id}:{index}:{content_hash}".encode()).hexdigest()
            return Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source="file:///test.txt",
                file_name="test.txt",
                file_type="text/plain",
                content_hash=content_hash,
                chunk_index=index,
                text=text,
            )

        try:
            store.ensure_collection(3, create_if_missing=True)
            old_chunks = [make_chunk(0, "old chunk"), make_chunk(1, "stale chunk")]
            old_sparse = [
                SparseVector(indices=[1, 2], values=[0.5, 0.25]),
                SparseVector(indices=[1, 3], values=[0.4, 0.3]),
            ]
            self.assertEqual(
                store.upsert_document(old_chunks, [[0.1, 0.2, 0.3]] * 2, old_sparse),
                2,
            )
            new_chunks = [make_chunk(0, "replacement chunk")]
            new_sparse = [SparseVector(indices=[4, 5], values=[0.7, 0.2])]
            self.assertEqual(
                store.upsert_document(new_chunks, [[0.2, 0.3, 0.4]], new_sparse),
                1,
            )
            points, _ = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0].payload["text"], "replacement chunk")
            self.assertIn("dense", points[0].vector)
            self.assertIn("sparse", points[0].vector)
            self.assertEqual(points[0].vector["sparse"].indices, [4, 5])
            store.validate_embedding_models()
            results = store.search_hybrid(
                [0.2, 0.3, 0.4],
                SparseVector(indices=[4, 5], values=[0.7, 0.2]),
                limit=1,
                prefetch_limit=5,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].payload["text"], "replacement chunk")
        finally:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.close()


if __name__ == "__main__":
    unittest.main()
