"""Safe Qdrant collection validation and chunk upserts."""

from collections.abc import Sequence
from uuid import UUID

from qdrant_client import QdrantClient, models

from ..config import RagConfig
from ..documents.models import Chunk
from ..embeddings.sparse import SparseVector


class QdrantChunkStore:
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        config: RagConfig,
        client: QdrantClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

    def ensure_collection(
        self,
        vector_size: int,
        *,
        create_if_missing: bool = False,
    ) -> None:
        name = self.config.collection_name
        if not self.client.collection_exists(name):
            if not create_if_missing:
                raise RuntimeError(
                    f"Qdrant collection '{name}' does not exist. Re-run with "
                    "create_collection=True to create it explicitly."
                )
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    self.DENSE_VECTOR_NAME: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams()
                    )
                },
            )
            return

        collection = self.client.get_collection(name)
        vectors = collection.config.params.vectors
        if not isinstance(vectors, dict):
            raise ValueError(
                f"Collection '{name}' uses an unnamed dense vector. This pipeline "
                "requires named 'dense' and 'sparse' vectors. Set "
                "QDRANT_COLLECTION_NAME to a new collection; the existing "
                "collection will not be modified."
            )
        dense_vector = vectors.get(self.DENSE_VECTOR_NAME)
        if dense_vector is None:
            raise ValueError(
                f"Collection '{name}' has no '{self.DENSE_VECTOR_NAME}' vector."
            )
        if dense_vector.size != vector_size:
            raise ValueError(
                f"Collection '{name}' has dense vector size {dense_vector.size}, "
                f"but model '{self.config.embedding_model}' produces {vector_size}. "
                "Use a separate collection or re-embedding migration."
            )
        if dense_vector.distance != models.Distance.COSINE:
            raise ValueError(
                f"Collection '{name}' dense vector uses {dense_vector.distance}; "
                "expected COSINE."
            )
        sparse_vectors = collection.config.params.sparse_vectors or {}
        if self.SPARSE_VECTOR_NAME not in sparse_vectors:
            raise ValueError(
                f"Collection '{name}' has no '{self.SPARSE_VECTOR_NAME}' sparse "
                "vector. Use a new collection; the existing collection will not "
                "be modified."
            )

    def upsert_document(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> int:
        if len(chunks) != len(vectors) or len(chunks) != len(sparse_vectors):
            raise ValueError("Chunk count and dense/sparse embedding counts do not match")
        if not chunks:
            return 0
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("upsert_document accepts chunks from one document only")

        points: list[models.PointStruct] = []
        for chunk, vector, sparse_vector in zip(chunks, vectors, sparse_vectors):
            if (
                len(sparse_vector.indices) != len(sparse_vector.values)
                or not sparse_vector.indices
            ):
                raise ValueError("Sparse vector indices and values do not match")
            points.append(
                models.PointStruct(
                    id=str(UUID(chunk.chunk_id[:32])),
                    vector={
                        self.DENSE_VECTOR_NAME: list(vector),
                        self.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "file_name": chunk.file_name,
                        "file_type": chunk.file_type,
                        "content_hash": chunk.content_hash,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "embedding_model": self.config.embedding_model,
                        "sparse_embedding_model": self.config.sparse_embedding_model,
                    },
                )
            )

        document_id = chunks[0].document_id
        self.validate_embedding_models()

        existing_ids: list[str | int] = []
        offset = None
        while True:
            existing, offset = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids.extend(point.id for point in existing)
            if offset is None:
                break

        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )

        new_ids = {point.id for point in points}
        stale_ids = [point_id for point_id in existing_ids if point_id not in new_ids]
        if stale_ids:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )
        return len(points)

    def validate_embedding_models(self) -> None:
        points, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=1,
            with_payload=["embedding_model", "sparse_embedding_model"],
            with_vectors=False,
        )
        if not points:
            return
        payload = points[0].payload or {}
        indexed_dense_model = payload.get("embedding_model")
        indexed_sparse_model = payload.get("sparse_embedding_model")
        if indexed_dense_model != self.config.embedding_model:
            raise ValueError(
                f"Collection '{self.config.collection_name}' contains dense "
                f"vectors from model '{indexed_dense_model}', but the configured "
                f"model is '{self.config.embedding_model}'. Use a new collection "
                "or re-embed all documents."
            )
        if indexed_sparse_model != self.config.sparse_embedding_model:
            raise ValueError(
                f"Collection '{self.config.collection_name}' contains sparse "
                f"vectors from model '{indexed_sparse_model}', but the configured "
                f"model is '{self.config.sparse_embedding_model}'. Use a new "
                "collection or re-embed all documents."
            )

    def search_hybrid(
        self,
        dense_vector: Sequence[float],
        sparse_vector: SparseVector,
        *,
        limit: int,
        prefetch_limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if prefetch_limit < limit:
            raise ValueError("prefetch_limit must be at least limit")
        if len(sparse_vector.indices) != len(sparse_vector.values) or not sparse_vector.indices:
            raise ValueError("Sparse vector indices and values do not match")
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=list(dense_vector),
                    using=self.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points
