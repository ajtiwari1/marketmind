"""Hybrid retrieval over dense and sparse Qdrant vectors."""

import argparse
from dataclasses import dataclass, field
import json
from math import isfinite
from typing import Any, Protocol, Sequence

from qdrant_client import models
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse

from libs.langfuse import flush_langfuse, get_langfuse_client

from .config import RagConfig
from .embeddings.sparse import SparseTextEmbedder, SparseVector
from .vector_store.qdrant import QdrantChunkStore


@dataclass(frozen=True)
class RetrievalFilters:
    document_ids: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()

    def to_qdrant_filter(self) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if self.document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(self.document_ids)),
                )
            )
        if self.file_types:
            conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchAny(any=list(self.file_types)),
                )
            )
        return models.Filter(must=conditions) if conditions else None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    file_name: str
    file_type: str
    chunk_index: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DenseQueryEmbedder:
    """Lazy adapter for the configured Sentence Transformers query model."""

    def __init__(self, config: RagConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = config.embedding_model
        self.model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
        )
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(
                f"Embedding model has no known output dimension: {config.embedding_model}"
            )
        self.dimension = dimension
        self.batch_size = config.embedding_batch_size

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


class HybridSearchStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...

    def validate_embedding_models(self) -> None: ...

    def search_hybrid(
        self,
        dense_vector: Sequence[float],
        sparse_vector: SparseVector,
        *,
        limit: int,
        prefetch_limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]: ...


class DenseQueryModel(Protocol):
    model_name: str
    dimension: int

    def embed_query(self, text: str) -> list[float]: ...


class SparseQueryModel(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


def _to_result(point: models.ScoredPoint) -> RetrievedChunk:
    payload = point.payload
    if payload is None:
        raise ValueError(f"Search result {point.id} has no payload")
    required = ("chunk_id", "document_id", "text", "source", "file_name", "file_type")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Search result {point.id} is missing required payload fields: "
            f"{', '.join(missing)}"
        )
    return RetrievedChunk(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        text=str(payload["text"]),
        source=str(payload["source"]),
        file_name=str(payload["file_name"]),
        file_type=str(payload["file_type"]),
        chunk_index=int(payload.get("chunk_index", 0)),
        score=float(point.score),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {*required, "chunk_index"}
        },
    )


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    filters: RetrievalFilters | None = None,
    config: RagConfig | None = None,
    dense_embedder: DenseQueryModel | None = None,
    sparse_embedder: SparseQueryModel | None = None,
    store: HybridSearchStore | None = None,
) -> list[RetrievedChunk]:
    """Embed a query, run dense+sparse Qdrant search, and fuse with RRF."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    settings = config or RagConfig.from_env()
    settings.validate()
    dense_model = dense_embedder or DenseQueryEmbedder(settings)
    sparse_model = sparse_embedder or SparseTextEmbedder(
        settings.sparse_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    if dense_model.model_name != settings.embedding_model:
        raise ValueError(
            f"Configured embedding model '{settings.embedding_model}' does not "
            f"match the loaded model '{dense_model.model_name}'"
        )
    if sparse_model.model_name != settings.sparse_embedding_model:
        raise ValueError(
            f"Configured sparse embedding model '{settings.sparse_embedding_model}' "
            f"does not match the loaded model '{sparse_model.model_name}'"
        )

    vector_store = store or QdrantChunkStore(settings)
    client = get_langfuse_client()
    with client.start_as_current_observation(
        as_type="span",
        name="rag.retrieve",
        input={
            "query_length": len(query.strip()),
            "top_k": top_k,
            "collection": settings.collection_name,
        },
    ) as observation:
        vector_store.ensure_collection(dense_model.dimension)
        vector_store.validate_embedding_models()
        with client.start_as_current_observation(
            as_type="embedding",
            name="rag.query_embedding.dense",
            model=dense_model.model_name,
            input={"query_length": len(query.strip())},
        ) as dense_observation:
            dense_vector = dense_model.embed_query(query.strip())
            dense_observation.update(
                output={"dimension": len(dense_vector)}
            )
        with client.start_as_current_observation(
            as_type="embedding",
            name="rag.query_embedding.sparse",
            model=sparse_model.model_name,
            input={"query_length": len(query.strip())},
        ) as sparse_observation:
            sparse_vectors = sparse_model.embed_documents([query.strip()])
            sparse_observation.update(output={"vector_count": len(sparse_vectors)})

        if (
            len(dense_vector) != dense_model.dimension
            or any(not isfinite(value) for value in dense_vector)
        ):
            raise ValueError(
                f"Dense query model '{dense_model.model_name}' returned an invalid "
                f"vector; expected {dense_model.dimension} finite values"
            )
        if (
            len(sparse_vectors) != 1
            or not sparse_vectors[0].indices
            or len(sparse_vectors[0].indices) != len(sparse_vectors[0].values)
            or any(not isfinite(value) for value in sparse_vectors[0].values)
        ):
            raise ValueError(
                f"Sparse query model '{sparse_model.model_name}' returned an invalid vector"
            )

        prefetch_limit = max(top_k * 3, 20)
        with client.start_as_current_observation(
            as_type="retriever",
            name="rag.qdrant.hybrid_search",
            input={"prefetch_limit": prefetch_limit},
        ) as search_observation:
            points = vector_store.search_hybrid(
                dense_vector,
                sparse_vectors[0],
                limit=top_k,
                prefetch_limit=prefetch_limit,
                query_filter=(filters or RetrievalFilters()).to_qdrant_filter(),
            )
            results = [_to_result(point) for point in points]
            search_observation.update(output={"result_count": len(results)})
        observation.update(output={"result_count": len(results)})
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed documents with hybrid RAG")
    parser.add_argument("query", help="Text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Restrict results to a document ID; may be repeated",
    )
    parser.add_argument(
        "--file-type",
        action="append",
        default=[],
        help="Restrict results to a MIME type; may be repeated",
    )
    arguments = parser.parse_args()
    try:
        results = retrieve(
            arguments.query,
            top_k=arguments.top_k,
            filters=RetrievalFilters(
                document_ids=tuple(arguments.document_id),
                file_types=tuple(arguments.file_type),
            ),
        )
    except (
        ApiException,
        OSError,
        ResponseHandlingException,
        RuntimeError,
        UnexpectedResponse,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    finally:
        flush_langfuse()
    print(
        json.dumps(
            [
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "text": result.text,
                    "source": result.source,
                    "file_name": result.file_name,
                    "file_type": result.file_type,
                    "chunk_index": result.chunk_index,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
