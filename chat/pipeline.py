"""Document-to-Qdrant ingestion pipeline and CLI."""

import argparse
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol, Sequence

from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse

from libs.langfuse import flush_langfuse, get_langfuse_client

from .chunking.base import Tokenizer
from .chunking.text import chunk_document
from .config import RagConfig
from .documents.loader import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document
from .documents.models import Chunk, Document
from .embeddings.sparse import SparseTextEmbedder, SparseVector
from .vector_store.qdrant import QdrantChunkStore

if TYPE_CHECKING:
    from .embeddings.sentence_transformer import SentenceTransformerEmbedder


class Embedder(Protocol):
    tokenizer: Tokenizer
    dimension: int
    model_name: str
    max_input_tokens: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


class ChunkStore(Protocol):
    def ensure_collection(
        self,
        vector_size: int,
        *,
        create_if_missing: bool = False,
    ) -> None: ...

    def upsert_document(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> int: ...


class IngestionError(RuntimeError):
    """Raised when a document cannot be parsed, embedded, or indexed."""


@dataclass
class IngestionSummary:
    accepted_files: int = 0
    skipped_files: int = 0
    sections_extracted: int = 0
    chunks_upserted: int = 0
    errors: dict[str, str] = field(default_factory=dict)


def _load_embedder(config: RagConfig) -> "SentenceTransformerEmbedder":
    from .embeddings.sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size,
        device=config.embedding_device,
    )


def _load_sparse_embedder(config: RagConfig) -> SparseTextEmbedder:
    return SparseTextEmbedder(
        model_name=config.sparse_embedding_model,
        batch_size=config.embedding_batch_size,
    )


def _ingest_document(
    document: Document,
    settings: RagConfig,
    *,
    dry_run: bool,
    create_collection: bool,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder | None,
    store: ChunkStore | None,
) -> IngestionSummary:
    client = get_langfuse_client()
    with client.start_as_current_observation(
        as_type="span",
        name="rag.ingest_document",
        input={
            "file_type": document.file_type,
            "section_count": len(document.sections),
            "dry_run": dry_run,
        },
    ) as observation:
        result = _ingest_document_operation(
            document,
            settings,
            dry_run=dry_run,
            create_collection=create_collection,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            store=store,
        )
        observation.update(
            output={
                "accepted_files": result.accepted_files,
                "sections_extracted": result.sections_extracted,
                "chunks_upserted": result.chunks_upserted,
            }
        )
        return result


def _ingest_document_operation(
    document: Document,
    settings: RagConfig,
    *,
    dry_run: bool,
    create_collection: bool,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder | None,
    store: ChunkStore | None,
) -> IngestionSummary:
    if settings.embedding_model != embedder.model_name:
        raise IngestionError(
            f"Configured embedding model '{settings.embedding_model}' does not "
            f"match the loaded model '{embedder.model_name}'"
        )
    chunks = chunk_document(
        document,
        embedder.tokenizer,
        max_tokens=min(settings.chunk_size_tokens, embedder.max_input_tokens),
        overlap_tokens=min(
            settings.chunk_overlap_tokens,
            max(0, min(settings.chunk_size_tokens, embedder.max_input_tokens) - 1),
        ),
    )
    summary = IngestionSummary(
        accepted_files=1,
        sections_extracted=len(document.sections),
        chunks_upserted=0 if dry_run else len(chunks),
    )
    if dry_run:
        return summary

    active_sparse_embedder = sparse_embedder or _load_sparse_embedder(settings)
    if active_sparse_embedder.model_name != settings.sparse_embedding_model:
        raise IngestionError(
            f"Configured sparse embedding model '{settings.sparse_embedding_model}' "
            f"does not match loaded model '{active_sparse_embedder.model_name}'"
        )

    vector_store = store or QdrantChunkStore(settings)
    texts = [chunk.text for chunk in chunks]
    with get_langfuse_client().start_as_current_observation(
        as_type="embedding",
        name="rag.embedding.dense",
        model=embedder.model_name,
        model_parameters={"batch_size": settings.embedding_batch_size},
        input={"chunk_count": len(chunks)},
    ) as observation:
        vectors = embedder.embed_documents(texts)
        observation.update(
            output={
                "vector_count": len(vectors),
                "dimension": embedder.dimension,
            }
        )
    with get_langfuse_client().start_as_current_observation(
        as_type="embedding",
        name="rag.embedding.sparse",
        model=active_sparse_embedder.model_name,
        model_parameters={"batch_size": settings.embedding_batch_size},
        input={"chunk_count": len(chunks)},
    ) as observation:
        sparse_vectors = active_sparse_embedder.embed_documents(texts)
        observation.update(
            output={
                "vector_count": len(sparse_vectors),
                "nonzero_terms": sum(len(vector.indices) for vector in sparse_vectors),
            }
        )
    if len(vectors) != len(chunks) or any(
        len(vector) != embedder.dimension
        or any(not isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise IngestionError(
            f"Embedding model '{embedder.model_name}' returned vectors with an "
            f"unexpected count, dimension, or non-finite values; expected "
            f"{len(chunks)} finite vectors of dimension {embedder.dimension}"
        )
    if len(sparse_vectors) != len(chunks) or any(
        len(vector.indices) != len(vector.values)
        or not vector.indices
        or any(not isfinite(value) for value in vector.values)
        for vector in sparse_vectors
    ):
        raise IngestionError(
            f"Sparse embedding model '{active_sparse_embedder.model_name}' returned "
            "vectors with an unexpected count or invalid indices/values"
        )
    with get_langfuse_client().start_as_current_observation(
        as_type="span",
        name="rag.qdrant.upsert",
        input={
            "collection": settings.collection_name,
            "chunk_count": len(chunks),
        },
    ) as observation:
        vector_store.ensure_collection(
            embedder.dimension,
            create_if_missing=create_collection,
        )
        summary.chunks_upserted = vector_store.upsert_document(
            chunks,
            vectors,
            sparse_vectors,
        )
        observation.update(output={"upserted_points": summary.chunks_upserted})
    return summary


def ingest_file(
    path: str | Path,
    *,
    config: RagConfig | None = None,
    dry_run: bool = False,
    create_collection: bool = False,
    embedder: Embedder | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    store: ChunkStore | None = None,
) -> IngestionSummary:
    """Parse, chunk, embed, and optionally upsert one document."""
    settings = config or RagConfig.from_env()
    settings.validate()
    try:
        document = load_document(path, max_file_size_bytes=settings.max_file_size_bytes)
    except DocumentLoadError as error:
        raise IngestionError(str(error)) from error

    model = embedder or _load_embedder(settings)
    return _ingest_document(
        document,
        settings,
        dry_run=dry_run,
        create_collection=create_collection,
        embedder=model,
        sparse_embedder=sparse_embedder,
        store=store,
    )


def ingest_directory(
    path: str | Path,
    *,
    config: RagConfig | None = None,
    recursive: bool = True,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    dry_run: bool = False,
    create_collection: bool = False,
    embedder: Embedder | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    store: ChunkStore | None = None,
) -> IngestionSummary:
    """Ingest supported files in a directory and report individual failures."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise IngestionError(f"Not a directory: {root}")

    settings = config or RagConfig.from_env()
    settings.validate()
    candidates = root.rglob("*") if recursive else root.iterdir()
    summary = IngestionSummary()
    active_embedder = embedder
    active_sparse_embedder = sparse_embedder
    for file_path in sorted(candidates):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            summary.skipped_files += 1
            continue
        relative_path = file_path.relative_to(root).as_posix()
        relative = PurePosixPath(relative_path)
        if include and not any(relative.match(pattern) for pattern in include):
            summary.skipped_files += 1
            continue
        if any(relative.match(pattern) for pattern in exclude):
            summary.skipped_files += 1
            continue
        try:
            document = load_document(
                file_path,
                max_file_size_bytes=settings.max_file_size_bytes,
            )
        except DocumentLoadError as error:
            summary.errors[relative_path] = str(error)
            continue
        if active_embedder is None:
            active_embedder = _load_embedder(settings)
        if not dry_run and active_sparse_embedder is None:
            active_sparse_embedder = _load_sparse_embedder(settings)
        try:
            result = _ingest_document(
                document,
                settings,
                dry_run=dry_run,
                create_collection=create_collection,
                embedder=active_embedder,
                sparse_embedder=active_sparse_embedder,
                store=store,
            )
        except (
            ApiException,
            IngestionError,
            ResponseHandlingException,
            RuntimeError,
            UnexpectedResponse,
            ValueError,
            OSError,
        ) as error:
            summary.errors[relative_path] = str(error)
            continue
        summary.accepted_files += result.accepted_files
        summary.sections_extracted += result.sections_extracted
        summary.chunks_upserted += result.chunks_upserted
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into local Qdrant")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk without Qdrant writes")
    parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Explicitly allow creation of the configured collection if missing",
    )
    parser.add_argument("--non-recursive", action="store_true", help="Do not recurse into subdirectories")
    parser.add_argument("--include", action="append", default=[], help="Include glob; may be repeated")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude glob; may be repeated")
    arguments = parser.parse_args()

    try:
        if arguments.path.is_file():
            summary = ingest_file(
                arguments.path,
                dry_run=arguments.dry_run,
                create_collection=arguments.create_collection,
            )
        else:
            summary = ingest_directory(
                arguments.path,
                recursive=not arguments.non_recursive,
                include=tuple(arguments.include),
                exclude=tuple(arguments.exclude),
                dry_run=arguments.dry_run,
                create_collection=arguments.create_collection,
            )
    except (
        ApiException,
        IngestionError,
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
        f"Accepted: {summary.accepted_files}; skipped: {summary.skipped_files}; "
        f"sections: {summary.sections_extracted}; chunks upserted: "
        f"{summary.chunks_upserted}"
    )
    for source, message in summary.errors.items():
        print(f"ERROR {source}: {message}")
    if summary.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
