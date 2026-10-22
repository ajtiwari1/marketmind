"""Environment-backed configuration for document ingestion."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: str | None = None
    embedding_device: str | None = None
    embedding_batch_size: int = 32
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "marketmind_documents"
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75
    max_file_size_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "RagConfig":
        """Load RAG settings from environment variables."""
        config = cls(
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            sparse_embedding_model=os.getenv(
                "RAG_SPARSE_EMBEDDING_MODEL",
                "Qdrant/bm25",
            ),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
            embedding_device=os.getenv("RAG_EMBEDDING_DEVICE") or None,
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv(
                "QDRANT_COLLECTION_NAME",
                "marketmind_documents",
            ),
            chunk_size_tokens=int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "500")),
            chunk_overlap_tokens=int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "75")),
            max_file_size_bytes=int(
                os.getenv("RAG_MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.embedding_model.strip():
            raise ValueError("RAG_EMBEDDING_MODEL must not be empty")
        if not self.sparse_embedding_model.strip():
            raise ValueError("RAG_SPARSE_EMBEDDING_MODEL must not be empty")
        if not self.gemini_model.strip():
            raise ValueError("GEMINI_MODEL must not be empty")
        if not self.qdrant_url.strip():
            raise ValueError("QDRANT_URL must not be empty")
        if not self.collection_name.strip():
            raise ValueError("QDRANT_COLLECTION_NAME must not be empty")
        if self.embedding_batch_size < 1:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE must be at least 1")
        if self.chunk_size_tokens < 1:
            raise ValueError("RAG_CHUNK_SIZE_TOKENS must be at least 1")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError(
                "RAG_CHUNK_OVERLAP_TOKENS must be non-negative and smaller "
                "than RAG_CHUNK_SIZE_TOKENS"
            )
        if self.max_file_size_bytes < 1:
            raise ValueError("RAG_MAX_FILE_SIZE_BYTES must be at least 1")
