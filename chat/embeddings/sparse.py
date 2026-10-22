"""Sparse text embeddings using FastEmbed's SPLADE++ model."""

from collections.abc import Sequence
from dataclasses import dataclass

from fastembed import SparseTextEmbedding


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseTextEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        embeddings = self.model.embed(list(texts), batch_size=self.batch_size)
        return [
            SparseVector(indices=list(embedding.indices), values=list(embedding.values))
            for embedding in embeddings
        ]
