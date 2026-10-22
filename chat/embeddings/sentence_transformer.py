"""Lazy Sentence Transformers model adapter."""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from ..chunking.base import Tokenizer


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer: Tokenizer = self.model.tokenizer
        self.max_input_tokens = int(self.model.max_seq_length) - int(
            self.tokenizer.num_special_tokens_to_add(pair=False)
        )
        if self.max_input_tokens < 1:
            raise ValueError(f"Embedding model has no usable token capacity: {model_name}")
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Embedding model has no known output dimension: {model_name}")
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
