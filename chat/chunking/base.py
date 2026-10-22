"""Tokenizer interface used for model-aligned chunk size measurement."""

from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, **kwargs: object) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> str: ...
