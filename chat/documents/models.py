"""Typed document structures shared by parsers, chunkers, and indexers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentSection:
    text: str
    kind: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    file_name: str
    file_type: str
    content_hash: str
    sections: tuple[DocumentSection, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    file_name: str
    file_type: str
    content_hash: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
