"""Chunking that preserves document sections and repeats table headers."""

import hashlib

from ..documents.models import Chunk, Document, DocumentSection
from .base import Tokenizer


def _tokenize(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _split_tokens(
    tokenizer: Tokenizer,
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    *,
    table: bool,
) -> list[str]:
    token_ids = _tokenize(tokenizer, text)
    if len(token_ids) <= max_tokens:
        return [text]

    if table:
        lines = text.splitlines()
        if len(lines) > 1:
            header = lines[0]
            header_tokens = _tokenize(tokenizer, header)
            if len(header_tokens) >= max_tokens:
                return _split_tokens(
                    tokenizer,
                    text,
                    max_tokens,
                    overlap_tokens,
                    table=False,
                )
            row_budget = max_tokens - len(header_tokens)
            groups: list[str] = []
            current_rows: list[str] = []
            for row in lines[1:]:
                row_tokens = _tokenize(tokenizer, row)
                if len(row_tokens) > row_budget:
                    if current_rows:
                        groups.append("\n".join([header, *current_rows]))
                        current_rows.clear()
                    row_chunks = _split_tokens(
                        tokenizer,
                        row,
                        row_budget,
                        min(overlap_tokens, max(0, row_budget - 1)),
                        table=False,
                    )
                    groups.extend(f"{header}\n{part}" for part in row_chunks)
                    continue
                candidate = "\n".join([header, *current_rows, row])
                if len(_tokenize(tokenizer, candidate)) > max_tokens and current_rows:
                    groups.append("\n".join([header, *current_rows]))
                    current_rows = [row]
                else:
                    current_rows.append(row)
            if current_rows:
                groups.append("\n".join([header, *current_rows]))
            return groups

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        part = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if part:
            chunks.append(part)
        if end == len(token_ids):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks


def _section_chunks(
    section: DocumentSection,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    metadata = section.metadata
    headings = metadata.get("heading_path", ())
    heading_prefix = " > ".join(headings) if headings else ""
    heading_text = f"{heading_prefix}\n" if heading_prefix else ""
    body = section.text.strip()

    if section.kind == "table":
        heading_token_count = len(_tokenize(tokenizer, heading_text))
        table_budget = max_tokens - heading_token_count
        if table_budget < 1:
            heading_text = ""
            table_budget = max_tokens
        table_chunks = _split_tokens(
            tokenizer,
            body,
            table_budget,
            overlap_tokens,
            table=True,
        )
        return [f"{heading_text}{part}".strip() for part in table_chunks]

    heading_token_count = len(_tokenize(tokenizer, heading_text))
    body_budget = max_tokens - heading_token_count
    if body_budget < 1:
        heading_text = ""
        body_budget = max_tokens
    group_budget = (
        body_budget - overlap_tokens
        if overlap_tokens and body_budget > overlap_tokens
        else body_budget
    )
    paragraphs = body.split("\n\n")
    groups: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(_tokenize(tokenizer, candidate)) <= group_budget:
            current = candidate
            continue
        if current:
            groups.append(current)
        if len(_tokenize(tokenizer, paragraph)) <= group_budget:
            current = paragraph
        else:
            groups.extend(
                _split_tokens(
                    tokenizer,
                    paragraph,
                    group_budget,
                    0,
                    table=False,
                )
            )
            current = ""
    if current:
        groups.append(current)
    if not groups:
        groups = [body]
    results: list[str] = []
    previous_tokens: list[int] = []
    for group in groups:
        group_tokens = _tokenize(tokenizer, group)
        available_overlap = max(0, body_budget - len(group_tokens))
        overlap = previous_tokens[-min(overlap_tokens, available_overlap):]
        overlap_text = tokenizer.decode(overlap, skip_special_tokens=True).strip()
        text = "\n".join(
            part for part in (heading_text.rstrip(), overlap_text, group) if part
        )
        while len(_tokenize(tokenizer, text)) > max_tokens and overlap:
            overlap = overlap[1:]
            overlap_text = tokenizer.decode(overlap, skip_special_tokens=True).strip()
            text = "\n".join(
                part for part in (heading_text.rstrip(), overlap_text, group) if part
            )
        results.append(text)
        previous_tokens = group_tokens
    return results


def chunk_document(
    document: Document,
    tokenizer: Tokenizer,
    max_tokens: int = 500,
    overlap_tokens: int = 75,
) -> tuple[Chunk, ...]:
    """Create model-tokenizer-aligned chunks and retain source metadata."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be non-negative and less than max_tokens")

    chunks: list[Chunk] = []
    for section_index, section in enumerate(document.sections):
        for text in _section_chunks(section, tokenizer, max_tokens, overlap_tokens):
            text = text.strip()
            if not text:
                continue
            chunk_index = len(chunks)
            chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{document.document_id}:{chunk_index}:{chunk_hash}".encode("utf-8")
            ).hexdigest()
            metadata = dict(section.metadata)
            metadata.update({"section_index": section.metadata.get("section_index", section_index)})
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    content_hash=document.content_hash,
                    chunk_index=chunk_index,
                    text=text,
                    metadata=metadata,
                )
            )
    return tuple(chunks)
