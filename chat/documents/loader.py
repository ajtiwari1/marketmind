"""File-type dispatch and document identity generation."""

import hashlib
import mimetypes
from pathlib import Path
from uuid import UUID, uuid5

from .models import Document, DocumentSection
from .parsers.html import parse_html
from .parsers.office import parse_docx, parse_pptx
from .parsers.pdf import parse_pdf
from .parsers.structured import parse_csv, parse_json, parse_xlsx
from .parsers.text import parse_markdown, parse_text


class DocumentLoadError(ValueError):
    """Raised when a document is unsupported, invalid, or cannot be extracted."""


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".rst",
    ".log",
    ".json",
}


def load_document(path: str | Path, max_file_size_bytes: int = 50 * 1024 * 1024) -> Document:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise DocumentLoadError(f"Document is not a file: {file_path}")
    if file_path.stat().st_size > max_file_size_bytes:
        raise DocumentLoadError(
            f"Document exceeds the {max_file_size_bytes}-byte size limit: {file_path}"
        )

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported document type '{extension or '[no extension]'}': {file_path}"
        )

    parsers = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".pptx": parse_pptx,
        ".xlsx": parse_xlsx,
        ".csv": parse_csv,
        ".html": parse_html,
        ".htm": parse_html,
        ".md": parse_markdown,
        ".txt": parse_text,
        ".rst": parse_text,
        ".log": parse_text,
        ".json": parse_json,
    }
    try:
        sections = parsers[extension](file_path)
    except Exception as error:
        raise DocumentLoadError(f"Failed to read {file_path}: {error}") from error
    sections = tuple(
        DocumentSection(
            text=section.text.replace("\r\n", "\n").replace("\r", "\n").strip(),
            kind=section.kind,
            metadata=section.metadata,
        )
        for section in sections
        if section.text.strip()
    )
    if not sections:
        raise DocumentLoadError(f"No text could be extracted from {file_path}")

    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    document_id = str(uuid5(UUID("59b20ac1-21ef-4c00-a95d-f2871b9e3db9"), str(file_path.resolve())))
    return Document(
        document_id=document_id,
        source=file_path.resolve().as_uri(),
        file_name=file_path.name,
        file_type=mimetypes.guess_type(file_path.name)[0] or extension.lstrip("."),
        content_hash=content_hash,
        sections=sections,
    )
