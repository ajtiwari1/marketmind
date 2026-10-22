"""PDF parser preserving page boundaries."""

from pathlib import Path

import pymupdf as fitz

from ..models import DocumentSection


def parse_pdf(path: Path) -> tuple[DocumentSection, ...]:
    sections: list[DocumentSection] = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                sections.append(
                    DocumentSection(
                        text=text,
                        metadata={"page_number": page_index + 1},
                    )
                )
    if not sections:
        raise ValueError(
            "No text could be extracted from this PDF; scanned PDFs require OCR"
        )
    return tuple(sections)
