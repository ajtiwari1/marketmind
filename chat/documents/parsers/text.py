"""Plain-text and Markdown document parsers."""

import re
from pathlib import Path

from ..models import DocumentSection


def parse_text(path: Path) -> tuple[DocumentSection, ...]:
    text = path.read_text(encoding="utf-8-sig")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return tuple(
        DocumentSection(text=paragraph, metadata={"section_index": index})
        for index, paragraph in enumerate(paragraphs)
    )


def parse_markdown(path: Path) -> tuple[DocumentSection, ...]:
    text = path.read_text(encoding="utf-8-sig")
    sections: list[DocumentSection] = []
    heading_path: list[str] = []
    current: list[str] = []
    in_code_fence = False
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        body = "\n".join(current).strip()
        if body:
            sections.append(
                DocumentSection(
                    text=body,
                    metadata={
                        "section_index": section_index,
                        "heading_path": tuple(heading_path),
                    },
                    kind="code" if body.startswith("```") else "text",
                )
            )
            section_index += 1
        current.clear()

    for line in text.splitlines():
        fence = line.lstrip().startswith(("```", "~~~"))
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading and not in_code_fence:
            flush()
            depth = len(heading.group(1))
            heading_path[:] = heading_path[: depth - 1]
            heading_path.append(heading.group(2).strip())
            continue
        if fence and not in_code_fence:
            flush()
        current.append(line)
        if fence:
            in_code_fence = not in_code_fence
            if not in_code_fence:
                flush()
    flush()
    return tuple(sections)
