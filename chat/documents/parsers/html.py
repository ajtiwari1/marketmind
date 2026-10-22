"""HTML parser that removes non-content elements and preserves headings."""

from pathlib import Path

from bs4 import BeautifulSoup

from ..models import DocumentSection


def parse_html(path: Path) -> tuple[DocumentSection, ...]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    content = soup.find("main") or soup.body or soup
    sections: list[DocumentSection] = []
    heading_path: list[str] = []
    for element in content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "table"],
        recursive=True,
    ):
        if element.name == "table":
            text = "\n".join(
                " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"]))
                for row in element.find_all("tr")
            )
        else:
            text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            depth = int(element.name[1])
            heading_path[:] = heading_path[: depth - 1]
            heading_path.append(text)
            continue
        sections.append(
            DocumentSection(
                text=text,
                kind="table" if element.name == "table" else (
                    "code" if element.name == "pre" else "text"
                ),
                metadata={
                    "heading_path": tuple(heading_path),
                    "title": title,
                    "section_index": len(sections),
                },
            )
        )
    return tuple(sections)
