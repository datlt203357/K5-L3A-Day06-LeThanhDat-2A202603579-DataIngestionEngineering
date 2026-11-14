"""Fixed-size and paragraph-aware chunking with source metadata preserved."""
import re

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    strategy: str,
    index: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}:p{page.page}:{strategy}:{index:04d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=dict(page.metadata),
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into fixed character windows with character overlap."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = page.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    stride = size - overlap
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(_make_chunk(page, "fixed", len(chunks), piece))
        if end == len(text):
            break
        start += stride

    return chunks


def _is_heading(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    if not letters or len(line) > 120:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return len(letters) >= 4 and uppercase_ratio >= 0.8


def _is_list_start(line: str) -> bool:
    return bool(re.match(r"^(?:[-•+]\s+|\d+[.)]\s+|[a-zđ]\)\s+)", line, re.I))


def _ends_sentence(line: str) -> bool:
    return bool(re.search(r"[.!?…][\"'”’)\]]*$", line))


def _logical_paragraphs(text: str) -> list[str]:
    """Group extracted PDF lines into approximate paragraphs and headings."""
    paragraphs: list[str] = []

    for block in re.split(r"\n\s*\n+", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        current: list[str] = []

        def flush() -> None:
            if current:
                paragraphs.append(" ".join(current).strip())
                current.clear()

        for line in lines:
            if _is_heading(line):
                flush()
                paragraphs.append(line)
            elif _is_list_start(line):
                flush()
                current.append(line)
            else:
                if current and _ends_sentence(current[-1]):
                    flush()
                current.append(line)

        flush()

    return paragraphs


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph near word boundaries."""
    pieces: list[str] = []
    remaining = text.strip()

    while len(remaining) > max_size:
        cut = remaining.rfind(" ", 0, max_size + 1)
        if cut <= 0:  # A single long token: hard-split to honor the size cap.
            cut = max_size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        pieces.append(remaining)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack paragraph-like units without exceeding max_size characters."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")
    if not page.text.strip():
        return []

    paragraphs = _logical_paragraphs(page.text)
    units = [
        piece
        for paragraph in paragraphs
        for piece in _split_long_text(paragraph, max_size)
    ]

    chunks: list[Chunk] = []
    current: list[str] = []
    current_length = 0

    for unit in units:
        extra_length = len(unit) + (2 if current else 0)
        if current and current_length + extra_length > max_size:
            chunks.append(
                _make_chunk(page, "paragraph", len(chunks), "\n\n".join(current))
            )
            current = []
            current_length = 0
            extra_length = len(unit)

        current.append(unit)
        current_length += extra_length

    if current:
        chunks.append(
            _make_chunk(page, "paragraph", len(chunks), "\n\n".join(current))
        )

    return chunks
