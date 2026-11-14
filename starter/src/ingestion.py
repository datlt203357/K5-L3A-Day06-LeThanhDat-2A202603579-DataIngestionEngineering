"""Read and normalize PDF pages while preserving source traceability."""
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.splitlines()
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _section_for_page(
    page_number: int,
    sections: list[dict[str, Any]],
) -> str | None:
    matches = [
        section
        for section in sections
        if section["pdf_page_range"][0]
        <= page_number
        <= section["pdf_page_range"][1]
    ]

    if len(matches) > 1:
        raise ValueError(f"PDF page {page_number} belongs to multiple sections")

    return matches[0]["section_id"] if matches else None


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    sections: list[dict[str, Any]] | None = None,
    excluded_pages: set[int] | None = None,
    text_corrections: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract pages, using 1-based physical PDF page numbers."""
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted: {path}")

    sections = sections or []
    excluded_pages = excluded_pages or set()
    corrections_by_page: dict[int, list[dict[str, Any]]] = {}
    for correction in text_corrections or []:
        corrections_by_page.setdefault(int(correction["pdf_page"]), []).append(correction)

    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        if page_number in excluded_pages:
            continue

        text = normalize_text(pdf_page.extract_text() or "")
        metadata: dict[str, Any] = {}

        applied_corrections = []
        for correction in corrections_by_page.get(page_number, []):
            old_text = correction["find"]
            new_text = correction["replace"]
            if old_text in text:
                text = text.replace(old_text, new_text)
                applied_corrections.append(correction.get("reason", "Manual text correction"))
            elif new_text not in text:
                logger.warning(
                    "Could not apply documented text correction on PDF page %s in %s",
                    page_number,
                    path,
                )

        if applied_corrections:
            metadata["manual_corrections"] = applied_corrections
            logger.info(
                "Applied %s documented text correction(s) on PDF page %s",
                len(applied_corrections),
                page_number,
            )

        section_id = _section_for_page(page_number, sections)
        if sections and section_id is None:
            raise ValueError(f"No section mapping for included PDF page {page_number}")
        if section_id is not None:
            metadata["section"] = section_id

        if not text:
            metadata["extraction_status"] = "empty"
            logger.warning("No extractable text on PDF page %s in %s", page_number, path)
        elif "\ufffd" in text:
            logger.warning("Possible text encoding issue on PDF page %s in %s", page_number, path)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )

    return pages
