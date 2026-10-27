
from pathlib import Path

import fitz


def extract_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []

    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "source": path.name,
                        "page": page_index + 1,
                        "text": text,
                    }
                )

    return pages
