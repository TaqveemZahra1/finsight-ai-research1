
import fitz 
from src.finsight.ingestion.pdf_loader import extract_pdf


def test_extract_pdf_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # Create a small sample PDF for the test.
    with fitz.open() as document:
        for content in [
            "Revenue increased.",
            "Operating costs decreased.",
        ]:
            page = document.new_page()
            page.insert_text((72, 72), content)

        document.save(str(pdf_path))

    # Extract the PDF content.
    pages = extract_pdf(pdf_path)

    # Verify that both pages and their page numbers are preserved.
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert "Revenue increased." in pages[0]["text"]
    assert "Operating costs decreased." in pages[1]["text"]
