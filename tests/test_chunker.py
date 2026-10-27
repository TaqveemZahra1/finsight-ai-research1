
import pytest

from finsight.ingestion.chunker import chunk_text


def test_chunk_text_splits_long_text():
    pages = [
        {
            "source": "report.pdf",
            "page": 1,
            "text": "one two three four five six seven eight nine ten",
        }
    ]

    chunks = chunk_text(pages, chunk_size=4, overlap=1)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six seven"
    assert chunks[2]["text"] == "seven eight nine ten"


def test_chunk_text_preserves_metadata():
    pages = [
        {
            "source": "report.pdf",
            "page": 3,
            "text": "Revenue increased this year.",
        }
    ]

    chunks = chunk_text(pages, chunk_size=10, overlap=2)

    assert chunks[0]["source"] == "report.pdf"
    assert chunks[0]["page"] == 3
    assert chunks[0]["chunk_index"] == 0


def test_chunk_text_rejects_invalid_overlap():
    pages = [{"source": "report.pdf", "page": 1, "text": "hello world"}]

    with pytest.raises(ValueError):
        chunk_text(pages, chunk_size=5, overlap=5)


def test_chunk_text_rejects_zero_chunk_size():
    with pytest.raises(ValueError):
        chunk_text([], chunk_size=0, overlap=0)


def test_chunk_text_handles_empty_text():
    pages = [{"source": "report.pdf", "page": 1, "text": ""}]

    assert chunk_text(pages) == []
