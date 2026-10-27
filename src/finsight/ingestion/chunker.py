
def chunk_text(
    pages: list[dict],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """Split extracted PDF pages into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for page_data in pages:
        words = page_data["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                break

            chunks.append(
                {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": len(chunks),
                    "text": " ".join(chunk_words),
                }
            )

            if start + chunk_size >= len(words):
                break

    return chunks
