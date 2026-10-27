
def retrieve_chunks(
    query: str,
    chunks: list[dict],
    model,
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant chunks for a query."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not query.strip():
        raise ValueError("query must not be empty.")

    if not chunks:
        return []

    # Create vectors for the query and all document chunks.
    texts = [query] + [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = embeddings[0]
    chunk_vectors = embeddings[1:]

    # Normalized vectors can be compared using their dot product.
    scores = chunk_vectors @ query_vector

    # Rank chunk indexes from highest similarity to lowest.
    ranked_indexes = sorted(
        range(len(chunks)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results = []

    for index in ranked_indexes[:top_k]:
        results.append(
            {
                **chunks[index],
                "score": float(scores[index]),
            }
        )

    return results
