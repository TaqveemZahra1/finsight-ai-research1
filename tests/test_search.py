
import numpy as np
import pytest

from finsight.retrieval.search import retrieve_chunks


class FakeEmbeddingModel:
    """A predictable model for testing retrieval without downloading a model."""

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        vectors = []

        for text in texts:
            if text == "technology spending":
                vector = [1.0, 0.0]
            elif text == "IT expenditure increased":
                vector = [0.9, 0.1]
            elif text == "technology costs":
                vector = [0.8, 0.2]
            else:
                vector = [0.0, 1.0]

            vectors.append(vector)

        vectors = np.array(vectors, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if normalize_embeddings:
            vectors = vectors / norms

        return vectors


def test_retrieve_chunks_ranks_relevant_results():
    chunks = [
        {
            "source": "report.pdf",
            "page": 1,
            "text": "Weather conditions improved.",
        },
        {
            "source": "report.pdf",
            "page": 2,
            "text": "IT expenditure increased",
        },
        {
            "source": "report.pdf",
            "page": 3,
            "text": "technology costs",
        },
    ]

    results = retrieve_chunks(
        "technology spending",
        chunks,
        FakeEmbeddingModel(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == "IT expenditure increased"
    assert results[1]["text"] == "technology costs"
    assert results[0]["score"] > results[1]["score"]


def test_retrieve_chunks_returns_empty_for_no_chunks():
    assert retrieve_chunks(
        "technology spending",
        [],
        FakeEmbeddingModel(),
    ) == []


def test_retrieve_chunks_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_chunks(
            "technology spending",
            [],
            FakeEmbeddingModel(),
            top_k=0,
        )


def test_retrieve_chunks_rejects_empty_query():
    with pytest.raises(ValueError):
        retrieve_chunks(
            "   ",
            [],
            FakeEmbeddingModel(),
        )
