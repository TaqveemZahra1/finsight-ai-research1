
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL,
) -> SentenceTransformer:
    """Load a sentence-transformer embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
) -> list[list[float]]:
    """Convert text strings into embedding vectors."""
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()
