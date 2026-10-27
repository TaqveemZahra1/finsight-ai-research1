
from finsight.retrieval.embeddings import (
    embed_texts,
    load_embedding_model,
)

model = load_embedding_model()

texts = [
    "The bank increased its technology expenditure.",
    "The financial institution spent more money on IT.",
    "The weather is sunny today.",
]

vectors = embed_texts(texts, model)

print("Number of texts:", len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("First five values:", vectors[0][:5])
