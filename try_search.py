
from finsight.ingestion.chunker import chunk_text
from finsight.retrieval.embeddings import load_embedding_model
from finsight.retrieval.search import retrieve_chunks


pages = [
    {
        "source": "annual_report.pdf",
        "page": 12,
        "text": (
            "The bank increased technology expenditure by 15 percent. "
            "Digital banking adoption grew significantly during the year. "
            "Employee training costs remained stable."
        ),
    },
    {
        "source": "annual_report.pdf",
        "page": 18,
        "text": (
            "The bank expanded its IT infrastructure and invested "
            "in online banking services."
        ),
    },
]

chunks = chunk_text(pages, chunk_size=40, overlap=5)
model = load_embedding_model()

results = retrieve_chunks(
    query="Did the bank spend more money on technology?",
    chunks=chunks,
    model=model,
    top_k=2,
)

for result in results:
    print(f"\nSource: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Text: {result['text']}")
