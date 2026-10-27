
from pathlib import Path

from finsight.ingestion.pdf_loader import extract_pdf
from finsight.ingestion.chunker import chunk_text
from finsight.retrieval.embeddings import load_embedding_model
from finsight.retrieval.search import retrieve_chunks
from finsight.generation.answer import generate_answer


def main():
    print("\n=== FinSight AI Research Lab ===")

    pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

    if not Path(pdf_path).is_file():
        print("Error: PDF file not found. Please check the path.")
        return

    print("\n[1/4] Extracting PDF text...")
    pages = extract_pdf(pdf_path)

    if not pages:
        print("No readable text found in this PDF.")
        return

    print(f"Extracted text from {len(pages)} pages.")

    print("\n[2/4] Splitting text into chunks...")
    chunks = chunk_text(pages, chunk_size=200, overlap=40)
    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No text chunks were created.")
        return

    print("\n[3/4] Loading embedding model...")
    model = load_embedding_model()

    print("\n[4/4] Ready to answer questions.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = retrieve_chunks(
            query=question,
            chunks=chunks,
            model=model,
            top_k=3,
        )

        answer = generate_answer(question, results)

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved passages:")
        for result in results:
            print(
                f"- {result['source']}, page {result['page']} "
                f"(similarity: {result['score']:.3f})"
            )


if __name__ == "__main__":
    main()
