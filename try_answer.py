
from finsight.generation.answer import generate_answer

retrieved_chunks = [
    {
        "source": "annual_report.pdf",
        "page": 12,
        "text": (
            "The bank increased technology expenditure by 15 percent "
            "during the year. Digital banking adoption also grew "
            "significantly."
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

question = "How much did the bank increase technology expenditure?"

answer = generate_answer(question, retrieved_chunks)

print("\nQuestion:", question)
print("\nGenerated answer:\n", answer)
