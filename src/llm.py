import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "llama3.2"


def generate_answer(question, retrieved_documents):
    """
    Generate an answer using the locally running Ollama model.
    The model receives only the chunks retrieved by our RAG pipeline.
    """

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"Source: {document['filename']}\n"
            f"Content:\n{document['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an AI Project Intelligence and Risk Advisor.

Answer the user's question using ONLY the information
provided in the retrieved project documents.

Rules:
1. Do not invent project information.
2. If the answer is not available in the documents,
   say that the information was not found.
3. Give a clear and professional answer.
4. Mention the source document when useful.
5. For blockers or risks, explain their impact.

User Question:
{question}

Retrieved Project Context:
{context}

Now provide the answer.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]