from src.llm import generate_answer


def analyze_scope(retrieved_documents):
    """
    Analyze uploaded project documents and extract
    project scope and deliverables.
    """

    question = """
You are an AI Project Scope & Deliverables Analysis Agent.

Analyze the provided project documents and identify:

1. Project Objective
2. Project Scope
3. Main Modules or Features
4. Key Deliverables
5. Important Requirements
6. Important Deadlines or Milestones

Return the result in a clear and professional structured format.

Use ONLY the information available in the provided project documents.

Do NOT invent, assume, or add information that is not
supported by the documents.

If any information is not available, clearly write:
"Not found in the provided documents."
"""

    context = "\n\n".join(
        document["text"]
        for document in retrieved_documents
    )

    return generate_answer(
        question,
        context
    )