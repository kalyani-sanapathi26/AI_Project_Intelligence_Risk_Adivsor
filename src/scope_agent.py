from src.llm import generate_answer


def analyze_scope(retrieved_documents):
    """
    Analyze uploaded project documents and extract
    project scope and deliverables.
    """

    question = """
Analyze the uploaded project documents and identify:

1. Project Objective
2. Project Scope
3. Main Modules or Features
4. Key Deliverables
5. Important Requirements
6. Important Deadlines or Milestones

Return the result in a clear structured format.

Use ONLY the information available in the retrieved documents.
Do not invent information.
"""

    return generate_answer(
        question,
        retrieved_documents
    )