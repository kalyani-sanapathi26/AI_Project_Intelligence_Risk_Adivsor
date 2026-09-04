from src.llm import generate_answer


def analyze_blockers(retrieved_documents):
    """
    Analyze project documents and identify blockers,
    action items, owners, and recommended next steps.
    """

    question = """
Analyze the retrieved project documents and identify
the current project blockers and required action items.

For each blocker, provide:

1. Blocker
2. Reason
3. Impact on the project
4. Action Required
5. Suggested Owner, if mentioned in the documents
6. Priority - High, Medium, or Low

Also identify any pending tasks that require immediate attention.

Use ONLY the information available in the retrieved
project documents. Do not invent information.

Return the result in a clear and structured format.
"""

    return generate_answer(
        question,
        retrieved_documents
    )