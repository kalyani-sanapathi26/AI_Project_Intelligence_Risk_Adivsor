from src.llm import generate_answer


def analyze_risks(retrieved_documents):
    """
    Analyze project documents and identify risks,
    their severity, impact, and recommended actions.
    """

    question = """
Analyze the retrieved project documents and identify the
important project risks and delivery challenges.

For each risk, provide:

1. Risk
2. Severity - High, Medium, or Low
3. Reason
4. Potential Project Impact
5. Recommended Action

Also identify:
6. Schedule or delivery risks
7. Dependencies that may cause delays

Return the result in a clear and structured format.

Use ONLY the information available in the retrieved
project documents. Do not invent risks or information.
"""

    return generate_answer(
        question,
        retrieved_documents
    )