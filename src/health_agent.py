from src.llm import generate_answer


def analyze_project_health(retrieved_documents):
    """
    Analyze the overall health of the project based on
    the retrieved project documents.
    """

    question = """
Evaluate the overall health of the project using the
retrieved project documents.

Analyze these areas:

1. Overall Project Health
   - Healthy
   - At Risk
   - Critical

2. Progress Status

3. Risk Level
   - Low
   - Medium
   - High

4. Blocker Level
   - Low
   - Medium
   - High

5. Delivery Confidence
   - High
   - Medium
   - Low

6. Key Reasons for the Health Status

7. Immediate Actions Required

Return the result in a clear professional format.

Use ONLY information available in the retrieved documents.
Do not invent project information.
"""

    return generate_answer(
        question,
        retrieved_documents
    )