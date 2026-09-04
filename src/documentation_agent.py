from src.llm import generate_answer


def generate_project_documentation(retrieved_documents):
    """
    Generate a structured project documentation summary
    from the retrieved project documents.
    """

    question = """
Create a professional project documentation summary using
the retrieved project documents.

Include these sections:

1. Project Overview
2. Objectives
3. Functional Requirements
4. Main Modules
5. Key Deliverables
6. Current Project Status
7. Identified Risks
8. Current Blockers
9. Pending Tasks
10. Important Milestones

Use ONLY information available in the retrieved documents.

Do not invent or assume project information.

If a particular piece of information is not available,
clearly mention that it was not found in the documents.

Present the result with clear headings and bullet points.
"""

    return generate_answer(
        question,
        retrieved_documents
    )