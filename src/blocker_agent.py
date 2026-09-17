from src.llm import generate_answer


def analyze_blockers(chunks):
    """
    Analyze project documents and identify blockers
    using a fixed output format.
    """

    context_parts = []

    for chunk in chunks:
        filename = chunk.get("filename", "Unknown")
        text = chunk.get("text", "")

        if text and text.strip():
            context_parts.append(
                f"Source: {filename}\n"
                f"Content:\n{text}"
            )

    context = "\n\n".join(context_parts)

    if not context.strip():
        return """
## Blockers and Action Items

### Blocker 1

- **Blocker:** No blocker identified in the uploaded documents
- **Description:** No blocker-related information was found
- **Priority:** Not specified in the uploaded documents
- **Owner:** Not specified in the uploaded documents
- **Required Action:** Continue monitoring the project documents
- **Status:** Not specified in the uploaded documents
"""

    prompt = f"""
You are an AI Project Blocker and Action Item Identification Agent.

Analyze ONLY the uploaded project documents.

Identify:
1. Current blockers
2. Unresolved issues
3. Pending decisions
4. Action items
5. Dependencies causing delays

STRICT OUTPUT RULES:

- Return ONLY the structured output.
- Do not write an introduction.
- Do not write a summary.
- Do not write a conclusion.
- Do not write pending tasks separately.
- Do not add notes at the end.
- Use the exact field names given below.
- Do not rename any field.
- Do not use alternative field names.
- Create one separate block for every blocker.
- Do not repeat the same blocker.
- Use only information found in the uploaded documents.
- Do not invent owners, priorities, or statuses.
- If information is missing, write exactly:
  Not specified in the uploaded documents

DO NOT USE THESE OLD FIELD NAMES:

- Reason
- Impact on the project
- Action Required
- Suggested Owner

USE ONLY THIS EXACT FORMAT:

## Blockers and Action Items

### Blocker 1

- **Blocker:** Name of the blocker
- **Description:** Clear description of the issue
- **Priority:** High / Medium / Low / Not specified in the uploaded documents
- **Owner:** Responsible person or Not specified in the uploaded documents
- **Required Action:** Specific action needed to resolve the blocker
- **Status:** Open / In Progress / Blocked / Resolved / Not specified in the uploaded documents

### Blocker 2

- **Blocker:** Name of the blocker
- **Description:** Clear description of the issue
- **Priority:** High / Medium / Low / Not specified in the uploaded documents
- **Owner:** Responsible person or Not specified in the uploaded documents
- **Required Action:** Specific action needed to resolve the blocker
- **Status:** Open / In Progress / Blocked / Resolved / Not specified in the uploaded documents

Repeat the same format for all unique blockers.

Uploaded project documents:

{context}
"""

    try:
        response = generate_answer(prompt)

        if hasattr(response, "content"):
            return response.content.strip()

        return str(response).strip()

    except Exception as error:
        return f"""
## Blockers and Action Items

### Blocker 1

- **Blocker:** Unable to analyze blockers
- **Description:** An error occurred while analyzing the uploaded documents
- **Priority:** Not specified in the uploaded documents
- **Owner:** Not specified in the uploaded documents
- **Required Action:** Check the LLM configuration and uploaded documents
- **Status:** Blocked

**Technical Error:** {str(error)}
"""