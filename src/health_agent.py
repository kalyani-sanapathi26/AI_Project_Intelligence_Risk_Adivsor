from src.llm import generate_answer


def analyze_project_health(retrieved_documents):
    """
    Analyze project health using only retrieved project documents.
    """

    context = "\n\n".join(
        document.get("text", "")
        for document in retrieved_documents
        if document.get("text")
    )

    if not context.strip():
        return """
# PROJECT HEALTH SCORE

# PROJECT HEALTH SCORE

Score: Insufficient information in the provided documents.
Status: Insufficient information in the provided documents.

# DIMENSION SCORES

Scope Clarity: Insufficient information in the provided documents.
Progress: Insufficient information in the provided documents.
Timeline / Delivery: Insufficient information in the provided documents.
Risk Management: Insufficient information in the provided documents.
Blocker Management: Insufficient information in the provided documents.
Delivery Confidence: Insufficient information in the provided documents.

# PROGRESS STATUS

Insufficient information in the provided documents.

# KEY REASONS

- Insufficient information in the provided documents.

# IMMEDIATE ACTIONS REQUIRED

1. Upload and process sufficient project documents.
2. Provide project progress, timeline, risk, and blocker information.

# HEALTH SUMMARY

Insufficient information in the provided documents.
"""

    question = """
You are an AI Project Health Scoring Agent.

Evaluate the project health using ONLY the information
available in the provided project document context.

Do NOT invent, assume, or create project information.

==================================================
1. DIMENSION SCORES
==================================================

Evaluate these six dimensions from 0 to 100:

- Scope Clarity
- Progress
- Timeline / Delivery
- Risk Management
- Blocker Management
- Delivery Confidence

Every numeric score MUST be supported by evidence
from the project documents.

If there is not enough evidence for a dimension, write:

Insufficient information in the provided documents.

==================================================
2. OVERALL HEALTH SCORE
==================================================

Calculate the Overall Health Score using the available
numeric dimension scores.

When all six dimension scores are available:

Overall Health Score =
average of the six dimension scores.

Round the result to the nearest whole number.

Do NOT choose an arbitrary overall score.

If one or more dimensions have insufficient information,
calculate the average only from the available numeric
dimension scores and clearly mention that some dimensions
had insufficient information.

==================================================
3. OVERALL HEALTH STATUS
==================================================

Choose exactly one:

- Healthy
- At Risk
- Critical

Use these score ranges:

80-100 = Healthy
50-79 = At Risk
0-49 = Critical

The status MUST match the overall numerical score.

==================================================
4. PROGRESS STATUS
==================================================

Explain the current project progress using ONLY evidence
from the provided documents.

==================================================
5. KEY REASONS
==================================================

List the main factors affecting project health.

Use only documented information.

==================================================
6. IMMEDIATE ACTIONS REQUIRED
==================================================

List the most important actions required to improve
project health.

Actions must be based on documented risks, blockers,
delays, dependencies, or incomplete work.

Do NOT invent owners or deadlines.

==================================================
7. HEALTH SUMMARY
==================================================

Give a short professional summary based only on
the provided project documents.

==================================================
STRICT RULES
==================================================

- Use ONLY information from the provided project documents.
- Do NOT invent numbers.
- Do NOT invent deadlines.
- Do NOT invent owners.
- Do NOT invent risks.
- Do NOT invent blockers.
- Do NOT invent progress.
- Do NOT invent project status.
- Do NOT assume missing information.
- Keep all scores consistent with the evidence.
- Overall score MUST be calculated from the dimension scores.
- Overall status MUST match the overall score.
- Do not add extra sections.

Return EXACTLY this structure:

# PROJECT HEALTH SCORE

Score: XX/100
Status: Healthy / At Risk / Critical

# DIMENSION SCORES

Scope Clarity: XX/100
Progress: XX/100
Timeline / Delivery: XX/100
Risk Management: XX/100
Blocker Management: XX/100
Delivery Confidence: XX/100

# PROGRESS STATUS

...

# KEY REASONS

- ...
- ...
- ...

# IMMEDIATE ACTIONS REQUIRED

1. ...
2. ...
3. ...

# HEALTH SUMMARY

...
"""

    return generate_answer(
        question,
        context
    )