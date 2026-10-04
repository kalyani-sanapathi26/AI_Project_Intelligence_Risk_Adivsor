import re
from difflib import SequenceMatcher

from src.llm import generate_answer


def _build_context(
    retrieved_documents,
    scope_analysis="",
    risk_analysis="",
    blocker_analysis="",
    health_analysis=""
):
    """
    Build complete project context.
    """

    document_parts = []

    for document in retrieved_documents:

        text = document.get("text", "")

        if text and text.strip():

            filename = document.get(
                "filename",
                "Unknown document"
            )

            document_parts.append(
                f"Source: {filename}\n"
                f"Content:\n{text}"
            )

    context = "\n\n".join(
        document_parts
    )

    # Add Scope Analysis
    if scope_analysis:

        context += (
            "\n\n===== PROJECT SCOPE ANALYSIS =====\n"
            + str(scope_analysis)
        )

    # Add Risk Analysis
    if risk_analysis:

        context += (
            "\n\n===== RISK ANALYSIS =====\n"
            + str(risk_analysis)
        )

    # Add Blocker Analysis
    if blocker_analysis:

        context += (
            "\n\n===== BLOCKER & ACTION ANALYSIS =====\n"
            + str(blocker_analysis)
        )

    # Add Health Analysis
    if health_analysis:

        context += (
            "\n\n===== PROJECT HEALTH ANALYSIS =====\n"
            + str(health_analysis)
        )

    return context


def _response_text(response):
    """Extract readable text from common LangChain response shapes."""

    if response is None:
        return ""

    if hasattr(response, "content"):
        return _response_text(response.content)

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        if isinstance(response.get("text"), str):
            return response["text"].strip()
        if "content" in response:
            return _response_text(response["content"])
        return ""

    if isinstance(response, (list, tuple)):
        return "\n".join(
            text
            for item in response
            if (text := _response_text(item))
        ).strip()

    if hasattr(response, "text"):
        return _response_text(response.text)

    return ""


def _generate_section(
    prompt,
    context
):
    """
    Generate one documentation section batch.
    """

    try:

        response = generate_answer(
            prompt,
            context
        )

        return _response_text(response)

    except Exception as error:

        print(
            "Documentation generation error:",
            error
        )

        return ""


def _fallback_user_stories(context):
    """Build simple stories only for capabilities named in the context."""

    documented_capabilities = (
        ("Student Management", ("student management", "student records")),
        ("Faculty Management", ("faculty management", "faculty information")),
        ("Attendance", ("attendance",)),
        ("Examination", ("examination", "examinations", "exams")),
        ("Fees", ("fees", "fee payment", "fee processing")),
        ("Notification", ("notification",)),
        ("Integration Testing", ("integration testing",)),
    )
    normalized_context = context.casefold()
    stories = [
        f"- I want to use the documented {capability} capability."
        for capability, evidence_terms in documented_capabilities
        if any(term in normalized_context for term in evidence_terms)
    ]

    return "\n".join(stories)


def _is_unusable_section(content, missing_information):
    content = str(content or "").strip()
    return (
        not content
        or content.casefold() == missing_information.casefold()
        or content.casefold().startswith("all configured llm providers failed")
    )


def _usable_user_stories(content, missing_information):
    if _is_unusable_section(content, missing_information):
        return False

    lines = [line.strip() for line in str(content).splitlines() if line.strip()]
    for line in lines:
        story_match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if not story_match or not story_match.group(1).rstrip().endswith((".", "!", "?")):
            return False

    return bool(lines)


def _extract_risk_records(*sources):
    risk_start = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\d+[.)]\s*)?"
        r"(?:\*\*)?Risk(?:\s+\d+)?(?:\*\*)?\s*:\s*(?:\*\*)?(.+?)\s*$"
    )
    field_pattern = re.compile(
        r"(?i)^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
        r"(Potential Project Impact|Recommended Action|Required Action|"
        r"Description|Impact|Probability|Likelihood|Severity|Owner|Status|"
        r"Mitigation)\s*\*{0,2}\s*[:\-]\s*\*{0,2}\s*(.*?)\s*$"
    )
    records = {}

    for source in sources:
        text = str(source or "").replace("\\n", "\n")
        matches = list(risk_start.finditer(text))
        for index, match in enumerate(matches):
            name = match.group(1).strip().strip("*")
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.end():end]
            fields = {}
            current_field = None

            for line in block.splitlines():
                if line.lstrip().startswith(("=====", "#")) or re.match(
                    r"(?i)^\s*Overall Risk Summary\s*:", line
                ):
                    current_field = None
                    continue

                field_match = field_pattern.match(line)
                if field_match:
                    label = field_match.group(1).casefold()
                    if label in ("potential project impact", "impact"):
                        current_field = "Impact"
                    elif label in ("probability", "likelihood"):
                        current_field = "Probability"
                    elif label == "recommended action":
                        current_field = "Recommended Action"
                    elif label == "required action":
                        current_field = "Required Action"
                    else:
                        current_field = label.title()

                    value = field_match.group(2).strip().strip("*")
                    if value:
                        fields[current_field] = value
                    continue

                if current_field and line.strip():
                    value = line.strip().strip("-*• ")
                    if value and not value.startswith("Source:"):
                        fields[current_field] = " ".join(
                            filter(None, (fields.get(current_field, ""), value))
                        )

            record_key = name.casefold()
            existing = records.setdefault(record_key, {"Risk": name})
            existing.update({key: value for key, value in fields.items() if value})

    # Risk facts in uploaded documents and analyses are not always formatted as
    # explicit ``Risk: ...`` records. Preserve only the source lines that state
    # a concrete uncertainty, delay, dependency, or limitation.
    risk_evidence_pattern = re.compile(
        r"(?i)\b(?:risk|delay(?:ed)?|blocked|blocker|pending|not finalized|"
        r"unfinished|limited(?:\s+[\w-]+){0,3}\s+availability|"
        r"availability limitation|"
        r"dependency|schedule slip|schedule delay|behind schedule)\b"
    )
    for source in sources:
        for line in str(source or "").replace("\\n", "\n").splitlines():
            evidence = re.sub(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)", "", line)
            evidence = evidence.replace("**", "").strip()
            heading_text = evidence.strip(" =#*-\t")
            if re.fullmatch(
                r"(?i)#{0,6}\s*(?:risk(?: analysis| register)?|key risks|"
                r"overall risk summary)\s*:?\s*",
                heading_text
            ):
                continue
            if (
                not evidence
                or evidence.casefold().startswith(("source:", "content:"))
                or len(evidence) > 500
                or not risk_evidence_pattern.search(evidence)
            ):
                continue

            normalized = re.sub(r"(?i)^risk\s*:\s*", "", evidence).strip()
            if not normalized or normalized.casefold() in records:
                continue
            records.setdefault(normalized.casefold(), {"Risk": normalized})

    return list(records.values())


def _fallback_risk_register(context, risk_analysis=""):
    records = _extract_risk_records(risk_analysis, context)
    output = []
    field_order = (
        "Description",
        "Impact",
        "Severity",
        "Probability",
        "Owner",
        "Status",
        "Mitigation",
        "Recommended Action",
        "Required Action",
    )

    for record in records:
        lines = [f"- **Risk:** {record['Risk']}"]
        lines.extend(
            f"  - **{field}:** {record[field]}"
            for field in field_order
            if record.get(field)
        )
        output.append("\n".join(lines))

    return "\n".join(output)


def _fallback_action_items(context, blocker_analysis="", risk_analysis=""):
    action_label = re.compile(
        r"(?i)^(Required Action|Action Required|Recommended Action|"
        r"Mitigation|Next Steps?|Follow[- ]?up Actions?|Action Items?)"
        r"\s*[:\-]?\s*(.*)$"
    )
    item_pattern = re.compile(r"^(?:[-*•]|(?:A\s*)?\d+[.)])\s+(.+?)\s*$", re.I)
    stop_pattern = re.compile(
        r"(?i)^(Risk(?:\s+\d+)?|Description|Impact|Potential Project Impact|"
        r"Probability|Likelihood|Severity|Owner|Status|Priority|Blocker|"
        r"Source|Overall Risk Summary)\s*[:\-]"
    )
    action_heading = re.compile(
        r"(?i)^#{0,6}\s*(?:recommended\s+)?(?:action items?|next steps?)\s*:?\s*$"
    )
    action_verbs = re.compile(
        r"(?i)^(?:confirm|prepare|complete|continue|assign|track|escalate|"
        r"update|review|create|test|resolve|implement|finalize|integrate|"
        r"document|coordinate|validate|deploy|conduct|monitor|prioritize)\b"
    )
    actions = {}

    for source in (blocker_analysis, risk_analysis, context):
        lines = str(source or "").replace("\\n", "\n").splitlines()
        in_action_section = False
        for index, line in enumerate(lines):
            stripped_line = line.strip()
            normalized_heading = stripped_line.replace("**", "").strip()
            if action_heading.match(normalized_heading):
                in_action_section = True
                continue
            if stripped_line.startswith(("#", "=====")):
                in_action_section = False

            normalized_line = re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*)", "", line)
            normalized_line = normalized_line.replace("**", "").strip()
            label_match = action_label.match(normalized_line)
            if label_match:
                inline_value = label_match.group(2).strip().strip("-*• ")
                candidates = [inline_value] if inline_value else []
                for following_line in lines[index + 1:]:
                    if not following_line.strip():
                        if candidates:
                            break
                        continue

                    following = following_line.strip()
                    following_normalized = re.sub(
                        r"^(?:[-*•]\s*|\d+[.)]\s*)", "", following
                    ).replace("**", "").strip()
                    if action_label.match(following_normalized) or stop_pattern.match(following_normalized):
                        break

                    item_match = item_pattern.match(following)
                    if item_match:
                        candidates.append(item_match.group(1).strip())
                        continue

                    if not candidates and label_match.group(1).casefold() != "mitigation":
                        candidates.append(following_normalized)
                    break

                for candidate in candidates:
                    candidate = candidate.strip().strip("-*• ")
                    if not candidate or re.search(
                        r"(?i)^(not specified|not found|none|n/?a|specific action needed|"
                        r"recommended action|continue monitoring the project documents)\b",
                        candidate
                    ):
                        continue
                    actions.setdefault(candidate.casefold(), candidate)

            # Also preserve source-authored imperative action bullets, including
            # numbered action IDs, even when they are not under a recognized field.
            item_match = item_pattern.match(stripped_line)
            if item_match:
                candidate = item_match.group(1).strip().replace("**", "")
                if action_verbs.match(candidate):
                    actions.setdefault(candidate.casefold(), candidate)

            # Some documents put required actions under a titled action list.
            if in_action_section and item_match:
                candidate = item_match.group(1).strip().replace("**", "")
                if candidate and not stop_pattern.match(candidate):
                    actions.setdefault(candidate.casefold(), candidate)

            # Capture an explicit imperative sentence in prose only when it begins
            # with an action verb; the wording is copied from supplied context.
            for sentence in re.split(r"(?<=[.!?])\s+", normalized_line):
                sentence = sentence.strip(" -*•")
                if action_verbs.match(sentence):
                    actions.setdefault(sentence.casefold(), sentence)

    return "\n".join(f"- {action}" for action in actions.values())


def _deduplicate_action_items(content):
    """Remove repeated action items while retaining their clearest wording."""

    item_pattern = re.compile(r"^(\s*)(?:[-*•]|(?:A\s*)?\d+[.)])\s+(.+?)\s*$", re.I)
    kept_lines = []
    seen = []
    comparison_fillers = {
        "a", "an", "the", "immediately", "promptly", "urgently", "regularly",
        "early", "advance", "work", "on", "independent", "module", "modules",
        "such", "as", "project", "task", "tasks", "fee", "integration",
    }

    def normalized_action(text):
        text = re.split(
            r"\b(?:after|before|once|when|if|while|until)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        words = re.findall(r"[a-z0-9]+", text.casefold())
        normalized = []
        for word in words:
            if word in comparison_fillers:
                continue
            if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            normalized.append(word)
        return tuple(normalized)

    for line in str(content or "").splitlines():
        match = item_pattern.match(line)
        if not match:
            kept_lines.append(line)
            continue

        item_text = match.group(2).replace("**", "").strip()
        comparison_text = re.split(
            r"\s+(?:\||;|—|–)\s*(?:owner|priority|due date|status)\s*:",
            item_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        normalized_words = normalized_action(comparison_text)

        if not normalized_words:
            kept_lines.append(line)
            continue

        duplicate_index = None
        for index, previous in enumerate(seen):
            previous_words = previous["words"]
            same_action = normalized_words[0] == previous_words[0]
            current_target = set(normalized_words[1:])
            previous_target = set(previous_words[1:])
            current_text = " ".join(normalized_words)
            previous_text = " ".join(previous_words)
            token_similarity = len(current_target & previous_target) / max(
                len(current_target | previous_target), 1
            )

            if normalized_words == previous_words or (
                same_action
                and (
                    (
                        bool(current_target)
                        and bool(previous_target)
                        and (
                            current_target.issubset(previous_target)
                            or previous_target.issubset(current_target)
                            or token_similarity >= 0.9
                        )
                    )
                    or SequenceMatcher(None, current_text, previous_text).ratio() >= 0.9
                )
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            seen.append({"words": normalized_words, "line_index": len(kept_lines)})
            kept_lines.append(line)
        else:
            previous = seen[duplicate_index]
            previous_target = set(previous["words"][1:])
            current_target = set(normalized_words[1:])
            # Prefer the more specific wording when one item adds a concrete
            # target; otherwise retain the first documented wording.
            if previous_target.issubset(current_target) and previous_target != current_target:
                kept_lines[previous["line_index"]] = line
                seen[duplicate_index] = {
                    "words": normalized_words,
                    "line_index": previous["line_index"],
                }

    return "\n".join(kept_lines).strip()


def generate_project_documentation(
    retrieved_documents,
    scope_analysis="",
    risk_analysis="",
    blocker_analysis="",
    health_analysis=""
):
    """
    Generate complete project documentation.
    """

    # ---------------------------------------------
    # BUILD COMPLETE CONTEXT
    # ---------------------------------------------

    context = _build_context(
        retrieved_documents,
        scope_analysis,
        risk_analysis,
        blocker_analysis,
        health_analysis
    )

    if not context.strip():

        return (
            "No relevant project information was found "
            "in the uploaded project documents."
        )

    missing_information = "Not found in the provided project information."

    section_prompts = [
        (
            "USER STORIES",
            """
You are generating only the User Stories section of project documentation.
Read the provided project context and identify documented modules, features, requirements, and activities that describe project capabilities. Create one concise user story for each distinct supported capability; the source does not need a section labelled "User Stories" or user-story wording.
Use only facts supported by the context. Do not invent a role or benefit. Write each story as a bullet in this simple format:
- I want to [documented capability].
Add an "As a ..." role or "so that ..." benefit only if the context explicitly supports it. Do not include placeholders for missing fields. Avoid duplicates. If no capability is documented, return exactly:
Not found in the provided project information.

Return only the story bullets, with no heading or other section.
"""
        ),
        (
            "RISK REGISTER",
            """
You are generating only the Risk Register section of project documentation.
Review the entire provided context, including every retrieved document and optional analysis.
Treat documented unresolved dependencies or specifications, delays, resource limitations, blockers, and other explicit uncertainties as risk evidence even if they are not labeled "risk".
Do not require the source to contain a section labelled "Risk Register". Include each distinct risk supported by the context, describing its documented uncertainty or blocker. Add an impact, likelihood, mitigation, owner, date, or status only when explicitly supported; omit unavailable fields instead of rejecting the risk or writing a missing-information placeholder. If no risk evidence is present, return exactly:
Not found in the provided project information.
Do not include a Project Health section or a separate Blockers section.

Return only concise readable bullets, with no heading or other section. Include documented mitigation or impact details inline only when available.
"""
        ),
        (
            "ACTION ITEMS",
            """
You are generating only the Action Items section of project documentation.
Review the entire provided context, including every retrieved document and optional analysis.
Extract documented action statements and actionable next steps supported by blockers, dependencies, unfinished integrations, or testing constraints. Where the context documents unfinished work or a dependency, state the direct step needed to resolve or complete it; do not require the source to phrase that step as an imperative.
Avoid duplicate or repetitive actions and generic project-management actions.
Do not invent actions, owners, priorities, due dates, or statuses. Omit unavailable metadata rather than adding a missing-information placeholder. Do not require the source to use a particular action-list heading.
If no documented or clearly required action is supported by the context, return exactly:
Not found in the provided project information.

Return only concise readable bullets or a numbered list, with no heading or other section. Include an owner or other metadata inline only when the context provides it.
"""
        ),
        (
            "PROJECT SUMMARY",
            """
You are generating only the concise Project Summary section of project documentation.
Use only information from the provided context. Do not repeat the risk register or action-item table.
When an initial interface or prototype is documented as complete while its integration remains in progress, report those as separate statuses; do not describe the whole feature as complete.
For every unavailable field, write exactly:
Not found in the provided project information.
Do not add subsections, health scoring, a RAG workflow, AI agents, or validation details.

Return only these fields, with no heading or other section:
Project Name:
Project Objective:
Main Modules / Features:
Current Status:
Completed Work:
Work In Progress:
Blocked / Pending Work:
Key Risks:
"""
        ),
    ]

    documentation_parts = []
    successful_sections = 0

    for section_number, (section_name, prompt) in enumerate(section_prompts, start=1):

        section_content = _generate_section(
            prompt,
            context
        )
        had_section_content = bool(section_content)

        if section_content:
            successful_sections += 1
            section_content = "\n".join(
                line
                for line in section_content.splitlines()
                if not line.lstrip().startswith("#")
            ).strip()

        if section_name == "USER STORIES" and not _usable_user_stories(
            section_content,
            missing_information
        ):
            section_content = _fallback_user_stories(context)
        elif section_name == "RISK REGISTER" and _is_unusable_section(
            section_content,
            missing_information
        ):
            section_content = _fallback_risk_register(context, risk_analysis)
        elif section_name == "ACTION ITEMS" and _is_unusable_section(
            section_content,
            missing_information
        ):
            section_content = _fallback_action_items(
                context,
                blocker_analysis,
                risk_analysis
            )

        if section_name == "ACTION ITEMS":
            section_content = _deduplicate_action_items(section_content)
        elif section_name == "USER STORIES":
            # Render one deterministic story per capability evidenced in context;
            # module interfaces and integrations collapse to their parent capability.
            section_content = _fallback_user_stories(context) or missing_information

        if section_content and not had_section_content:
            successful_sections += 1

        if not section_content:
            section_content = missing_information

        documentation_parts.append(
            f"# {section_number}. {section_name}\n\n{section_content}"
        )

    if successful_sections == 0:

        return (
            "Documentation generation failed. "
            "The LLM did not return a usable response."
        )

    return "\n\n".join(documentation_parts)
