import re
from difflib import SequenceMatcher

from src.llm import generate_answer


def _build_document_context(retrieved_documents):
    document_parts = []

    for document in retrieved_documents:
        text = document.get("text", "")
        if text and text.strip():
            filename = document.get("filename", "Unknown document")
            document_parts.append(f"Source: {filename}\nContent:\n{text}")

    return "\n\n".join(document_parts)


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

    context = _build_document_context(retrieved_documents)

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
    normalized_context = re.sub(r"[^a-z0-9]+", " ", context.casefold())
    capabilities = [
        capability
        for capability, evidence_terms in documented_capabilities
        if any(re.sub(r"[^a-z0-9]+", " ", term.casefold()) in normalized_context for term in evidence_terms)
    ]
    return "\n\n".join(
        f"US-{index:02d}\n"
        f"User Story: As a project user, I want to use the documented "
        f"{capability} capability so that project work involving "
        f"{capability} is supported.\n"
        f"Related Module: {capability}"
        for index, capability in enumerate(capabilities, start=1)
    )


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
        r"(?im)^\s*(?:#{1,6}\s*)?(?:[-*•]\s*)?(?:\d+[.)]\s*)?"
        r"(?:\*\*)?Risk(?:\s+\d+)?(?:\*\*)?\s*:\s*(?:\*\*)?(.+?)\s*$"
    )
    field_pattern = re.compile(
        r"(?i)^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
        r"(Mitigation\s*/\s*Required Action|Severity/Impact|Potential Project Impact|Recommended Action|Required Action|"
        r"Description|Reason|Impact|Probability|Likelihood|Severity|Owner|Status|"
        r"Mitigation)\s*\*{0,2}\s*[:\-]\s*\*{0,2}\s*(.*?)\s*$"
    )
    records = {}

    def add_record(name, fields):
        name = re.sub(r"\s+", " ", name.replace("**", "")).strip(" -*•|:")
        if (
            not name
            or len(name) > 240
            or "|" in name
            or re.fullmatch(r"(?i)(?:management|analysis|register|summary|pending|blocked|risk)", name)
            or re.match(r"(?i)^(?:task\s*#?\w+|t\d+|project health|scope|blocker)\b", name)
            or re.search(r"(?i)\b(?:\d{1,3}\s*/\s*100|completed|in progress|not started)\b", name)
        ):
            return
        key = re.sub(r"[^a-z0-9]+", " ", re.sub(r"(?i)^Risk(?:\s+\d+)?\s*:\s*", "", name).casefold()).strip()
        for old_key, old_record in records.items():
            old_words, new_words = set(old_key.split()), set(key.split())
            overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)
            if (
                key == old_key
                or (old_words and new_words and old_words <= new_words)
                or (old_words and new_words and new_words <= old_words)
                or (old_words and new_words and overlap >= 0.82)
                or SequenceMatcher(None, old_key, key).ratio() >= 0.9
            ):
                old_record.update({field: value for field, value in fields.items() if value and not old_record.get(field)})
                return
        records[key] = {"Risk": name, **{field: value for field, value in fields.items() if value}}

    for source in sources:
        text = str(source or "").replace("\\n", "\n")
        normalized_lines = []
        for line in text.splitlines():
            line = re.sub(
                r"^\s*(?:[-*•]\s*)?\*\*\s*[-*•]\s*([^*]+?)\s*\*\*\s*:\s*",
                r"\1: ",
                line,
            )
            line = re.sub(
                r"^\s*(?:[-*•]\s*)?\*\*([^*]+?)\*\*\s*:\s*",
                r"\1: ",
                line,
            )
            normalized_lines.append(line)
        text = "\n".join(normalized_lines)
        lines = text.splitlines()
        matches = list(risk_start.finditer(text))
        for index, match in enumerate(matches):
            name = match.group(1).strip().strip("*")
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.end():block_end]
            fields, current_field = {}, None
            for line in block.splitlines():
                if "|" in line or line.lstrip().startswith(("=====", "#")):
                    current_field = None
                    continue
                field_match = field_pattern.match(line)
                if field_match:
                    label = field_match.group(1).casefold()
                    current_field = {
                        "mitigation / required action": "Mitigation",
                        "severity/impact": "Severity/Impact",
                        "potential project impact": "Impact", "impact": "Impact",
                        "reason": "Description",
                        "probability": "Probability", "likelihood": "Probability",
                        "recommended action": "Recommended Action", "required action": "Required Action",
                    }.get(label, label.title())
                    value = field_match.group(2).strip().strip("* ")
                    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
                    value = re.sub(r"(?i)\s+Priority\s*:\s*[^\n]+$", "", value).strip()
                    if value and not value.casefold().startswith(("not found", "not specified")):
                        fields[current_field] = value
                    continue
                if current_field and line.strip() and not line.strip().startswith(("Source:", "Content:")):
                    value = line.strip().strip("-*• ")
                    if value and "|" not in value:
                        fields[current_field] = " ".join(filter(None, (fields.get(current_field, ""), value)))
            add_record(name, fields)

        # Tables are accepted only inside the risk-specific input (the LLM risk
        # response or Risk Analysis). Headers define columns; numeric row IDs
        # without a header are interpreted only as Risk | Severity rows.
        table_headers = None
        for line in lines:
            if "|" not in line:
                continue
            cells = [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]
            normalized_cells = [re.sub(r"[^a-z ]", "", cell.casefold()).strip() for cell in cells]
            if any("risk" == cell or cell.startswith("risk name") for cell in normalized_cells):
                table_headers = normalized_cells
                continue
            if not cells or all(re.fullmatch(r"[-: ]+", cell or "-") for cell in cells):
                continue
            if table_headers:
                field_aliases = {
                    "risk": "Risk", "risk name": "Risk", "severity": "Severity",
                    "impact": "Impact", "severity impact": "Severity/Impact",
                    "probability": "Probability", "likelihood": "Probability",
                    "owner": "Owner", "status": "Status", "description": "Description",
                    "reason": "Description", "mitigation": "Mitigation",
                    "recommended action": "Recommended Action", "required action": "Required Action",
                }
                row = {field_aliases[header]: value for header, value in zip(table_headers, cells) if header in field_aliases and value}
                risk_name = row.pop("Risk", "")
                if risk_name and risk_name.casefold() not in {"risk", "risk name"}:
                    add_record(risk_name, row)
                continue
            if len(cells) >= 2 and re.fullmatch(r"\d+", cells[0]) and cells[1].casefold() not in {"risk", "risk name"}:
                fields = {}
                if len(cells) >= 3 and cells[2].casefold() in {"high", "medium", "low", "critical"}:
                    fields["Severity"] = cells[2]
                add_record(cells[1], fields)

    return list(records.values())


def _fallback_risk_register(context, risk_analysis=""):
    records = _extract_risk_records(risk_analysis, context)
    output = []
    for index, record in enumerate(records, start=1):
        lines = [f"Risk {index}: {record['Risk']}"]
        severity_impact = record.get("Severity/Impact") or " / ".join(
            record[field] for field in ("Severity", "Impact") if record.get(field)
        )
        if severity_impact:
            lines.append(f"Severity/Impact: {severity_impact}")
        if record.get("Probability"):
            lines.append(f"Probability: {record['Probability']}")
        lines.append(f"Owner: {record.get('Owner', 'Not specified')}")
        if record.get("Status"):
            lines.append(f"Status: {record['Status']}")
        if record.get("Description"):
            lines.append(f"Description: {record['Description']}")
        action = record.get("Mitigation") or record.get("Recommended Action") or record.get("Required Action")
        if action:
            lines.append(f"Mitigation / Required Action: {action}")
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


def _clean_action_items(content):
    action_verbs = re.compile(
        r"(?i)^(?:confirm|prepare|complete|continue|assign|track|escalate|"
        r"update|review|create|test|resolve|implement|finalize|integrate|"
        r"document|coordinate|validate|deploy|conduct|monitor|prioritize|"
        r"request|verify|provide|ensure|schedule|identify|define|develop)\b"
    )
    records = []
    for line in str(content or "").replace("\\n", "\n").splitlines():
        line = line.strip()
        if not line or "|" in line or line.startswith(("#", "=====")):
            continue
        line = line.replace("**", "").replace("`", "")
        if re.fullmatch(r"\d+[.)]", line):
            continue
        item_match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+?)\s*$", line)
        candidate = item_match.group(1).strip() if item_match else line
        candidate = re.sub(r"(?i)^Action\s*\d*\s*[:.)-]\s*", "", candidate).strip()
        if re.match(r"(?i)^(?:owner|priority|status|due date)\s*:", candidate):
            if records:
                label, value = candidate.split(":", 1)
                if label.casefold() in ("owner", "priority") and value.strip():
                    records[-1][label.title()] = value.strip()
            continue
        if candidate.casefold().startswith((
            "not found", "no documented actions", "action items:",
            "risk register", "risk analysis", "blocker analysis", "health analysis",
            "project scope", "project summary",
        )):
            continue
        if candidate.casefold().rstrip(".!") in {"pending", "blocked", "completed", "in progress", "not started"}:
            continue
        if re.match(r"(?i)^(?:risk(?: register| analysis)?|blocker analysis|health analysis|scope analysis)\s*[:#]", candidate):
            continue
        if not action_verbs.match(candidate):
            continue
        metadata = {}
        for label in ("Owner", "Priority"):
            match = re.search(rf"(?i)(?:\||;|—|–)\s*{label}\s*:\s*([^;|]+)", candidate)
            if match:
                metadata[label] = match.group(1).strip()
                candidate = candidate[:match.start()].strip()
        candidate = candidate.strip(" -*•|;:")
        if candidate:
            records.append({"Action": candidate, **metadata})

    if not records:
        return ""

    dedupe_input = "\n".join(
        "- " + record["Action"] + "".join(
            f" | {field}: {record[field]}" for field in ("Owner", "Priority") if record.get(field)
        )
        for record in records
    )
    deduped_lines = _deduplicate_action_items(dedupe_input).splitlines()
    cleaned_records = []
    for line in deduped_lines:
        match = re.match(r"^\s*[-*•]\s+(.+?)\s*$", line)
        if not match:
            continue
        value = match.group(1)
        record = {"Action": value}
        for label in ("Owner", "Priority"):
            metadata_match = re.search(rf"(?i)(?:\||;|—|–)\s*{label}\s*:\s*([^;|]+)", value)
            if metadata_match:
                record[label] = metadata_match.group(1).strip()
                value = value[:metadata_match.start()].strip()
        record["Action"] = value
        cleaned_records.append(record)

    return "\n\n".join(
        f"Action {index}: {record['Action']}"
        + (f"\nOwner: {record['Owner']}" if record.get("Owner") else "")
        + (f"\nPriority: {record['Priority']}" if record.get("Priority") else "")
        for index, record in enumerate(cleaned_records, start=1)
    )


def _format_risk_records(*sources):
    records = _extract_risk_records(*sources)
    rendered = []
    for index, record in enumerate(records, start=1):
        lines = [f"Risk {index}: {record['Risk']}"]
        severity_impact = record.get("Severity/Impact") or " / ".join(
            record[field] for field in ("Severity", "Impact") if record.get(field)
        )
        if severity_impact:
            lines.append(f"Severity/Impact: {severity_impact}")
        if record.get("Probability"):
            lines.append(f"Probability: {record['Probability']}")
        lines.append(f"Owner: {record.get('Owner', 'Not specified')}")
        if record.get("Status"):
            lines.append(f"Status: {record['Status']}")
        if record.get("Description"):
            lines.append(f"Description: {record['Description']}")
        action = record.get("Mitigation") or record.get("Recommended Action") or record.get("Required Action")
        if action:
            lines.append(f"Mitigation / Required Action: {action}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _format_project_summary(content):
    labels = (
        ("Project Name", ("Project Name", "Project Title")),
        ("Project Objective", ("Project Objective", "Objective")),
        ("Current Sprint / Status", ("Current Sprint / Status", "Current Sprint", "Current Status", "Project Status")),
        ("Completed Work", ("Completed Work", "Completed")),
        ("Work In Progress", ("Work In Progress", "In Progress", "Ongoing Work")),
        ("Blocked / Pending Work", ("Blocked / Pending Work", "Blocked Work", "Pending Work", "Blocked / Pending")),
        ("Key Risks", ("Key Risks", "Risks")),
        ("Next Priorities", ("Next Priorities", "Next Priority", "Priorities")),
    )
    aliases = {alias.casefold(): label for label, names in labels for alias in names}
    values = {}
    current_label = None
    for raw_line in str(content or "").replace("\\n", "\n").splitlines():
        line = raw_line.strip().replace("**", "").replace("`", "")
        if not line or line.startswith(("#", "=====")):
            continue
        match = re.match(r"^([^:：]+)\s*[:：]\s*(.*)$", line)
        if "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                table_label = aliases.get(cells[0].casefold())
                if table_label and cells[1] and not cells[1].casefold().startswith(("not found", "not specified", "---")):
                    values.setdefault(table_label, []).append(cells[1])
                    current_label = table_label
            continue
        if match and match.group(1).strip().casefold() in aliases:
            current_label = aliases[match.group(1).strip().casefold()]
            value = match.group(2).strip(" -*•")
            if value and not value.casefold().startswith(("not found", "not specified", "n/a")):
                values.setdefault(current_label, []).append(value)
            continue
        item = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if current_label and item and not item.casefold().startswith(("not found", "not specified", "n/a")):
            values.setdefault(current_label, []).append(item)
    return "\n".join(
        f"{label}:\n" + "\n".join(f"- {value}" for value in dict.fromkeys(values[label]))
        for label, _ in labels if values.get(label)
    )


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

    document_context = _build_document_context(retrieved_documents)
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
    section_contexts = {
        "USER STORIES": _build_context(retrieved_documents, scope_analysis=scope_analysis),
        "RISK REGISTER": _build_context(retrieved_documents, risk_analysis=risk_analysis),
        "ACTION ITEMS": _build_context(
            retrieved_documents,
            risk_analysis=risk_analysis,
            blocker_analysis=blocker_analysis,
        ),
        "PROJECT SUMMARY": _build_context(
            retrieved_documents,
            scope_analysis=scope_analysis,
            risk_analysis=risk_analysis,
            blocker_analysis=blocker_analysis,
        ),
    }

    section_prompts = [
        (
            "USER STORIES",
            """
You are generating only the User Stories section of project documentation.
Use only explicitly documented capabilities. Do not create a story for an API, UI, or integration separately when it is part of a documented parent capability. Use this format for each supported capability:
US-01
User Story: As a project user, I want to use [documented capability] functionality.
Related Module: [documented capability]
Do not invent a more specific role or benefit. Avoid duplicates. If no capability is documented, return exactly:
Not found in the provided project information.

Return only the story records, with no heading or other section.
"""
        ),
        (
            "RISK REGISTER",
            """
You are generating only the Risk Register section of project documentation.
Use the retrieved project documents and the Risk Analysis only. Do not treat Blocker Analysis, Health Analysis, Scope Analysis, status tables, or markdown table rows as separate risks.
Treat documented unresolved dependencies or specifications, delays, resource limitations, blockers, and other explicit uncertainties as risk evidence even if they are not labeled "risk".
Do not require the source to contain a section labelled "Risk Register". Include each distinct risk supported by the context, describing its documented uncertainty or blocker. Add an impact, likelihood, mitigation, owner, date, or status only when explicitly supported; omit unavailable fields instead of rejecting the risk or writing a missing-information placeholder. If no risk evidence is present, return exactly:
Not found in the provided project information.
Never create risks from a health score/dimension, a task/status row, a blocker heading, a table header, or an isolated word such as "Pending" or "Blocked".

For each risk provide a risk name and only supported fields: Severity/Impact, Probability, Owner, Status, Description, and Mitigation / Required Action. If no owner is documented, write "Not specified". Omit other unavailable fields.
Return only concise risk records, with no heading or other section.
"""
        ),
        (
            "ACTION ITEMS",
            """
You are generating only the Action Items section of project documentation.
Use the retrieved project documents, Risk Analysis, and Blocker Analysis only. Do not copy table headers, section headings, health scores, task rows, or markdown fragments.
Extract documented action statements and actionable next steps supported by blockers, dependencies, unfinished integrations, or testing constraints. Where the context documents unfinished work or a dependency, state the direct step needed to resolve or complete it; do not require the source to phrase that step as an imperative.
Avoid duplicate or repetitive actions and generic project-management actions.
Do not invent actions, owners, priorities, due dates, or statuses. Omit unavailable metadata rather than adding a missing-information placeholder. Do not require the source to use a particular action-list heading.
If no documented or clearly required action is supported by the context, return exactly:
Not found in the provided project information.

Return concise actions as separate records. Include Owner or Priority only when the context provides them. Do not include a separate numbering-only line.
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

Return only these fields when supported, with no heading or other section. Add Next Priorities only if explicit next priorities are supported by the context:
Project Name:
Project Objective:
Current Sprint / Status:
Completed Work:
Work In Progress:
Blocked / Pending Work:
Key Risks:
Next Priorities:
"""
        ),
    ]

    documentation_parts = []
    successful_sections = 0

    for section_number, (section_name, prompt) in enumerate(section_prompts, start=1):

        section_content = _generate_section(
            prompt,
            section_contexts[section_name]
        )
        had_section_content = bool(section_content)

        if section_content:
            successful_sections += 1
            section_content = "\n".join(
                line
                for line in section_content.splitlines()
                if not line.lstrip().startswith("#")
            ).strip()

        if section_name == "USER STORIES":
            section_content = _fallback_user_stories(section_contexts[section_name]) or missing_information
        elif section_name == "RISK REGISTER":
            section_content = _format_risk_records(
                risk_analysis,
                section_content,
            ) or _fallback_risk_register("", risk_analysis) or missing_information
        elif section_name == "ACTION ITEMS":
            cleaned_actions = _clean_action_items(section_content)
            if not cleaned_actions:
                cleaned_actions = _clean_action_items(
                    _fallback_action_items(
                        section_contexts[section_name],
                        blocker_analysis,
                        risk_analysis,
                    )
                )
            section_content = cleaned_actions or missing_information
        elif section_name == "PROJECT SUMMARY":
            section_content = _format_project_summary(section_content) or section_content.strip()

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
