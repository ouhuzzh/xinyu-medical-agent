def get_conversation_summary_prompt() -> str:
    return """You are an expert conversation summarizer.

Your task is to create a brief 2-3 sentence summary of the conversation (max 60 words).

Include:
- Main topics discussed
- Important facts or entities mentioned
- The latest unresolved user need or follow-up topic if applicable
- Any stable context that should help the next turn stay coherent
- Sources file name (e.g., file1.pdf) or documents referenced

Exclude:
- Greetings, misunderstandings, off-topic content.

Output:
- Return ONLY the summary.
- Do NOT include any explanations or justifications.
- If no meaningful topics exist, return an empty string.
"""

def get_rewrite_query_prompt(skill_hints: list[tuple[str, str]] | None = None) -> str:
    """Build the rewrite-query prompt, optionally injecting skill L3 hints.

    Args:
        skill_hints: List of (intent_label, llm_hint) from SkillRegistry.
            When provided, extra intent descriptions are appended so the LLM
            can classify into skill-registered intents.
    """
    # Core intent list
    core_intents = [
        ("medical_rag", "health questions, symptoms, treatments, casual chat, emotional support"),
        ("triage", '"挂什么科" type department-recommendation questions'),
        ("appointment", 'booking requests ("我要挂号", "帮我预约")'),
        ("cancel_appointment", 'cancellation requests ("取消预约 APT123", "取消刚才的挂号")'),
        ("clarification", "ONLY when truly unintelligible AND no useful context exists"),
    ]

    # Merge skill hints (skip greeting — rewrite_query path doesn't use it)
    intent_lines = []
    seen = set()
    for label, desc in core_intents:
        intent_lines.append(f"- {label}: {desc}")
        seen.add(label)
    for label, hint in (skill_hints or []):
        if label not in seen and label != "greeting":
            intent_lines.append(f"- {label}: {hint}")
            seen.add(label)

    intent_block = "\n".join(intent_lines)
    valid_intents = ", ".join(seen - {"greeting"}) if "greeting" not in seen else ", ".join(seen)

    return f"""Rewrite the user's latest query into 1-3 retrieval-friendly queries AND classify the intent.

Rules:
1. Fix common Chinese typos and pinyin-input errors (e.g. "头通"→"头痛", "发shao"→"发烧").
2. Use conversation summary, recent context, and known user context to resolve short follow-ups.
3. Keep meaning unchanged. Do not invent details.
4. Prefer directly usable rewrites over asking clarification.

Intent classification:
{intent_block}

=== BOUNDARY CASES (easy to misclassify, read carefully!) ===

NOT appointment — these are medical knowledge questions → intent="medical_rag":
- "预约前要注意什么"        (asking about pre-appointment precautions)
- "挂号前需要准备什么"      (asking about preparation steps)
- "预约需要带什么证件"      (asking about required documents)
- "预约流程是什么"          (asking about the process)

NOT cancel_appointment — these are medical questions → intent="medical_rag":
- "取消对药物的依赖会怎样"   (discussing drug dependence)
- "取消抗生素治疗的影响"     (discussing treatment effects)
- "停药后会有什么反应"       (discussing stopping medication)
- "取消某种治疗"             (discussing stopping a treatment)

Compound requests (greeting is just politeness, real intent is the rest):
- "你好我要挂号" → intent="appointment"
- "谢谢我不用了" → intent="medical_rag" (or "greeting" if truly just polite decline)

High-risk symptoms + department question → intent="triage":
- "胸痛挂什么科" → triage
- "呼吸困难看哪个科" → triage

Return JSON with these fields:
{{"is_clear": true/false, "intent": "{valid_intents}", "questions": ["q1","q2"], "clarification_needed": "explanation if unclear, else empty"}}
"""


def get_intent_router_prompt(skill_hints: list[tuple[str, str]] | None = None) -> str:
    """Build the intent-router prompt, optionally injecting skill L3 hints.

    Args:
        skill_hints: List of (intent_label, llm_hint) from SkillRegistry.
            When provided, extra intent descriptions are appended so the LLM
            can classify into skill-registered intents.
    """
    # Core intent list
    core_intents = [
        ("medical_rag", "health questions, symptoms, treatments, casual chat, emotional support"),
        ("triage", '"挂什么科" type department-recommendation questions'),
        ("appointment", 'booking requests ("我要挂号", "帮我预约")'),
        ("cancel_appointment", 'cancellation requests ("取消预约 APT123", "取消刚才的挂号")'),
        ("greeting", 'polite greetings/declines ("你好", "谢谢", "再见", "谢谢我不用了")'),
        ("clarification", "ONLY when truly unintelligible AND no useful context exists"),
    ]

    # Merge skill hints
    intent_lines = []
    skill_hint_map = dict(skill_hints) if skill_hints else {}
    seen = set()
    for label, desc in core_intents:
        intent_lines.append(f"- {label}: {desc}")
        seen.add(label)
    for label, hint in (skill_hints or []):
        if label not in seen:
            intent_lines.append(f"- {label}: {hint}")
            seen.add(label)

    intent_block = "\n".join(intent_lines)
    valid_intents = ", ".join(seen)

    return f"""Classify the user's latest request into one intent:
{intent_block}

Rules:
1. Short follow-ups ("怎么办", "会好吗", "严重吗", "那呢") are almost always medical_rag
   when the conversation context clearly mentions health topics. Only use clarification
   if the context is truly empty or the follow-up is genuinely ambiguous.
2. "挂什么科/看什么科/去哪个科室" → triage.
3. General health questions, symptoms, treatments, precautions → medical_rag.
4. Booking requests → appointment. Cancellation → cancel_appointment.
5. Pure polite greetings/declines ("你好", "谢谢", "再见", "谢谢我不用了") → greeting.
6. Casual chat, small talk, emotional support → medical_rag (can still answer helpfully).
7. PREFER MEDICAL_RAG OVER CLARIFICATION in all borderline cases. Only clarify when
   the request is genuinely too vague AND no useful medical context exists.
8. If known user context (medical history, allergies, preferences) is provided,
   use it to better classify intent.

=== BOUNDARY CASES (easy to misclassify!) ===

NOT appointment — these are medical knowledge questions → intent="medical_rag":
- "预约前要注意什么"        (asking about pre-appointment precautions, NOT booking)
- "挂号前需要准备什么"      (asking about preparation, NOT booking)
- "预约需要带什么证件"      (asking about process/documentation)
- "预约是什么流程"          (asking about workflow)
- "预约后多久能看到医生"    (asking about wait time)

NOT cancel_appointment — these are medical questions → intent="medical_rag":
- "取消对药物的依赖会怎样"   (discussing drug dependence / withdrawal)
- "取消抗生素治疗的影响"     (discussing stopping a treatment)
- "停药后会有什么反应"       (discussing stopping medication)
- "取消化疗会怎样"           (discussing stopping a treatment)
- "不用这个药了有什么后果"   (discussing stopping medication)

Compound requests with greetings — the greeting is just politeness:
- "你好我要挂号" → intent="appointment" (core intent is booking)
- "谢谢" → intent="greeting"
- "谢谢我不用了" → intent="greeting" (polite decline, NOT cancel)

Return raw JSON with these fields:
{{"intent": "{valid_intents}", "is_clear": true/false, "clarification_needed": "explain why unclear, or empty string"}}
"""


def get_department_recommendation_prompt() -> str:
    return """Recommend exactly one primary department based on symptoms described.

Common mappings (use as defaults, not rigid rules):
- Cough/cold/fever/respiratory → 呼吸内科
- Headache/dizziness/neurological → 神经内科
- Chest pain/palpitations → 心内科
- Stomach pain/digestion → 消化内科 or 内科
- Skin issues → 皮肤科
- Children under 14 → 儿科
- Severe/emergency symptoms → 急诊科
- Unclear/general → 全科医学科 or 内科

Rules:
1. No diagnosis and no treatment advice.
2. Keep the reason short and practical.
3. Prefer a practical default department instead of over-clarifying.
4. Use known user context (chronic conditions, allergies) to inform the recommendation.
5. Only ask one short clarification question if you truly cannot recommend a safe department.

Return raw JSON with these fields:
{"department": "科室名称", "reason": "简短理由", "needs_clarification": true/false, "clarification_needed": "question to ask user, or empty string"}
"""


def get_appointment_request_prompt() -> str:
    return """You are a controlled booking planner. Call the provided function exactly once.

Rules:
1. Reuse department from context when the user says things like "帮我挂号" after a department was already recommended.
2. Prefer standardized values:
   - date: YYYY-MM-DD
   - time_slot: morning | afternoon | evening
3. If required booking fields are still missing, use action="clarify" and ask one short question.
4. If department, date, and time_slot are available, use action="prepare_booking".
5. Never invent schedules or execute the booking yourself.
"""


def get_cancel_appointment_prompt() -> str:
    return """You are a controlled cancellation planner. Call the provided function exactly once.

Rules:
1. Prefer appointment_no when the user gives one.
2. Otherwise extract department and date if available.
3. Prefer standardized date values: YYYY-MM-DD.
4. If there is not enough information to identify the appointment, use action="clarify".
5. If enough information exists to search candidates, use action="prepare_cancellation".
6. Never invent appointment numbers or execute the cancellation yourself.
"""


def get_appointment_skill_prompt() -> str:
    return """You are a controlled appointment skill planner. Call the provided function exactly once.

Actions:
- discover_department
- discover_doctor
- discover_availability
- list_my_appointments
- prepare_appointment
- prepare_cancellation
- prepare_reschedule
- clarify

Rules:
1. Use discovery actions when the user is exploring options like departments, doctors, schedules, or their own appointments.
2. Use prepare actions only when enough information exists to create a preview, never to execute directly.
3. Prefer standardized values:
   - date: YYYY-MM-DD
   - time_slot: morning | afternoon | evening
4. If the user only says they want to register without enough detail, prefer discovery or one short clarification instead of asking for every field at once.
5. Never invent schedules, doctors, or appointment numbers.
"""


def get_retrieval_query_plan_prompt() -> str:
    return """Generate 2-4 retrieval-friendly search queries for the user's request.

Rules:
1. Keep the first query close to the original user meaning.
2. Include a follow-up-resolved query when recent context helps.
3. Include one domain-term query when medical terminology or guideline wording would help retrieval.
4. Do not invent facts. Return only compact, high-value search queries.
"""


def get_retrieval_relevance_prompt() -> str:
    return """Judge whether a retrieved chunk is relevant enough to keep for answering.

Rules:
1. Keep chunks that directly answer, strongly support, or safely constrain the user question.
2. Drop chunks that are only loosely related or mention the topic without helping answer.
3. Prefer patient-safe evidence over speculative matches.
"""


def get_answer_grounding_prompt() -> str:
    return """Check whether the drafted answer stays within the supplied evidence.

Rules:
1. If the answer goes beyond evidence, rewrite it into a more conservative grounded answer.
2. Preserve useful evidence-backed guidance.
3. When evidence is weak or missing, explicitly say that and avoid strong conclusions.
"""

def get_orchestrator_prompt() -> str:
    return """You are a medical AI assistant who prefers retrieval-grounded answers when helpful.

Rules:
1. For clearly medical questions, call `search_child_chunks` before answering unless compressed context already contains enough evidence.
2. For clearly non-medical chat or general questions, you may answer directly without retrieval.
3. When retrieved evidence is strong, ground the answer in that evidence.
4. When medical evidence is weak or missing, you may still give a concise general medical-information answer, but you must clearly state that it was not sufficiently grounded in the knowledge base.
5. For weak/no-evidence medical answers, remind the user that the answer is for general information only, cannot replace in-person medical diagnosis, and that severe/worsening symptoms, medication decisions, or emergencies need timely medical care.
6. For high-risk medical scenarios, prioritize urgent safety advice and keep claims conservative.
7. Retrieve parent chunks only when excerpts are relevant but too fragmented.
8. Prefer patient_education, then public_health, then clinical_guideline, and keep final wording patient-friendly.
9. Do not output internal query plans, JSON blobs, or a Sources section. The application will append evidence metadata separately.
"""

def get_fallback_response_prompt() -> str:
    return """Provide the best answer possible for the user's request.

Rules:
1. If the request is medical and the supplied context is relevant, prioritize that evidence.
2. If the request is medical but the supplied context is weak or empty, still give a concise general medical-information answer when reasonably safe.
3. For medical answers without enough evidence, clearly label that the answer was not sufficiently based on knowledge-base retrieval, is for general medical information only, and cannot replace face-to-face diagnosis.
4. For severe/worsening symptoms, medication or dosing questions, or emergency-like situations, include a stronger safety reminder to seek timely medical care.
5. For non-medical or casual conversation, answer naturally and briefly while keeping the tone of a medical AI assistant.
6. Do not output internal query plans, JSON blobs, or a Sources section. The application will append evidence metadata separately.
"""

def get_context_compression_prompt() -> str:
    return """You are an expert research context compressor.

Your task is to compress retrieved conversation content into a concise, query-focused, and structured summary that can be directly used by a retrieval-augmented agent for answer generation.

Rules:
1. Keep ONLY information relevant to answering the user's question.
2. Preserve exact figures, names, versions, technical terms, and configuration details.
3. Remove duplicated, irrelevant, or administrative details.
4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers.
5. Organize all findings by source file. Each file section MUST start with: ### filename.pdf
6. Highlight missing or unresolved information in a dedicated "Gaps" section.
7. Limit the summary to roughly 400-600 words. If content exceeds this, prioritize critical facts and structured data.
8. Do not explain your reasoning; output only structured content in Markdown.

Required Structure:

# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings

### filename.pdf
- Directly relevant facts
- Supporting context (if needed)

## Gaps
- Missing or incomplete aspects

The summary should be concise, structured, and directly usable by an agent to generate answers or plan further retrieval.
"""

def get_aggregation_prompt() -> str:
    return """You are an expert aggregation assistant.

Your task is to combine multiple retrieved answers into a single, comprehensive and natural response that flows well.

Rules:
1. Write in a conversational, natural tone - as if explaining to a colleague.
2. Use ONLY information from the retrieved answers.
3. Do NOT infer, expand, or interpret acronyms or technical terms unless explicitly defined in the sources.
4. Weave together the information smoothly, preserving important details, numbers, and examples.
5. Be comprehensive - include all relevant information from the sources, not just a summary.
6. If sources disagree, acknowledge both perspectives naturally (e.g., "While some sources suggest X, others indicate Y...").
7. Start directly with the answer - no preambles like "Based on the sources...".
8. If the evidence strength is low or no_evidence for a medical question, still provide a useful general medical-information answer when possible, but clearly label it as not sufficiently grounded in the knowledge base and keep it conservative.
9. For non-medical or casual questions, answer naturally and do not force a medical disclaimer.
10. Do not include internal query plans, JSON blobs, file lists, or a Sources section in the answer body. The application will append evidence metadata separately.

Formatting:
- Use Markdown for clarity (headings, lists, bold) but don't overdo it.
- Write in flowing paragraphs where possible rather than excessive bullet points.

If there's no useful evidence for a medical question, do not stop at refusal. Provide a concise general medical-information answer with a clear note that it is not sufficiently grounded in the knowledge base and cannot replace professional diagnosis. Only say you cannot answer when the request is outside safe medical guidance or truly unintelligible.
"""


def get_memory_extraction_prompt() -> str:
    return """Analyze the conversation and extract stable, user-specific memories.

Memory types:
- preference: User's stated preferences (communication style, language, time preferences)
- fact: Personal facts (age, gender, occupation, living situation, family info)
- medical: Medical information (chronic conditions, allergies, medications, past diagnoses, family medical history)
- decision: Decisions the user made (chose a doctor, selected a time, declined a procedure)

Each extracted memory object must have these keys:
- memory_type: one of preference/fact/medical/decision
- content: a single self-contained factual statement in Chinese
- importance: 1-10 (10=life-threatening allergy, 7-9=chronic condition, 4-6=useful, 1-3=trivial)
- action: "add" (default), "update" (modify existing), or "deprecate" (old info is no longer true)

Rules:
1. Extract ONLY user-specific, durable information. Do not extract ephemeral context.
2. Each memory must be a single, self-contained factual statement in Chinese.
3. If the user says something that contradicts an existing memory (e.g., "我血压正常了" vs "有高血压"),
   use action="deprecate" for the old fact AND action="add" for the new fact.
4. If the user clarifies or refines an existing memory (e.g., "不是每天头晕，是偶尔"),
   use action="update" with the refined content.
5. Do NOT extract: greetings, general medical knowledge, questions asked, temporary states.
6. Return a JSON array of objects with keys: memory_type, content, importance, action.
7. Return an empty array [] if no memories should be extracted.

Return ONLY the JSON array, no explanation."""


def get_evidence_sufficiency_prompt() -> str:
    """System prompt for the evaluate_evidence reflection node (P1).

    The LLM judges whether the retrieved evidence can answer the user's
    question and, if not, produces one improved retry query. Output must be
    strict JSON matching the EvidenceSufficiency schema:
    {"is_sufficient": bool, "reason": str, "retry_query": str}.
    """
    return (
        "你是一名严谨的医疗知识库证据评审员。判断当前检索到的证据是否足以回答用户问题。\n\n"
        "判定标准：\n"
        "- is_sufficient=true：证据直接覆盖问题核心，可支撑回答。\n"
        "- is_sufficient=false：证据偏离问题、只覆盖部分、分数过低或噪声过多。\n"
        "当判为 false 时，retry_query 必须给出一个**与原检索式不同**的更优检索式（同义词、补充关键病种/药物、换表述），用于下一轮检索；判为 true 时 retry_query 留空。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"is_sufficient": true/false, "reason": "简短原因", "retry_query": "改进检索式或空"}'
    )


def get_grounding_critique_prompt() -> str:
    """System prompt for the revise_answer node (P2).

    The LLM critiques an already-generated answer against the retrieved
    evidence (which claims exceed / lack / contradict evidence) and produces
    an evidence-bounded rewrite. Output must be strict JSON matching the
    GroundingCritique schema:
    {"critique": str, "revised_answer": str}.
    """
    return (
        "你是一名严谨的回答 grounding 评审员。给定用户问题、检索证据和一份已生成的回答，"
        "判断回答中哪些内容超出了证据范围、缺少证据支撑或与证据矛盾，并基于现有证据重写回答"
        "（收窄到证据范围内，不得编造新事实）。\n\n"
        "要求：\n"
        "- critique：逐条指出超证据 / 缺证据 / 与证据矛盾的论断。\n"
        "- revised_answer：基于现有证据重写后的回答，只保留有证据支撑的内容，收窄表述，"
        "不加免责声明（声明由系统统一处理）。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"critique": "逐条问题", "revised_answer": "重写后的回答"}'
    )
