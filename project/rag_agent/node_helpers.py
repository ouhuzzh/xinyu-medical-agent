from __future__ import annotations
"""Shared LangGraph node helpers.

Utility functions and constants used by two or more of the focused node
modules (routing_nodes, rag_nodes, appointment_nodes).  New code should
import from this module instead of reaching into a specific node module
for cross-cutting helpers.
"""

import re
import logging
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from .graph_state import State
import config
from config import HIGH_RISK_KEYWORDS


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns shared across node modules
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(\d{1,2})[:：点时]")
_YEAR_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_SLASH_DATE_RE = re.compile(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})")
_WEEKDAY_RE = re.compile(r"(下|这|本)?\s*周([一二三四五六日天])")
_ORDINAL_RE = re.compile(r"第\s*([1-9]\d*)\s*(个|条)?")
_APPOINTMENT_NO_RE = re.compile(r"\bAPT[A-Z0-9]+\b", re.IGNORECASE)
_CN_HOUR_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16,
    "十七": 17, "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22, "二十三": 23,
}
_CN_HOUR_RE = re.compile(r"(二十三|二十二|二十一|二十|十九|十八|十七|十六|十五|十四|十三|十二|十一|十|[一二三四五六七八九])")


# ---------------------------------------------------------------------------
# Keyword / hint tuples shared across node modules
# ---------------------------------------------------------------------------

_APPOINTMENT_CONFIRM_WORDS = (
    "确认预约", "确认挂号", "确认就诊", "确认预订", "请预约", "帮我预约", "现在预约", "立即预约", "确认",
)
_CANCEL_CONFIRM_WORDS = (
    "确认取消", "确认退号", "确定取消", "现在取消", "立即取消", "确认",
)
_ABORT_WORDS = (
    "先不用", "先不", "不用了", "算了", "取消这个操作", "放弃", "暂不", "不预约了", "不取消了",
)
_MEDICAL_FOLLOW_UP_HINTS = (
    "那", "这个", "这种情况", "这会", "还会", "严重吗", "怎么办", "注意什么", "要紧吗",
    "what about", "does that", "is that", "should i", "what should",
)
_MEDICAL_TERMS = (
    "高血压", "糖尿病", "感冒", "发烧", "发热", "低烧", "高烧", "头疼", "头痛", "偏头痛",
    "头晕", "眩晕", "咳嗽", "咳痰", "咽痛", "嗓子疼", "喉咙痛", "流鼻涕", "鼻塞",
    "腹痛", "肚子疼", "胃痛", "腹泻", "拉肚子", "便秘", "恶心", "呕吐", "反酸",
    "胸闷", "胸痛", "心悸", "心慌", "乏力", "疲劳", "呼吸困难", "气短", "喘",
    "肺炎", "哮喘", "鼻炎", "胃炎", "肠胃炎", "失眠", "焦虑", "抑郁", "皮疹", "过敏",
    "血压", "血糖", "检查", "药", "症状", "疾病", "炎", "癌", "病", "疫苗", "预防", "指南", "综合征",
    "hypertension", "diabetes", "fever", "cough", "dizziness", "headache", "pneumonia",
    "asthma", "symptom", "treatment", "disease", "medicine", "blood pressure",
)
_MEDICAL_QUESTION_PATTERNS = (
    "是什么", "怎么回事", "为什么", "原因", "症状", "表现", "怎么办", "如何", "怎么处理",
    "怎么缓解", "严重吗", "会不会", "会引起", "会导致", "能不能", "可以吗", "要不要",
    "是否", "注意事项", "治疗", "预防", "要紧吗", "还要看吗", "哪些人", "什么人", "什么时候", "是不是",
    "几片", "几粒", "几次", "剂量", "怎么吃", "怎么服用", "一天吃", "一天用", "多久吃一次", "多久用一次",
    "means", "what is", "why", "how to", "symptoms",
    "treatment", "can ", "could ", "does ", "is it",
)
_APPOINTMENT_KEYWORDS = ("挂号", "预约", "book appointment", "register")
_CANCEL_KEYWORDS = ("取消", "退号", "cancel appointment", "cancel booking")
_EXPLICIT_APPOINTMENT_CUES = ("帮我", "给我", "我要", "想", "安排", "预约一下", "挂一下", "register me", "book me")
_EXPLICIT_CANCEL_CUES = ("取消预约", "取消挂号", "退号", "帮我取消", "取消刚才", "cancel", "取消那个")
_GENERAL_CHAT_HINTS = (
    "我今天有点烦", "有点烦", "心情不好", "不开心", "有点累", "有点焦虑", "想聊聊", "聊聊",
    "谢谢你", "谢谢", "晚安", "早安", "中午好", "晚上好",
)
_NON_MEDICAL_TOPIC_HINTS = (
    "东京", "旅游", "景点", "好玩", "美食", "电影", "书", "天气", "旅行", "推荐一下", "介绍一下",
    "有什么好玩的", "周末去哪", "想放松", "可以聊聊天吗",
)
_MEDICATION_RISK_HINTS = (
    "一天吃几片", "一次吃几片", "一天几次", "一次几次", "剂量", "毫克", "mg", "用量", "服用",
    "怎么吃", "多久吃一次", "多久用一次", "bid", "tid", "qd",
)
_DEPARTMENT_HINTS = (
    "呼吸内科", "心内科", "神经内科", "消化内科", "内分泌科", "急诊科", "全科", "儿科",
    "妇科", "骨科", "皮肤科", "耳鼻喉科", "眼科", "呼吸科", "内科", "外科", "门诊",
)
_TOPIC_STOP_WORDS = ("一下", "一下子", "这个", "那个", "这种情况", "怎么", "怎么办", "需要", "是否", "一般")
_APPOINTMENT_LIST_HINTS = (
    "我的预约", "有哪些预约", "查预约", "看看预约", "预约列表",
    "挂了谁的号", "挂的是谁的号", "我挂了谁的号", "我之前挂了谁的号",
    "我现在挂了谁的号", "预约了谁", "约了谁的号",
)
_DOCTOR_DISCOVERY_HINTS = ("有哪些医生", "哪个医生", "医生有号", "谁有号", "查医生", "医生排班", "专家", "号源")
_ANY_DOCTOR_HINTS = ("任一", "任何医生", "随便医生", "都可以", "任选", "任意医生", "任一可用医生")
_EARLIEST_SLOT_HINTS = ("最早可用时段", "最早的", "最早号源", "最快的", "尽快")
_RESCHEDULE_HINTS = ("改约", "改到", "换到", "换成", "改成", "挪到")


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _structured_output_llm(llm, schema, *, temperature: float = 0.1, max_tokens: int | None = None):
    """Return an object with .invoke() that calls LLM and parses JSON.

    Avoids LangChain's with_structured_output (uses response_format: json_object
    which SiliconFlow/Qwen doesn't support — long retry loop).
    Instead: get raw text, extract JSON via regex.
    """
    import json as _json, re as _re

    base = llm.with_config(temperature=temperature)
    base = base.bind(max_tokens=max_tokens or 256)

    class _StructureParser:
        """Thin wrapper: __call__ + .invoke()."""

        def __call__(self, messages: list):
            try:
                raw = str(base.invoke(messages).content or "").strip()
            except Exception:
                logger.warning("_StructureParser: LLM invoke failed for schema %s, returning default", schema.__name__)
                return _default()
            if not raw:
                logger.debug("_StructureParser: empty LLM response for schema %s", schema.__name__)
                return _default()
            # Try JSON patterns
            for pattern in [r"```(?:json)?\s*\n?(.*?)```", r"(\{.*\})"]:
                m = _re.search(pattern, raw, _re.DOTALL)
                if m:
                    try:
                        return schema(**_json.loads(m.group(1)))
                    except Exception:
                        logger.debug("_StructureParser: JSON pattern %s matched but schema parse failed for %s", pattern[:20], schema.__name__)
            try:
                return schema(**_json.loads(raw))
            except Exception:
                logger.debug("_StructureParser: raw JSON parse failed for schema %s", schema.__name__)
            logger.warning("_StructureParser: all parse attempts failed for schema %s, returning default", schema.__name__)
            return _default()

        def invoke(self, messages: list):
            return self(messages)

    def _default():
        """Return a safe default based on the schema.

        Order matters: check container types (list) before scalar str, because
        a `List[str]` annotation string contains "str" and would otherwise
        default to "" — which Pydantic rejects for list fields, raising
        ValidationError and breaking the never-raise invariant.
        """
        vals = {}
        for fn, fi in schema.model_fields.items():
            t = str(fi.annotation).lower()
            if "list" in t:
                vals[fn] = []
            elif "bool" in t:
                vals[fn] = False
            elif "int" in t:
                vals[fn] = 0
            elif "str" in t:
                vals[fn] = ""
            else:
                vals[fn] = ""
        return schema(**vals)

    return _StructureParser()


def _clear_pending_action_state() -> dict:
    return {
        "pending_action_type": "",
        "pending_action_payload": {},
        "pending_confirmation_id": "",
        "pending_candidates": [],
    }


def _reset_pending_action_if_needed(state: State) -> dict:
    if not state.get("pending_action_type") and not state.get("pending_candidates"):
        return {}
    return _clear_pending_action_state()


def _build_appointment_context(existing: dict | None, updates: dict) -> dict:
    context = dict(existing or {})
    for key, value in updates.items():
        if value or (key in updates and isinstance(value, list)):
            context[key] = _json_safe_value(value)
    return context


def _json_safe_value(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _sanitize_pending_payload(payload: dict | None) -> dict:
    cleaned = dict(payload or {})
    for key in ("department", "date", "time_slot", "doctor_name", "appointment_no", "action", "hospital_code", "hospital_name"):
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = value.strip()
    return cleaned


# L1 strict-greeting word list — only very short, unambiguous greetings
_PURE_GREETINGS = {
    "你好", "您好", "hello", "hi", "hey", "嗨", "哈喽",
    "早上好", "下午好", "晚上好", "good morning", "good afternoon", "good evening",
    "谢谢", "感谢", "thanks", "thank you", "thx", "多谢",
    "再见", "拜拜", "bye", "goodbye", "晚安", "早安",
}


def _looks_like_greeting(query: str) -> bool:
    """L1-strict: only match short, pure greetings (<= 15 chars, exact match).

    Exact set match prevents substring issues — '你好我要挂号' is NOT a
    pure greeting even though it contains '你好'.  The 15-char ceiling
    accommodates common multi-word English greetings ('good morning').
    """
    normalized = (query or "").strip()
    if not normalized or len(normalized) > 15:
        return False
    normalized_lower = normalized.lower()
    return normalized_lower in _PURE_GREETINGS


def _starts_with_polite_decline(query: str) -> bool:
    """Check if query starts with polite refusal pattern like '谢谢我不用了'.

    These should NOT be treated as cancel/abort requests when a pending
    action exists — they are polite conversation-enders.
    """
    normalized = (query or "").strip()
    if not normalized:
        return False
    # Polite opening + refusal
    polite_opens = ("谢谢", "感谢", "thanks", "thank you", "多谢", "你好")
    refusal_words = ("不用", "不需要", "算了", "不了", "不用了", "先不", "暂时不")
    normalized_lower = normalized.lower()
    has_polite = any(normalized_lower.startswith(p) for p in polite_opens)
    has_refusal = any(r in normalized for r in refusal_words)
    return has_polite and has_refusal


def _looks_like_department_question(query: str) -> bool:
    """Check if the query looks like a department recommendation question."""
    normalized = (query or "").strip().lower()
    patterns = [
        "挂什么科",
        "挂哪个科",
        "看什么科",
        "看哪个科",
        "挂哪科",
        "看哪科",
        "咨询什么科",
        "consult which department",
        "which department",
        "what department should i visit",
        "what department should i register for",
    ]
    return any(pattern in normalized for pattern in patterns)


def _looks_like_medical_knowledge_question(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if not normalized or _looks_like_department_question(normalized):
        return False

    has_medical_term = any(term in normalized for term in _MEDICAL_TERMS)
    has_disease_suffix = bool(re.search(r"(综合征|感染|病|炎|癌|瘤)", normalized))
    has_question_pattern = any(pattern in normalized for pattern in _MEDICAL_QUESTION_PATTERNS) or normalized.endswith("?") or normalized.endswith("？") or normalized.endswith("吗")
    return (has_medical_term or has_disease_suffix) and has_question_pattern


def _looks_like_medication_risk_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    return any(token in normalized for token in _MEDICATION_RISK_HINTS)


def _context_has_medical_signal(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(term in normalized for term in _MEDICAL_TERMS)


def _looks_like_medical_request(query: str, *, conversation_summary: str = "", recent_context: str = "", topic_focus: str = "") -> bool:
    normalized = (query or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_department_question(query):
        return True
    if _looks_like_medical_knowledge_question(query) or _looks_like_medication_risk_query(query):
        return True
    if any(keyword.lower() in normalized for keyword in HIGH_RISK_KEYWORDS):
        return True
    if any(term in normalized for term in _MEDICAL_TERMS):
        return True
    context_text = "\n".join(part for part in (conversation_summary, recent_context, topic_focus) if str(part or "").strip())
    if any(token in normalized for token in _MEDICAL_FOLLOW_UP_HINTS) and _context_has_medical_signal(context_text):
        return True
    return False


def _looks_like_medical_follow_up(user_query: str, conversation_summary: str, recent_context: str = "") -> bool:
    normalized = (user_query or "").strip().lower()
    context_text = "\n".join(part for part in (conversation_summary, recent_context) if part and str(part).strip())
    if not normalized or not context_text.strip():
        return False
    if not any(token in normalized for token in _MEDICAL_FOLLOW_UP_HINTS):
        return False
    return _context_has_medical_signal(context_text)


def _looks_like_general_non_medical_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_greeting(query):
        return True
    if _looks_like_medical_request(query) or _looks_like_explicit_appointment_intent(query) or _looks_like_explicit_cancel_intent(query):
        return False
    if any(token in normalized for token in _GENERAL_CHAT_HINTS):
        return True
    if any(token in normalized for token in _NON_MEDICAL_TOPIC_HINTS):
        return True
    return False


def _needs_strict_medical_safety(query: str, risk_level: str = "normal") -> bool:
    return risk_level == "high" or _looks_like_medication_risk_query(query)


def _looks_like_explicit_appointment_intent(user_query: str) -> bool:
    """L1-strict: only match when the user CLEARLY wants to book an appointment.

    Requires BOTH an action cue AND a concrete entity (department / date / time).
    Pure '我要挂号' without entity does NOT match — goes to L2/L3 for safer handling.
    """
    normalized = (user_query or "").strip()
    if not normalized:
        return False

    # Must have an explicit booking action cue
    _STRICT_APPOINTMENT_CUES = (
        "帮我预约", "我要挂号", "我要预约", "帮我挂号",
        "帮我挂一个", "帮我挂个", "预约一下", "挂一下",
        "帮我安排", "给我挂", "给我预约",
    )
    has_action = any(cue in normalized for cue in _STRICT_APPOINTMENT_CUES)
    if not has_action:
        return False

    # Reject pre-appointment knowledge questions
    if re.search(r"(预约|挂号|挂)前", normalized):
        return False

    # Must also have a concrete entity: department, date, or time
    has_department = any(dep in normalized for dep in _DEPARTMENT_HINTS)
    has_date = bool(_normalize_date(user_query))
    has_time = bool(_normalize_time_slot(user_query))

    return has_department or has_date or has_time


def _looks_like_appointment_discovery_query(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_department_question(user_query):
        return False
    if any(token in normalized for token in _APPOINTMENT_LIST_HINTS):
        return True
    if any(token in normalized for token in _DOCTOR_DISCOVERY_HINTS):
        return True
    if "有哪些科室" in normalized or "什么科室" in normalized:
        return True
    if "有号吗" in normalized and ("医生" in normalized or any(dep in normalized for dep in _DEPARTMENT_HINTS)):
        return True
    return False


def _looks_like_explicit_cancel_intent(user_query: str) -> bool:
    """L1-strict: only match when user explicitly wants to cancel an appointment.

    Strategy: requires a cancel keyword ('取消'/'退号') PLUS a concrete
    appointment reference (number, '最近/刚才/那个').  This catches
    '取消我刚才的预约' while rejecting '取消对药物的依赖'.
    """
    normalized = (user_query or "").strip()
    if not normalized:
        return False

    # Must have a cancel keyword
    if not any(keyword in normalized for keyword in _CANCEL_KEYWORDS):
        return False

    # Must have a concrete appointment reference
    has_explicit_cue = any(cue in normalized for cue in ("取消预约", "取消挂号", "退号", "帮我取消"))
    has_reference = (
        has_explicit_cue
        or _should_use_last_appointment(user_query)
        or bool(_APPOINTMENT_NO_RE.search(user_query or ""))
    )
    if not has_reference:
        return False

    # Reject medical knowledge questions about stopping treatments
    # e.g. '取消对药物的依赖' has cancel keyword + 药物, but not appointment reference
    return True


def _normalize_time_slot(raw_value: str) -> str:
    normalized = (raw_value or "").strip().lower()
    if not normalized:
        return ""
    if normalized in {"morning", "afternoon", "evening"}:
        return normalized
    context_evening = ["晚上", "傍晚", "evening", "night", "晚间", "今晚"]
    context_afternoon = ["下午", "afternoon", "午后", "中午", "中午后"]
    context_morning = ["上午", "早上", "早晨", "morning", "清晨"]
    if any(token in normalized for token in context_evening):
        return "evening"
    if any(token in normalized for token in context_afternoon):
        return "afternoon"
    if any(token in normalized for token in context_morning):
        return "morning"

    hour_match = _TIME_RE.search(normalized)
    cn_hour_match = _CN_HOUR_RE.search(normalized)
    has_half = "半" in normalized or ":30" in normalized or "：30" in normalized
    hour = None
    if hour_match:
        try:
            hour = int(hour_match.group(1))
        except ValueError:
            pass
    elif cn_hour_match:
        hour = _CN_HOUR_MAP.get(cn_hour_match.group(1))
    if hour is not None:
        if hour >= 18 or (hour == 12 and has_half):
            return "evening"
        if hour >= 12:
            return "afternoon"
        return "morning"
    # 兜底: am/pm 标识
    if "am" in normalized:
        return "morning"
    if "pm" in normalized:
        return "afternoon"
    return ""


def _normalize_date(raw_value: str) -> str:
    normalized = (raw_value or "").strip().lower()
    if not normalized:
        return ""
    today = date.today()
    if "今天" in normalized or "today" in normalized:
        return today.isoformat()
    if "明天" in normalized or "tomorrow" in normalized:
        return (today + timedelta(days=1)).isoformat()
    if "后天" in normalized or "day after tomorrow" in normalized:
        return (today + timedelta(days=2)).isoformat()
    if "这个周末" in normalized or "本周末" in normalized:
        return (today + timedelta(days=(5 - today.weekday()) % 7)).isoformat()
    if "下周末" in normalized:
        return (today + timedelta(days=((5 - today.weekday()) % 7) + 7)).isoformat()

    if len(normalized) == 10 and normalized[4] == "-" and normalized[7] == "-":
        try:
            return date.fromisoformat(normalized).isoformat()
        except ValueError:
            return ""

    weekday_match = _WEEKDAY_RE.search(normalized)
    if weekday_match:
        prefix = weekday_match.group(1) or ""
        weekday_text = weekday_match.group(2)
        target_weekday = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}[weekday_text]
        delta = (target_weekday - today.weekday()) % 7
        if prefix == "下" or (not prefix and delta == 0 and "下" in normalized):
            delta += 7
        return (today + timedelta(days=delta)).isoformat()

    m = _YEAR_DATE_RE.search(normalized)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return ""

    m = _MONTH_DAY_RE.search(normalized)
    if m:
        try:
            candidate = date(today.year, int(m.group(1)), int(m.group(2)))
            if candidate < today:
                candidate = date(today.year + 1, int(m.group(1)), int(m.group(2)))
            return candidate.isoformat()
        except ValueError:
            return ""

    m = _SLASH_DATE_RE.search(normalized)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return ""
    return ""


def _extract_topic_focus(user_query: str, existing_topic: str = "", appointment_context: dict | None = None, recommended_department: str = "") -> str:
    normalized = (user_query or "").strip()
    for term in _MEDICAL_TERMS:
        if term in normalized.lower():
            return term
    for department in _DEPARTMENT_HINTS:
        if department in normalized:
            return department
    if recommended_department:
        return recommended_department
    appointment_context = appointment_context or {}
    for key in ("department", "doctor_name"):
        value = (appointment_context.get(key) or "").strip()
        if value:
            return value
    existing_topic = (existing_topic or "").strip()
    return existing_topic


def _build_recent_context(messages, keep_turns: int | None = None, *, exclude_latest_user: bool = True) -> str:
    if keep_turns is None:
        keep_turns = max(int(getattr(config, "RECENT_CONTEXT_TURNS", 3) or 3), 1)
    recent_messages = [
        msg for msg in (messages or [])
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]
    if exclude_latest_user and recent_messages and isinstance(recent_messages[-1], HumanMessage):
        recent_messages = recent_messages[:-1]
    recent_messages = recent_messages[-keep_turns * 2 :]
    lines = []
    for msg in recent_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _infer_risk_level(user_query: str, existing_risk: str = "normal") -> str:
    normalized = (user_query or "").strip().lower()
    if any(keyword.lower() in normalized for keyword in HIGH_RISK_KEYWORDS):
        return "high"
    return existing_risk or "normal"


def _is_explicit_confirmation(user_query: str, pending_action_type: str) -> bool:
    """Only match EXPLICIT confirmation phrases — NOT vague words like '好的'/'行'.

    The user must type the full intent-specific phrase (e.g. '确认预约', '确认取消').
    Generic acknowledgements ('好的', 'OK', '行', '可以') do NOT trigger execution.
    """
    normalized = (user_query or "").strip()
    if not normalized:
        return False
    if pending_action_type in {"appointment", "reschedule_appointment"}:
        return any(phrase in normalized for phrase in ("确认预约", "确认挂号"))
    if pending_action_type == "cancel_appointment":
        return any(phrase in normalized for phrase in ("确认取消", "确认退号", "确定取消"))
    if pending_action_type == "mcp_confirm":
        return any(phrase in normalized for phrase in ("确认预约", "确认", "确认提交", "好的"))
    return False


def _is_abort_request(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(word in normalized for word in _ABORT_WORDS)


def _pick_candidate_from_text(user_query: str, pending_candidates: list[dict]) -> dict | None:
    if not pending_candidates:
        return None

    match = _APPOINTMENT_NO_RE.search(user_query or "")
    if match:
        appointment_no = match.group(0).upper()
        for item in pending_candidates:
            if str(item.get("appointment_no", "")).upper() == appointment_no:
                return item

    ordinal = _ORDINAL_RE.search(user_query or "")
    if ordinal:
        index = int(ordinal.group(1)) - 1
        if 0 <= index < len(pending_candidates):
            return pending_candidates[index]
    return None


def _should_use_last_appointment(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    hints = ("最近的", "上次", "刚刚", "刚才", "上一个", "上一条", "那个预约", "这条预约")
    return any(token in normalized for token in hints)


def _strip_leading_query_plan_blob(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    patterns = [
        r'^\s*```(?:json)?\s*\{\s*"queries"\s*:\s*\[[\s\S]*?\]\s*\}\s*```\s*',
        r'^\s*\{\s*"queries"\s*:\s*\[[\s\S]*?\]\s*\}\s*',
    ]
    cleaned = raw
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if cleaned != raw:
            break
    return cleaned.strip() or raw


def _strip_trailing_sources_block(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    patterns = [
        r'\n\s*---\s*\n\s*\*\*Sources:\*\*\s*\n(?:\s*[-*].*(?:\n|$))+?\s*$',
        r'\n\s*\*\*Sources:\*\*\s*\n(?:\s*[-*].*(?:\n|$))+?\s*$',
        r'\n\s*参考来源：\s*\n(?:\s*[-*].*(?:\n|$))+?\s*$',
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _sanitize_final_answer_text(text: str) -> str:
    cleaned = _strip_leading_query_plan_blob(text)
    cleaned = _strip_trailing_sources_block(cleaned)
    return cleaned.strip()


def _build_medical_fallback_notice(*, risk_level: str = "normal", confidence_bucket: str = "no_evidence") -> str:
    mode_label = "回答模式：通用医学信息回答（本次未充分基于知识库检索结果）" if confidence_bucket == "no_evidence" else "回答模式：通用医学信息回答（知识库证据有限）"
    notice = (
        f"{mode_label}\n\n"
        "提醒：以上内容仅供一般医学信息参考，当前回答未充分基于知识库检索结果，不能替代专业医生面对面诊断。"
    )
    if risk_level == "high":
        notice += "\n如症状严重、持续加重，或出现胸痛、呼吸困难、意识异常等情况，请尽快线下就医或急诊评估。"
    else:
        notice += "\n如症状持续加重，或涉及用药、剂量、急症判断，请及时就医。"
    return notice


def _confidence_bucket_label(confidence_bucket: str) -> str:
    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "no_evidence": "未命中足够证据",
    }
    return mapping.get(str(confidence_bucket or "").strip().lower(), "未知")


def _confidence_bucket_explanation(confidence_bucket: str, *, is_medical_request: bool = False) -> str:
    normalized = str(confidence_bucket or "").strip().lower()
    if normalized == "high":
        return "当前回答主要依据知识库中较直接、较匹配的资料整理而成。"
    if normalized == "medium":
        return "当前回答参考了相关资料，但证据覆盖还不算充分，适合先作为初步参考。"
    if normalized == "low":
        return (
            "当前回答仅参考到少量相关资料，结论应保持保守。"
            if is_medical_request
            else "当前回答只参考到有限资料，可作为一般性参考。"
        )
    if normalized == "no_evidence":
        return (
            "知识库这次没有命中足够直接的相关资料，因此以下内容更偏通用医学信息。"
            if is_medical_request
            else "知识库这次没有命中足够直接的相关资料。"
        )
    return ""


def _source_type_label(source_type: str) -> str:
    mapping = {
        "patient_education": "患者教育",
        "public_health": "公共卫生",
        "clinical_guideline": "临床指南",
        "research_article": "研究资料",
        "unknown": "资料",
    }
    normalized = str(source_type or "").strip().lower()
    return mapping.get(normalized, normalized or "资料")


def _freshness_bucket_label(bucket: str) -> str:
    mapping = {
        "fresh": "较新",
        "current": "当前",
        "outdated": "较旧",
        "stale": "较旧",
    }
    normalized = str(bucket or "").strip().lower()
    return mapping.get(normalized, "")


def _format_reference_lines(sources: list[dict]) -> list[str]:
    lines = []
    for item in sources[:3]:
        title = str(item.get("title") or "未知来源").strip()
        source_label = _source_type_label(item.get("source_type", ""))
        freshness_label = _freshness_bucket_label(item.get("freshness_bucket", ""))
        meta_parts = [source_label]
        if freshness_label:
            meta_parts.append(f"时效：{freshness_label}")
        line = f"- {title}"
        if meta_parts:
            line += f"（{'，'.join(meta_parts)}）"
        original_url = str(item.get("original_url") or "").strip()
        if original_url:
            line += f" [链接]({original_url})"
        lines.append(line)
    return lines


def _wants_any_available_doctor(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(token in normalized for token in _ANY_DOCTOR_HINTS)


def _wants_earliest_available_slot(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(token in normalized for token in _EARLIEST_SLOT_HINTS)


def _build_history_reset_messages(messages, keep_recent: int = 5):
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    keep_ids = {getattr(m, "id", None) for m in non_system_messages[-keep_recent:]}
    delete_messages = []
    for message in non_system_messages:
        message_id = getattr(message, "id", None)
        if message_id and message_id not in keep_ids:
            delete_messages.append(RemoveMessage(id=message_id))
    return delete_messages


def _get_user_query(state: State) -> str:
    last_message = state["messages"][-1]
    return state.get("primary_user_query") or str(last_message.content).strip()


def _done_task_ids(state: State) -> set:
    """Ids of planned tasks recorded as done in task_results."""
    return {
        int(r.get("id", -1))
        for r in (state.get("task_results") or [])
        if isinstance(r, dict) and r.get("id") is not None
    }


def _undone_tasks(state: State) -> list:
    """Planned tasks whose id is not in task_results, in plan order."""
    done = _done_task_ids(state)
    out = []
    for t in (state.get("planned_tasks") or []):
        if not isinstance(t, dict):
            continue
        try:
            tid = int(t.get("id", -1))
        except (TypeError, ValueError):
            continue
        if tid not in done:
            out.append(t)
    return out


def _next_undone_task(state: State):
    """Lowest-id planned task whose id is not in task_results, or None."""
    return next(iter(_undone_tasks(state)), None)


def collect_skill_hints() -> tuple[list, list]:
    """Collect (skill_hints, intent_labels) from the skill registry for dynamic
    schema/prompt building. Returns ([], None) when the registry is unavailable."""
    try:
        from skills.registry import get_skill_registry
        _reg = get_skill_registry()
        return _reg.collect_llm_hints(), _reg.build_intent_labels()
    except Exception:
        return [], None


def _clear_per_task_rag_state() -> dict:
    """Reset dict clearing per-task medical-RAG fields so task N+1 doesn't see
    task N's leftovers. Reducer-backed lists use the __reset__ sentinel."""
    return {
        "sub_questions": [],
        "agent_answers": [{"__reset__": True}],
        "rewrittenQuestions": [],
        "questionIsClear": False,
        "grounding_passed": False,
        "grounding_rounds": 0,
        "grounding_critique": "",
        "grounding_evidence_score": None,
    }


def _get_appointment_context(state: State) -> dict:
    return dict(state.get("appointment_context") or {})


def _get_pending_payload(state: State) -> dict:
    return _sanitize_pending_payload(state.get("pending_action_payload"))


def _next_clarification_attempt(state: State) -> int:
    return int(state.get("clarification_attempts") or 0) + 1


__all__ = [
    # --- compiled regex patterns ---
    "_APPOINTMENT_NO_RE",
    "_CN_HOUR_MAP",
    "_CN_HOUR_RE",
    "_MONTH_DAY_RE",
    "_ORDINAL_RE",
    "_SLASH_DATE_RE",
    "_TIME_RE",
    "_WEEKDAY_RE",
    "_YEAR_DATE_RE",
    # --- keyword / hint tuples ---
    "_ABORT_WORDS",
    "_APPOINTMENT_CONFIRM_WORDS",
    "_APPOINTMENT_KEYWORDS",
    "_APPOINTMENT_LIST_HINTS",
    "_CANCEL_CONFIRM_WORDS",
    "_CANCEL_KEYWORDS",
    "_DEPARTMENT_HINTS",
    "_DOCTOR_DISCOVERY_HINTS",
    "_EXPLICIT_APPOINTMENT_CUES",
    "_EXPLICIT_CANCEL_CUES",
    "_GENERAL_CHAT_HINTS",
    "_MEDICAL_FOLLOW_UP_HINTS",
    "_MEDICAL_QUESTION_PATTERNS",
    "_MEDICAL_TERMS",
    "_MEDICATION_RISK_HINTS",
    "_NON_MEDICAL_TOPIC_HINTS",
    "_TOPIC_STOP_WORDS",
    # --- shared helper functions ---
    "_build_appointment_context",
    "_build_history_reset_messages",
    "_build_medical_fallback_notice",
    "_build_recent_context",
    "_clear_pending_action_state",
    "_confidence_bucket_explanation",
    "_confidence_bucket_label",
    "_context_has_medical_signal",
    "_extract_topic_focus",
    "_format_reference_lines",
    "_freshness_bucket_label",
    "_get_appointment_context",
    "_get_pending_payload",
    "_get_user_query",
    "_infer_risk_level",
    "_next_clarification_attempt",
    "_is_abort_request",
    "_is_explicit_confirmation",
    "_json_safe_value",
    "_looks_like_appointment_discovery_query",
    "_looks_like_department_question",
    "_looks_like_explicit_appointment_intent",
    "_looks_like_explicit_cancel_intent",
    "_looks_like_general_non_medical_query",
    "_looks_like_greeting",
    "_looks_like_medical_follow_up",
    "_looks_like_medical_knowledge_question",
    "_looks_like_medical_request",
    "_looks_like_medication_risk_query",
    "_starts_with_polite_decline",
    "_PURE_GREETINGS",
    "_needs_strict_medical_safety",
    "_normalize_date",
    "_normalize_time_slot",
    "_pick_candidate_from_text",
    "_reset_pending_action_if_needed",
    "_sanitize_final_answer_text",
    "_sanitize_pending_payload",
    "_should_use_last_appointment",
    "_source_type_label",
    "_strip_leading_query_plan_blob",
    "_strip_trailing_sources_block",
    "_structured_output_llm",
    # --- appointment-related constants also shared with appointment_nodes ---
    "_ANY_DOCTOR_HINTS",
    "_EARLIEST_SLOT_HINTS",
    "_RESCHEDULE_HINTS",
    # --- appointment-related helpers ---
    "_wants_any_available_doctor",
    "_wants_earliest_available_slot",
]
