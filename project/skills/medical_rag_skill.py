"""Medical RAG skill — health questions, symptoms, casual chat.

This is the default fallback intent.  It has NO L1 keywords — classification
is handled entirely by L2 embedding semantic matching and L3 LLM fallback.
It reuses the existing rewrite_query → RAG pipeline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .base_skill import BaseSkill


# Canonical medical knowledge hints — shared with ChatInterface for
# optimistic pre-classification in the SSE streaming path.
KB_HINTS: Tuple[str, ...] = (
    "高血压", "糖尿病", "感冒", "发烧", "发热", "低烧", "高烧", "头疼", "头痛", "偏头痛",
    "头晕", "眩晕", "咳嗽", "咳痰", "咽痛", "嗓子疼", "喉咙痛", "流鼻涕", "鼻塞",
    "腹痛", "肚子疼", "胃痛", "腹泻", "拉肚子", "便秘", "恶心", "呕吐", "胸闷", "胸痛",
    "心悸", "心慌", "乏力", "呼吸困难", "气短", "肺炎", "哮喘", "鼻炎", "胃炎", "肠胃炎",
    "血压", "血糖", "症状", "治疗", "检查", "药", "用药", "疼", "痛", "不舒服",
    "hypertension", "diabetes", "fever", "cough", "dizziness", "symptom", "treatment",
)

KB_QUESTION_HINTS: Tuple[str, ...] = (
    "是什么", "怎么回事", "为什么", "原因", "症状", "表现", "怎么办", "如何", "怎么处理",
    "怎么缓解", "严重吗", "会不会", "会引起", "会导致", "能不能", "可以吗", "要不要",
    "治疗", "预防", "what is", "why", "how to", "symptoms", "treatment",
)

FALLBACK_DANGER_HINTS: Tuple[str, ...] = (
    "胸痛", "胸闷", "呼吸困难", "气短", "意识模糊", "抽搐", "晕厥", "剧烈", "突然",
    "持续加重", "大出血", "高热", "肢体无力", "视物模糊",
)


class MedicalRagSkill(BaseSkill):
    """Handles medical knowledge question intents."""

    @property
    def name(self) -> str:
        return "medical_rag"

    @property
    def priority(self) -> int:
        return 60

    @property
    def intent_label(self) -> str:
        return "medical_rag"

    # No L1 keywords — medical_rag is the default, classification via L2/L3
    @property
    def keywords(self):
        return ()

    @property
    def utterances(self) -> List[str]:
        return [
            # Boundary cases — should be medical_rag NOT appointment/cancel
            "预约前要注意什么", "挂号前需要准备什么",
            "取消对药物的依赖会有什么后果", "停药后会有什么反应",
            # Standard medical queries — short
            "感冒吃什么药", "咳嗽怎么办", "发烧怎么办", "头痛怎么处理",
            "胃痛吃什么", "过敏用什么药", "失眠怎么缓解",
            # Standard medical queries — long
            "高血压怎么控制", "这个药有什么副作用", "手术后要注意什么",
            "胸痛是什么原因引起的", "咳嗽一直不好怎么办",
            # Medical follow-ups (very short)
            "严重吗", "怎么办", "会好吗", "需要注意什么",
            # Casual / non-medical
            "今天天气真好", "谢谢你帮我", "心情不太好",
            "what are the symptoms of", "how to treat",
        ]

    @property
    def llm_hint(self) -> str:
        return "health questions, symptoms, treatments, casual chat, emotional support"

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Provide a safe legacy match without turning medical terms into L1 routes.

        The graph's fast L1 router intentionally leaves this skill keyword-free
        so ambiguous medical language can still use the normal rewrite flow.
        ``SkillRegistry.classify_intent`` is also used by diagnostics and
        evaluators, though, and it needs a meaningful medical match.
        """
        del context
        normalized = (query or "").strip().lower()
        if not normalized:
            return False
        has_medical_hint = any(hint in normalized for hint in KB_HINTS)
        has_question = (
            any(hint in normalized for hint in KB_QUESTION_HINTS)
            or normalized.endswith(("?", "？", "吗"))
        )
        return has_medical_hint and has_question

    def get_state_schema(self) -> Dict[str, Any]:
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        return {}

    def get_route_targets(self) -> Dict[str, str]:
        return {"medical_rag": "rewrite_query"}
