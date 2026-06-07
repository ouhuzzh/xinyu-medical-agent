"""MCPSkill — registers MCP-driven hospital tools into the LangGraph agent.

When user mentions a remote hospital (e.g., "挂协和的号"), this skill:
  1. Resolves user_id from thread_id via ChatSessionStore
  2. Fetches the user's MCP tools via UserMCPPool
  3. Binds those tools to the strong LLM
  4. Lets the LLM decide which hospital tool to call
  5. Executes the tool, returns result to user

Differs from the local AppointmentSkill in that tools come from external
MCP servers (per-user, per-hospital) rather than the local PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)


# Hospital names commonly mentioned in user queries (used for fast detection)
# Production should derive this from the hospital_registry dynamically.
_HOSPITAL_KEYWORDS = (
    "协和", "仁济", "中山", "同济", "华西", "湘雅",
    "外院", "外部医院", "其他医院",
)

_SYSTEM_PROMPT = """你是医疗助手中的外部医院预约模块。

可用工具来自不同医院的 MCP 服务。每个工具名称以医院 code 开头（如 xiehe__book_appointment）。

规则:
1. 用户提到的医院名称要匹配到对应的工具前缀（"协和" → xiehe__*）。
2. 用户没指定医院时，列出可用医院让用户选择。
3. 调用工具前先组装参数，调用后用自然语言回复用户结果。
4. 严禁伪造工具调用结果。若所有工具调用都失败，明确告知用户。
"""


class MCPSkill(BaseSkill):
    """Skill that delegates to MCP-provided hospital tools."""

    @property
    def name(self) -> str:
        return "mcp_hospital"

    @property
    def priority(self) -> int:
        # Lower than greeting (10) but higher than medical_rag (60)
        # — explicit mentions of external hospitals should win.
        return 25

    @property
    def intent_label(self) -> str:
        return "mcp_hospital"

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Match queries mentioning external hospitals, including follow-ups."""
        import config
        if not config.MCP_ENABLED:
            return False
        normalized = (query or "").lower()

        # Must mention a known hospital
        has_hospital = any(kw in query for kw in _HOSPITAL_KEYWORDS)
        if not has_hospital:
            # Context-aware follow-up: "神经科" after asking about 协和
            recent_context = context.get("recent_context", "") or ""
            if any(kw in recent_context for kw in _HOSPITAL_KEYWORDS) and len(normalized) <= 10:
                return True
            return False

        # Hospital mentioned + appointment-related action: route to MCP
        from rag_agent.node_helpers import (
            _looks_like_appointment_discovery_query,
            _looks_like_explicit_appointment_intent,
            _looks_like_explicit_cancel_intent,
        )
        if (_looks_like_appointment_discovery_query(query)
                or _looks_like_explicit_appointment_intent(query)
                or _looks_like_explicit_cancel_intent(query)):
            return True

        # Hospital mentioned but no action (e.g. "协和医院很厉害吗"):
        # let medical_rag handle general knowledge questions
        return False

    def get_state_schema(self) -> Dict[str, Any]:
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        services = services or {}
        user_mcp_pool = services.get("user_mcp_pool")
        chat_sessions = services.get("chat_sessions")

        if user_mcp_pool is None or chat_sessions is None:
            logger.warning(
                "MCPSkill registered without user_mcp_pool/chat_sessions; will be a no-op."
            )

        # Pick the LLM tier
        llm = None
        if llm_router is not None and getattr(llm_router, "has_tiers", False):
            llm = llm_router.get_llm("strong")
        else:
            try:
                from model_factory import get_chat_model
                llm = get_chat_model()
            except Exception:
                logger.warning("Failed to get strong LLM for MCPSkill", exc_info=True)
                llm = None

        def mcp_handler(state: dict) -> dict:
            """Handler that loads user's MCP tools and lets LLM use them."""
            user_query = ""
            messages = state.get("messages") or []
            if messages:
                last = messages[-1]
                user_query = str(getattr(last, "content", "") or "")

            thread_id = state.get("thread_id", "")
            if user_mcp_pool is None or chat_sessions is None or llm is None:
                return {
                    "intent": "mcp_hospital",
                    "messages": [AIMessage(content="外部医院预约功能暂不可用。")],
                    "route_reason": "skill:mcp_no_services",
                    "decision_source": "skill",
                }

            # Resolve user_id
            try:
                session_info = chat_sessions.get_session(thread_id)
                user_id = (session_info or {}).get("owner_user_id", "") or ""
            except Exception:
                user_id = ""
            if not user_id:
                return {
                    "intent": "mcp_hospital",
                    "messages": [AIMessage(content="登录后才能使用外部医院预约功能。")],
                    "route_reason": "skill:mcp_no_user",
                    "decision_source": "skill",
                }

            # Fetch this user's MCP tools
            try:
                user_tools = user_mcp_pool.get_tools_for_user(user_id)
                connected = user_mcp_pool.get_connected_hospitals(user_id)
                failed = user_mcp_pool.get_failed_hospitals(user_id)
            except Exception:
                logger.exception("Failed to load MCP tools for user %s", user_id)
                user_tools, connected, failed = [], [], {}

            if not user_tools:
                hint = "你还没绑定任何外部医院。请在 账号-医院绑定 中添加 token。"
                if failed:
                    hint += f"\n已绑定但连接失败：{', '.join(failed.keys())}"
                return {
                    "intent": "mcp_hospital",
                    "messages": [AIMessage(content=hint)],
                    "route_reason": "skill:mcp_no_tools",
                    "decision_source": "skill",
                }

            # Build context-rich prompt
            context_parts = []
            conv_summary = state.get("conversation_summary", "") or ""
            recent_ctx = state.get("recent_context", "") or ""
            if conv_summary.strip():
                context_parts.append(f"对话摘要: {conv_summary}")
            if recent_ctx.strip():
                context_parts.append(f"最近对话: {recent_ctx}")
            context_text = "\n".join(context_parts)
            prompt_content = user_query
            if context_text:
                prompt_content = f"{context_text}\n\n用户当前问题: {user_query}"

            try:
                llm_with_tools = llm.bind_tools(user_tools)
                response = llm_with_tools.invoke(
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=prompt_content),
                    ]
                )
            except Exception as e:
                logger.exception("MCP LLM call failed")
                return {
                    "intent": "mcp_hospital",
                    "messages": [AIMessage(content=f"外部医院调用失败：{type(e).__name__}")],
                    "route_reason": "skill:mcp_llm_error",
                    "decision_source": "skill",
                }

            # If the LLM called a tool, execute it
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # LLM answered directly without tool call
                return {
                    "intent": "mcp_hospital",
                    "messages": [response],
                    "route_reason": "skill:mcp_direct_answer",
                    "decision_source": "skill",
                }

            # Execute the tool calls
            tool_map = {t.name: t for t in user_tools}
            tool_outputs = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {}) or {}
                tool = tool_map.get(tool_name)
                if tool is None:
                    tool_outputs.append(f"工具 {tool_name} 未找到")
                    continue
                try:
                    result = tool.invoke(tool_args)
                    tool_outputs.append(str(result))
                    # Track successful use
                    if user_mcp_pool and _NAMESPACE_SEP in tool_name:
                        hospital_code = tool_name.split(_NAMESPACE_SEP)[0]
                        try:
                            from .user_hospital_store import UserHospitalStore
                            UserHospitalStore().mark_used(user_id, hospital_code)
                        except Exception:
                            pass
                except Exception as te:
                    logger.warning("Tool %s execution failed", tool_name, exc_info=True)
                    tool_outputs.append(f"工具 {tool_name} 调用失败: {type(te).__name__}: {str(te)[:200]}")

            # Summarize tool results for the user
            try:
                summary = llm.invoke(
                    [
                        SystemMessage(content="把工具调用结果用 1-3 句中文回复用户。"),
                        HumanMessage(content="\n\n".join(tool_outputs)),
                    ]
                )
                final_text = str(summary.content)
            except Exception:
                final_text = "\n".join(tool_outputs)

            return {
                "intent": "mcp_hospital",
                "messages": [AIMessage(content=final_text)],
                "route_reason": "skill:mcp_tool_executed",
                "decision_source": "skill",
            }

        return {"mcp_hospital_handler": mcp_handler}

    def get_route_targets(self) -> Dict[str, str]:
        return {"mcp_hospital": "mcp_hospital_handler"}


# Lazy import to avoid circular dependencies
import config
_NAMESPACE_SEP = config.MCP_TOOL_NAMESPACE_SEPARATOR
