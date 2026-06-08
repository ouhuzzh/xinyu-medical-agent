"""MCPSkill — registers MCP tools into the LangGraph agent as a pluggable skill.

Any MCP service a user has bound credentials for (hospital booking, pharmacy
ordering, lab-result lookup, payment processing, ...) has its tools loaded
per-user and namespaced by server code.  The LLM sees all available tools
and decides which to call based on the user's request.

Architecture:
  1. User binds MCP server credentials (token + URL) via the settings UI.
  2. On first use, UserMCPPool connects to each server and loads its tools.
  3. MCPSkill's match() activates whenever the query has an MCP-action verb
     AND the user has bound servers (checked via context).
  4. The handler binds all of the user's tools to the LLM and lets it choose.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# System prompt for the MCP tool-calling agent (generic, service-agnostic)
_SYSTEM_PROMPT = """你是智能助手中的外部服务调用模块。

可用工具来自不同外部服务的 MCP 接口。每个工具名称以服务代码开头（如 xiehe__book_appointment、yaofang__check_stock）。

规则:
1. 根据用户请求，从工具列表中选择最合适的工具调用。
2. 调用工具前正确组装参数，用自然语言回复结果。
3. 用户没指定具体服务时，先描述可选的服务让用户选择。
4. 严禁伪造工具调用结果。若工具调用全部失败，明确告知用户。
"""

# MCP-action verbs — queries containing these likely need MCP tool access.
# NOT keyword-matching specific services — that's the LLM's job.
_MCP_ACTION_VERBS = (
    "帮我查", "帮我查一下", "帮我挂", "帮我预约", "帮我约", "帮我订",
    "帮我取消", "退号", "退掉", "查一下", "看一下", "看看",
    "有没有号", "有没有空", "还有号吗", "带我去", "去查",
    "有几个", "多少钱", "价格", "支付", "付一下",
    "有什么", "有哪些", "帮我找", "搜索", "搜一下",
    "挂号", "预约", "约号", "排号", "取号",
    "查医生", "查科室", "查排班", "查库存",
)


class MCPSkill(BaseSkill):
    """Skill that delegates to MCP-provided tools from any service."""

    @property
    def name(self) -> str:
        return "mcp_services"

    @property
    def priority(self) -> int:
        # Lower than greeting (10) but higher than medical_rag (60)
        return 25

    @property
    def intent_label(self) -> str:
        return "mcp_services"

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Match queries that are likely to need MCP service tool access.

        Strategy: detect action verbs (book, search, cancel, check, ...)
        rather than hardcoding specific hospital/service names. The LLM
        decides WHICH service to use based on the available tools.
        """
        import config
        if not config.MCP_ENABLED:
            return False

        normalized = (query or "").strip()
        if not normalized:
            return False

        # Direct action-verb match
        if any(verb in normalized for verb in _MCP_ACTION_VERBS):
            return True

        # Context-aware: short follow-up after a previous MCP interaction
        recent_context = context.get("recent_context", "") or ""
        if any(verb in recent_context for verb in _MCP_ACTION_VERBS):
            if len(normalized) <= 10:
                return True

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
                    "intent": "mcp_services",
                    "messages": [AIMessage(content="外部服务功能暂不可用。")],
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
                    "intent": "mcp_services",
                    "messages": [AIMessage(content="登录后才能使用外部服务功能。")],
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
                hint = "你还没绑定任何外部服务。请在 设置 → 服务绑定 中添加 token。"
                if failed:
                    hint += f"\n已绑定但连接失败：{', '.join(failed.keys())}"
                return {
                    "intent": "mcp_services",
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
            if connected:
                context_parts.append(f"可用服务: {', '.join(connected)}")
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
                    "intent": "mcp_services",
                    "messages": [AIMessage(content=f"外部服务调用失败：{type(e).__name__}")],
                    "route_reason": "skill:mcp_llm_error",
                    "decision_source": "skill",
                }

            # If the LLM called a tool, execute it
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # LLM answered directly without tool call
                return {
                    "intent": "mcp_services",
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
                except Exception as te:
                    logger.warning("Tool %s execution failed", tool_name, exc_info=True)
                    tool_outputs.append(f"工具 {tool_name} 调用失败: {type(te).__name__}")

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
                "intent": "mcp_services",
                "messages": [AIMessage(content=final_text)],
                "route_reason": "skill:mcp_tool_executed",
                "decision_source": "skill",
            }

        return {"mcp_services_handler": mcp_handler}

    def get_route_targets(self) -> Dict[str, str]:
        return {"mcp_services": "mcp_services_handler"}


_NAMESPACE_SEP = __import__("config").MCP_TOOL_NAMESPACE_SEPARATOR
