from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial
import logging

import config
from .appointment_nodes import handle_appointment, handle_appointment_skill, handle_cancel_appointment
from .edges import *
from .graph_state import AgentState, State
from .persistent_checkpointer import PersistentInMemorySaver
from .rag_nodes import (
    answer_grounding_check,
    collect_answer,
    compress_context,
    decompose_tasks,
    evaluate_evidence,
    fallback_response,
    grounded_answer_generation,
    orchestrator,
    plan_retrieval_queries,
    reset_turn_state,
    revise_answer,
    rewrite_query,
    self_eval,
    should_compress_context,
    plan_tasks,
    dispatch_next_task,
    advance_task,
    completeness_gate,
)
from .routing_nodes import (
    analyze_turn,
    intent_router,
    recommend_department,
    request_clarification,
    summarize_history,
)


logger = logging.getLogger(__name__)


def _build_checkpointer():
    if config.ENABLE_PERSISTENT_GRAPH_CHECKPOINT:
        return PersistentInMemorySaver(config.LANGGRAPH_CHECKPOINT_PATH)
    return InMemorySaver()

def create_agent_graph(llm, tools_list, appointment_service=None, llm_router=None, extra_services=None):
    # Determine which LLM to use for each tier of node
    _light_llm = llm
    _strong_llm = llm
    if llm_router is not None and llm_router.has_tiers:
        _light_llm = llm_router.get_llm("light")
        _strong_llm = llm_router.get_llm("strong")
    mcp_pool = (extra_services or {}).get("user_mcp_pool")

    llm_with_tools = _strong_llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    checkpointer = _build_checkpointer()

    logger.info("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=_strong_llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=_strong_llm))
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)
    if config.ENABLE_AGENTIC_RETRIEVAL:
        # evidence reflection is a LIGHT-tier task (intent/summarization-class)
        agent_builder.add_node("evaluate_evidence", partial(evaluate_evidence, llm=_light_llm))

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    if config.ENABLE_AGENTIC_RETRIEVAL:
        agent_builder.add_edge("tools", "evaluate_evidence")
        agent_builder.add_conditional_edges(
            "evaluate_evidence",
            route_after_evidence,
            {"should_compress_context": "should_compress_context", "fallback_response": "fallback_response"},
        )
    else:
        agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    graph_builder = StateGraph(State)
    # light tier: intent classification, summarization, query planning
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=_light_llm))
    graph_builder.add_node("analyze_turn", analyze_turn)
    graph_builder.add_node("intent_router", partial(intent_router, llm=_light_llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=_light_llm))
    graph_builder.add_node("decompose_tasks", partial(decompose_tasks, llm=_light_llm))
    graph_builder.add_node(reset_turn_state)
    # P4: online self-eval after grounding
    graph_builder.add_node("self_eval", partial(self_eval, llm=_light_llm))
    # strong tier: answer generation, department recommendation
    graph_builder.add_node("recommend_department", partial(recommend_department, llm=_strong_llm))
    graph_builder.add_node("handle_appointment_skill", partial(handle_appointment_skill, llm=_strong_llm, appointment_service=appointment_service, mcp_pool=mcp_pool))
    graph_builder.add_node("handle_appointment", partial(handle_appointment, llm=_strong_llm, appointment_service=appointment_service, mcp_pool=mcp_pool))
    graph_builder.add_node("handle_cancel_appointment", partial(handle_cancel_appointment, llm=_strong_llm, appointment_service=appointment_service, mcp_pool=mcp_pool))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("grounded_answer_generation", partial(grounded_answer_generation, llm=_strong_llm))
    graph_builder.add_node("answer_grounding_check", partial(answer_grounding_check, llm=_strong_llm))

    # Turn planner nodes.
    graph_builder.add_node("plan_tasks", partial(plan_tasks, llm=_light_llm))
    graph_builder.add_node("dispatch_next_task", dispatch_next_task)
    graph_builder.add_node("advance_task", advance_task)
    graph_builder.add_node("completeness_gate", completeness_gate)

    # Register skill nodes (if any skills are registered)
    _skill_route_targets = {}
    try:
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if registry.skills:
            services_dict = {"appointment_service": appointment_service}
            if extra_services:
                services_dict.update(extra_services)
            registry.register_all_nodes(
                graph_builder,
                llm_router=llm_router,
                tools_list=tools_list,
                services=services_dict,
            )
            _skill_route_targets = registry.get_route_mapping()
    except Exception as e:
        logger.warning("Skill framework not available: %s", e)

    # summarize_history was moved off the critical path — the summary from the
    # previous turn is pre-loaded via update_state before graph invocation.
    # It now runs as post-chat cleanup in ChatInterface to avoid blocking the
    # user on the first token.
    graph_builder.add_edge(START, "reset_turn_state")
    graph_builder.add_edge("reset_turn_state", "analyze_turn")
    # Resume branches (primary_intent set) route to intent_router; fresh turns route to plan_tasks.
    _analyze_map = {
        "intent_router": "intent_router",
        "plan_tasks": "plan_tasks",
    }
    graph_builder.add_conditional_edges(
        "analyze_turn",
        route_after_analyze_turn,
        _analyze_map,
    )

    # Planner chain - plan_tasks -> dispatch_next_task -> handler ->
    # advance_task -> (next task | completeness_gate) -> END.
    graph_builder.add_conditional_edges("plan_tasks", route_after_plan_tasks, {
        "dispatch_next_task": "dispatch_next_task",
        "completeness_gate": "completeness_gate",
    })
    _dispatch_map = {
        "rewrite_query": "rewrite_query",
        "handle_appointment_skill": "handle_appointment_skill",
        "recommend_department": "recommend_department",
        "__end__": END,
    }
    for _node_name in _skill_route_targets.values():
        if _node_name not in _dispatch_map:
            _dispatch_map[_node_name] = _node_name
    graph_builder.add_conditional_edges("dispatch_next_task", route_after_dispatch, _dispatch_map)
    graph_builder.add_conditional_edges("advance_task", route_to_next_or_gate, {
        "dispatch_next_task": "dispatch_next_task",
        "completeness_gate": "completeness_gate",
    })
    graph_builder.add_edge("completeness_gate", END)

    # Build the intent_router conditional edges mapping, merging static + skill routes
    _intent_route_map = {
        "rewrite_query": "rewrite_query",
        "recommend_department": "recommend_department",
        "handle_appointment_skill": "handle_appointment_skill",
        "request_clarification": "request_clarification",
        "__end__": END,
    }
    # Add skill route targets (e.g., "greeting_handler")
    for intent_label, node_name in _skill_route_targets.items():
        if node_name not in _intent_route_map.values():
            _intent_route_map[node_name] = node_name

    graph_builder.add_conditional_edges("intent_router", route_after_intent, _intent_route_map)

    # Skill handler nodes that don't have their own edges need a path to END
    for intent_label, node_name in _skill_route_targets.items():
        if node_name not in ("rewrite_query", "recommend_department",
                              "handle_appointment_skill", "request_clarification"):
            graph_builder.add_edge(node_name, END)
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite, {
        "request_clarification": "request_clarification",
        "decompose_tasks": "decompose_tasks",
        "handle_appointment_skill": "handle_appointment_skill",
        "recommend_department": "recommend_department",
        "__end__": END,
        **{node_name: node_name for node_name in _skill_route_targets.values()
           if node_name not in ("request_clarification", "decompose_tasks",
                                "handle_appointment_skill", "recommend_department",
                                "rewrite_query")},
    })
    graph_builder.add_conditional_edges("decompose_tasks", route_after_query_plan)
    graph_builder.add_conditional_edges("request_clarification", route_after_clarification, {
        "intent_router": "intent_router",
        "rewrite_query": "rewrite_query",
        "recommend_department": "recommend_department",
        "handle_appointment_skill": "handle_appointment_skill",
        "handle_appointment": "handle_appointment",
        "handle_cancel_appointment": "handle_cancel_appointment",
    })
    graph_builder.add_edge(["agent"], "grounded_answer_generation")
    _action_map = {
        "request_clarification": "request_clarification",
        "advance_task": "advance_task",
        "__end__": END,
    }
    for _action_src in ("recommend_department", "handle_appointment_skill", "handle_appointment", "handle_cancel_appointment"):
        graph_builder.add_conditional_edges(_action_src, route_after_action, _action_map)
    # P4: after self-eval, drain the next planned task via advance_task.
    _self_eval_map = {
        "advance_task": "advance_task",
    }
    graph_builder.add_conditional_edges("self_eval", route_after_self_eval, _self_eval_map)
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    if config.ENABLE_ANSWER_REFLECTION:
        # P2: answer reflection loop - critique + evidence-bounded rewrite, re-checked.
        # revise_answer is a LIGHT-tier task (critique/rewrite-class, like evaluate_evidence).
        graph_builder.add_node("revise_answer", partial(revise_answer, llm=_light_llm))
        _grounding_map = {"revise_answer": "revise_answer"}
        if config.ENABLE_SELF_EVAL:
            _grounding_map["self_eval"] = "self_eval"
        _grounding_map["advance_task"] = "advance_task"
        graph_builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            _grounding_map,
        )
        graph_builder.add_edge("revise_answer", "answer_grounding_check")
    else:
        _grounding_map = {"advance_task": "advance_task"}
        if config.ENABLE_SELF_EVAL:
            _grounding_map["self_eval"] = "self_eval"
        graph_builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            _grounding_map,
        )

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    logger.info("Agent graph compiled successfully.")
    return agent_graph
