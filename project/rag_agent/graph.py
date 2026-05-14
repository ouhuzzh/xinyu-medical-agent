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
    fallback_response,
    grounded_answer_generation,
    orchestrator,
    plan_retrieval_queries,
    rewrite_query,
    should_compress_context,
)
from .routing_nodes import (
    analyze_turn,
    intent_router,
    prepare_secondary_turn,
    recommend_department,
    request_clarification,
    summarize_history,
)


logger = logging.getLogger(__name__)


def _build_checkpointer():
    if config.ENABLE_PERSISTENT_GRAPH_CHECKPOINT:
        return PersistentInMemorySaver(config.LANGGRAPH_CHECKPOINT_PATH)
    return InMemorySaver()

def create_agent_graph(llm, tools_list, appointment_service=None):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    checkpointer = _build_checkpointer()

    logger.info("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("analyze_turn", analyze_turn)
    graph_builder.add_node("intent_router", partial(intent_router, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node("plan_retrieval_queries", partial(plan_retrieval_queries, llm=llm))
    graph_builder.add_node("recommend_department", partial(recommend_department, llm=llm))
    graph_builder.add_node("handle_appointment_skill", partial(handle_appointment_skill, llm=llm, appointment_service=appointment_service))
    graph_builder.add_node("handle_appointment", partial(handle_appointment, llm=llm, appointment_service=appointment_service))
    graph_builder.add_node("handle_cancel_appointment", partial(handle_cancel_appointment, llm=llm, appointment_service=appointment_service))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("prepare_secondary_turn", prepare_secondary_turn)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("grounded_answer_generation", partial(grounded_answer_generation, llm=llm))
    graph_builder.add_node("answer_grounding_check", partial(answer_grounding_check, llm=llm))

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "analyze_turn")
    graph_builder.add_edge("analyze_turn", "intent_router")
    graph_builder.add_conditional_edges("intent_router", route_after_intent, {
        "rewrite_query": "rewrite_query",
        "recommend_department": "recommend_department",
        "handle_appointment_skill": "handle_appointment_skill",
        "request_clarification": "request_clarification",
        "__end__": END,
    })
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite, {
        "request_clarification": "request_clarification",
        "plan_retrieval_queries": "plan_retrieval_queries",
    })
    graph_builder.add_conditional_edges("plan_retrieval_queries", route_after_query_plan)
    graph_builder.add_conditional_edges("request_clarification", route_after_clarification, {
        "intent_router": "intent_router",
        "rewrite_query": "rewrite_query",
        "recommend_department": "recommend_department",
        "handle_appointment_skill": "handle_appointment_skill",
        "handle_appointment": "handle_appointment",
        "handle_cancel_appointment": "handle_cancel_appointment",
    })
    graph_builder.add_edge(["agent"], "grounded_answer_generation")
    graph_builder.add_conditional_edges("recommend_department", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "__end__": END})
    graph_builder.add_conditional_edges("handle_appointment_skill", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "__end__": END})
    graph_builder.add_conditional_edges("handle_appointment", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "__end__": END})
    graph_builder.add_conditional_edges("handle_cancel_appointment", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "__end__": END})
    graph_builder.add_conditional_edges("prepare_secondary_turn", route_after_prepare_secondary_turn, {
        "rewrite_query": "rewrite_query",
        "handle_appointment": "handle_appointment",
        "handle_cancel_appointment": "handle_cancel_appointment",
        "recommend_department": "recommend_department",
    })
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    graph_builder.add_edge("answer_grounding_check", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    logger.info("Agent graph compiled successfully.")
    return agent_graph
