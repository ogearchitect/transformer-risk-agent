from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.orchestrator import handle_query


class OrchestratorState(TypedDict, total=False):
    message: str
    plan: list[str]
    tool_calls: list[dict]
    answer: str


def _orchestrate(state: OrchestratorState) -> OrchestratorState:
    response = handle_query(state.get("message", ""))
    return {"plan": response.plan, "tool_calls": response.tool_calls, "answer": response.answer}


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("orchestrator", _orchestrate)
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", END)
    return graph.compile()
