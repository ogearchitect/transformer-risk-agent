from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.llm import get_llm
from src.agents.tools import get_failure_probability, simulate_what_if


class ScenarioInputs(TypedDict, total=False):
    asset_id: str
    loading_pct: float
    defer_years: int
    ambient_delta_c: float


class ScenarioState(TypedDict, total=False):
    asset_id: str | None
    history: list[dict]
    inputs: ScenarioInputs
    last_result: dict | None
    baseline_curve: dict | None
    next: Literal["gather", "simulate", "narrate", "done"]


class _GraphState(ScenarioState, total=False):
    user_msg: str
    assistant_text: str


DEFAULT_INPUTS: ScenarioInputs = {"loading_pct": 95.0, "defer_years": 0, "ambient_delta_c": 0.0}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_asset_id(text: str) -> str | None:
    for token in text.upper().replace(",", " ").split():
        cleaned = token.strip(".!?;:")
        if cleaned.startswith("T-") and len(cleaned) == 6 and cleaned[2:].isdigit():
            return cleaned
    return None


def _extract_years(text: str) -> int | None:
    tokens = text.lower().replace("-", " ").replace(",", " ").split()
    for i, token in enumerate(tokens):
        if token.startswith("year") and i > 0 and tokens[i - 1].isdigit():
            return int(tokens[i - 1])
    return None


def _extract_loading_pct(text: str) -> float | None:
    for pattern in (r"loading\s*(?:at|to)?\s*(\d+(?:\.\d+)?)\s*%?", r"(\d+(?:\.\d+)?)\s*%\s*loading", r"\b(\d+(?:\.\d+)?)\s*%\b"):
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            return float(match.group(1))
    return None


def _extract_defer_years(text: str) -> int | None:
    if match := re.search(r"defer(?:red|ring)?\s*(?:by\s*)?(\d+)\s*years?", text, flags=re.IGNORECASE):
        return int(match.group(1))
    return _extract_years(text) if ("defer" in text.lower() or "year" in text.lower()) else None


def _extract_ambient_delta_c(text: str) -> float | None:
    for pattern in (r"ambient[^\d+-]*([+-]?\d+(?:\.\d+)?)\s*°?\s*c", r"([+-]?\d+(?:\.\d+)?)\s*°?\s*c\s*ambient", r"temperature[^\d+-]*([+-]?\d+(?:\.\d+)?)\s*°?\s*c"):
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            return float(match.group(1))
    return None


def _ready_for_simulation(state: _GraphState) -> bool:
    inputs = state.get("inputs", {})
    return bool(state.get("asset_id")) and all(key in inputs for key in ("asset_id", "loading_pct", "defer_years", "ambient_delta_c"))


def new_state(asset_id: str | None = None) -> ScenarioState:
    return {
        "asset_id": asset_id,
        "history": [],
        "inputs": {"asset_id": asset_id} if asset_id else {},
        "last_result": None,
        "baseline_curve": None,
        "next": "gather",
    }


def gather_inputs(state: _GraphState) -> _GraphState:
    user_msg = state.get("user_msg", "")
    history = [*state.get("history", []), {"role": "user", "content": user_msg}]
    prior_asset_id = state.get("asset_id")
    parsed_asset_id = _extract_asset_id(user_msg)
    asset_id = parsed_asset_id or prior_asset_id
    inputs: ScenarioInputs = dict(state.get("inputs", {}))
    baseline_curve, last_result = state.get("baseline_curve"), state.get("last_result")
    if parsed_asset_id and parsed_asset_id != prior_asset_id:
        inputs, baseline_curve, last_result = {"asset_id": parsed_asset_id}, None, None
    if asset_id:
        inputs["asset_id"] = asset_id
    extracted = {
        "loading_pct": _extract_loading_pct(user_msg),
        "defer_years": _extract_defer_years(user_msg),
        "ambient_delta_c": _extract_ambient_delta_c(user_msg),
    }
    for key, value in extracted.items():
        if value is None:
            continue
        if key == "loading_pct":
            inputs[key] = _clamp(float(value), 10.0, 180.0)
        elif key == "defer_years":
            inputs[key] = int(_clamp(float(value), 0.0, 10.0))
        else:
            inputs[key] = _clamp(float(value), -20.0, 20.0)
    for key, value in DEFAULT_INPUTS.items():
        inputs.setdefault(key, value)
    if not asset_id:
        question = "Which transformer should I simulate? Please provide an asset ID like T-0042."
        history.append({"role": "assistant", "content": question})
        return {"history": history, "asset_id": None, "inputs": inputs, "last_result": last_result, "baseline_curve": baseline_curve, "assistant_text": question, "next": "done"}
    return {"history": history, "asset_id": asset_id, "inputs": inputs, "last_result": last_result, "baseline_curve": baseline_curve, "assistant_text": "", "next": "simulate"}


def run_simulation(state: _GraphState) -> _GraphState:
    asset_id = state["asset_id"]
    inputs = {**DEFAULT_INPUTS, **state.get("inputs", {})}
    baseline_curve = state.get("baseline_curve")
    try:
        if not baseline_curve or baseline_curve.get("id") != asset_id:
            baseline_curve = get_failure_probability.invoke({"id": asset_id, "horizon_years": 5})
        last_result = simulate_what_if.invoke({"id": asset_id, "loading_pct": inputs["loading_pct"], "defer_years": inputs["defer_years"], "ambient_delta_c": inputs["ambient_delta_c"]})
    except Exception as exc:
        last_result = {"id": asset_id, "error": str(exc), "inputs": inputs}
    return {"baseline_curve": baseline_curve, "last_result": last_result, "next": "narrate"}


def narrate(state: _GraphState) -> _GraphState:
    scenario = state.get("last_result") or {}
    recommendation = (
        "Scenario execution failed; verify model artifacts and rerun the simulation."
        if scenario.get("error")
        else "Scenario increases risk; consider reducing loading, limiting ambient exposure, or avoiding deferral."
        if float(scenario.get("delta_p5", 0.0)) > 0
        else "Scenario does not worsen 5-year risk; confirm with maintenance and operational constraints."
    )
    tool_context = (
        f"Asset: {state.get('asset_id')}\n"
        f"Scenario inputs: {state.get('inputs', {})}\n"
        f"Baseline failure output: {state.get('baseline_curve') or {}}\n"
        f"Scenario simulation output: {scenario}\n"
        f"Recommendation hint: {recommendation}"
    )
    prompt = "Summarize baseline versus scenario 5-year failure risk, cite the delta, mention the driver inputs, and end with a practical recommendation."
    assistant_text = get_llm().invoke(prompt, tool_context=tool_context).content
    history = [*state.get("history", []), {"role": "assistant", "content": assistant_text}]
    return {"history": history, "assistant_text": assistant_text, "next": "done"}


def _route_after_gather(state: _GraphState) -> Literal["simulate", "done"]:
    return "simulate" if _ready_for_simulation(state) and state.get("next") == "simulate" else "done"


def _build_graph():
    graph = StateGraph(_GraphState)
    graph.add_node("gather_inputs", gather_inputs)
    graph.add_node("run_simulation", run_simulation)
    graph.add_node("narrate", narrate)
    graph.set_entry_point("gather_inputs")
    graph.add_conditional_edges("gather_inputs", _route_after_gather, {"simulate": "run_simulation", "done": END})
    graph.add_edge("run_simulation", "narrate")
    graph.add_edge("narrate", END)
    return graph.compile()


_SCENARIO_GRAPH = _build_graph()


def run_scenario_turn(state: ScenarioState, user_msg: str) -> tuple[ScenarioState, str]:
    graph_state: _GraphState = {**new_state(state.get("asset_id")), **state, "user_msg": user_msg, "assistant_text": ""}
    result = _SCENARIO_GRAPH.invoke(graph_state)
    next_state: ScenarioState = {key: result.get(key) for key in ("asset_id", "history", "inputs", "last_result", "baseline_curve", "next")}
    return next_state, result.get("assistant_text", "")
