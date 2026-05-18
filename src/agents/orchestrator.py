from __future__ import annotations

from dataclasses import dataclass

from src.agents.llm import get_llm
from src.agents.tools import (
    explain_transformer_risk,
    get_failure_probability,
    rank_fleet,
    recommend_replacement_plan,
    recommend_spare_strategy,
    simulate_twin_scenario,
    simulate_what_if,
)


@dataclass
class AgentResponse:
    plan: list[str]
    tool_calls: list[dict]
    answer: str


def _extract_asset_id(text: str) -> str | None:
    upper = text.upper()
    for token in upper.replace(",", " ").split():
        cleaned = token.strip(".!?;:")
        if cleaned.startswith("T-") and len(cleaned) == 6 and cleaned[2:].isdigit():
            return cleaned
    return None


def _extract_top_n(text: str, default: int = 5) -> int:
    tokens = text.lower().replace(",", " ").split()
    for i, token in enumerate(tokens[:-1]):
        if token == "top" and tokens[i + 1].isdigit():
            return int(tokens[i + 1])
    return default


def _extract_budget_musd(text: str, default: float = 15.0) -> float:
    token = ""
    seen_dollar = False
    for ch in text:
        if ch == "$":
            seen_dollar = True
            token = ""
            continue
        if seen_dollar and (ch.isdigit() or ch == "."):
            token += ch
        elif seen_dollar and token:
            break
    return float(token) if token else default


def _extract_years(text: str, default: int = 5) -> int:
    tokens = text.lower().replace("-", " ").replace(",", " ").split()
    for i, token in enumerate(tokens):
        if token.startswith("year") and i > 0 and tokens[i - 1].isdigit():
            return int(tokens[i - 1])
    return default


def _extract_voltage(text: str, default: int = 230) -> int:
    for candidate in (69, 138, 230, 345, 500):
        if str(candidate) in text:
            return candidate
    return default


def handle_query(message: str) -> AgentResponse:
    q = message.lower()
    plan: list[str] = []
    calls: list[dict] = []

    if "top" in q and "risk" in q:
        plan.append("Rank fleet by monetized risk")
        n = _extract_top_n(q, default=5)
        result = rank_fleet.invoke({"top_n": n, "sort_by": "monetized_risk_usd"})
        calls.append({"tool": "rank_fleet", "args": {"top_n": n}, "result": result})
    elif any(k in q for k in ("twin", "heat wave", "heat-wave", "contingency", "simulate", "scenario")):
        plan.append("Run physics digital-twin scenario")
        aid = _extract_asset_id(message) or "T-0001"
        scenario = "heat_wave"
        if "summer" in q or "peak" in q:
            scenario = "summer_peak"
        elif "contingency" in q or "transfer" in q:
            scenario = "contingency_transfer"
        elif "cooling" in q and "loss" in q:
            scenario = "cooling_loss"
        elif "normal" in q:
            scenario = "normal_day"
        sync = "azure" in q or "adt" in q or "mirror" in q
        result = simulate_twin_scenario.invoke({"id": aid, "scenario": scenario, "repeat_days": 1, "sync_to_adt": sync})
        calls.append({"tool": "simulate_twin_scenario", "args": {"id": aid, "scenario": scenario, "sync_to_adt": sync}, "result": result})
    elif "replacement" in q or "budget" in q:
        plan.append("Generate budget-constrained replacement plan")
        b = _extract_budget_musd(q, default=15.0)
        result = recommend_replacement_plan.invoke({"budget_musd": b, "horizon_years": 5})
        calls.append({"tool": "recommend_replacement_plan", "args": {"budget_musd": b, "horizon_years": 5}, "result": result})
    elif "spare" in q:
        plan.append("Run Monte Carlo spare strategy")
        kv = _extract_voltage(q, default=230)
        result = recommend_spare_strategy.invoke({"voltage_class_kv": kv})
        calls.append({"tool": "recommend_spare_strategy", "args": {"voltage_class_kv": kv}, "result": result})
    elif "what-if" in q or "defer" in q:
        plan.append("Run scenario simulation")
        aid = _extract_asset_id(message) or "T-0001"
        defer = _extract_years(q, default=2)
        result = simulate_what_if.invoke({"id": aid, "loading_pct": 95, "defer_years": defer, "ambient_delta_c": 2})
        calls.append(
            {"tool": "simulate_what_if", "args": {"id": aid, "loading_pct": 95, "defer_years": defer, "ambient_delta_c": 2}, "result": result}
        )
    elif "failure probability" in q or ("probability" in q and "t-" in q):
        plan.append("Get multi-horizon failure probability")
        aid = _extract_asset_id(message) or "T-0001"
        horizon = min(10, max(1, _extract_years(q, default=5)))
        result = get_failure_probability.invoke({"id": aid, "horizon_years": horizon})
        calls.append({"tool": "get_failure_probability", "args": {"id": aid, "horizon_years": horizon}, "result": result})
    else:
        plan.append("Explain selected transformer risk")
        aid = _extract_asset_id(message) or "T-0001"
        calls.append({"tool": "get_failure_probability", "args": {"id": aid, "horizon_years": 5}, "result": get_failure_probability.invoke({"id": aid, "horizon_years": 5})})
        calls.append({"tool": "explain_transformer_risk", "args": {"id": aid}, "result": explain_transformer_risk.invoke({"id": aid})})

    tool_context = "\n".join([f"- {c['tool']}: {str(c['result'])[:900]}" for c in calls])
    llm = get_llm()
    answer = llm.invoke(message, tool_context=tool_context).content

    return AgentResponse(plan=plan, tool_calls=calls, answer=answer)
