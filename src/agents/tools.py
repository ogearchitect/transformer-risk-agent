from __future__ import annotations

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.loader import load_seed_data
from src.models.dga_diagnostics import diagnose_latest
from src.models.explain import build_narrative, top_risk_factors
from src.models.failure_prob import get_asset_features, get_asset_probability
from src.models.fleet_optimizer import build_replacement_plan, monte_carlo_spares
from src.models.health_index import compute_health_index


class AssetInput(BaseModel):
    id: str = Field(..., description="Transformer asset id like T-0001")


class FailureProbInput(BaseModel):
    id: str
    horizon_years: int = Field(..., ge=1, le=10)


class RankFleetInput(BaseModel):
    top_n: int = Field(default=5, ge=1, le=50)
    sort_by: str = Field(default="monetized_risk_usd")


class WhatIfInput(BaseModel):
    id: str
    loading_pct: float = Field(..., ge=10, le=180)
    defer_years: int = Field(default=0, ge=0, le=10)
    ambient_delta_c: float = Field(default=0.0, ge=-20, le=20)


class ReplacementPlanInput(BaseModel):
    budget_musd: float = Field(..., ge=1.0)
    horizon_years: int = Field(default=5, ge=1, le=20)


class SpareInput(BaseModel):
    voltage_class_kv: int


@lru_cache(maxsize=1)
def _ctx() -> dict[str, pd.DataFrame]:
    return load_seed_data()


def _asset_bundle(asset_id: str) -> dict[str, pd.DataFrame]:
    data = _ctx()
    return {
        k: df[df["id"] == asset_id].copy() if "id" in df.columns else df for k, df in data.items()
    }


def _asset_snapshot(asset_id: str) -> dict:
    data = _ctx()
    tx = data["transformers"].set_index("id").loc[asset_id].to_dict()
    op = data["operational_history"]
    latest_op = op[op["id"] == asset_id].sort_values("date").iloc[-1].to_dict()
    return {**tx, **{f"op_{k}": v for k, v in latest_op.items() if k != "id"}}


def _health(asset_id: str) -> dict[str, float]:
    b = _asset_bundle(asset_id)
    age = float(pd.Timestamp.now().year - _ctx()["transformers"].set_index("id").loc[asset_id]["vintage_year"])
    return compute_health_index(b["dga_history"], b["pd_history"], b["bushing_history"], b["maintenance_log"], age)


@lru_cache(maxsize=1)
def _build_risk_table() -> pd.DataFrame:
    tx = _ctx()["transformers"].copy()
    rows = []
    for aid in tx["id"]:
        p1 = get_asset_probability(aid, 1)
        p3 = get_asset_probability(aid, 3)
        p5 = get_asset_probability(aid, 5)
        p10 = get_asset_probability(aid, 10)
        consequence = float(tx[tx["id"] == aid]["failure_consequence_usd"].iloc[0])
        monetized = p1 * consequence
        repl_cost = float(tx[tx["id"] == aid]["mva_rating"].iloc[0] * 10_500)
        rows.append(
            {
                "id": aid,
                "voltage_class_kv": int(tx[tx["id"] == aid]["voltage_class_kv"].iloc[0]),
                "p_fail_1y": p1,
                "p_fail_3y": p3,
                "p_fail_5y": p5,
                "p_fail_10y": p10,
                "monetized_risk_usd": monetized,
                "replacement_cost_usd": repl_cost,
                "risk_reduction_per_usd": (monetized * 0.75) / max(repl_cost, 1.0),
            }
        )
    return pd.DataFrame(rows).sort_values("monetized_risk_usd", ascending=False)


@tool(args_schema=AssetInput)
def get_transformer(id: str) -> dict:
    """Get nameplate and latest operational snapshot for one transformer."""
    snap = _asset_snapshot(id)
    return {"id": id, "snapshot": snap}


@tool(args_schema=AssetInput)
def get_dga_diagnosis(id: str) -> dict:
    """Get latest DGA diagnosis and trend for a transformer."""
    dga = _asset_bundle(id)["dga_history"]
    diag = diagnose_latest(dga)
    return {
        "id": id,
        "fault_type": diag.fault_type,
        "severity": diag.severity,
        "trend": diag.trend,
        "ieee_condition": diag.ieee_condition,
        "ratios": diag.ratios,
    }


@tool(args_schema=AssetInput)
def get_health_index(id: str) -> dict:
    """Compute health index and component scores for a transformer."""
    return {"id": id, **_health(id)}


@tool(args_schema=FailureProbInput)
def get_failure_probability(id: str, horizon_years: int) -> dict:
    """Get calibrated failure probability for a transformer at a horizon."""
    p = get_asset_probability(id, horizon_years)
    all_horizons = {h: get_asset_probability(id, h) for h in (1, 3, 5, 10)}
    return {"id": id, "horizon_years": horizon_years, "p_failure": p, "curve": all_horizons}


@tool(args_schema=FailureProbInput)
def get_monetized_risk(id: str, horizon_years: int) -> dict:
    """Get monetized risk = P(failure)*consequence for a transformer."""
    tx = _ctx()["transformers"].set_index("id")
    consequence = float(tx.loc[id]["failure_consequence_usd"])
    p = get_asset_probability(id, horizon_years)
    return {"id": id, "horizon_years": horizon_years, "p_failure": p, "consequence_usd": consequence, "monetized_risk_usd": p * consequence}


@tool(args_schema=RankFleetInput)
def rank_fleet(top_n: int = 5, sort_by: str = "monetized_risk_usd") -> dict:
    """Rank fleet by selected risk metric."""
    risk = _build_risk_table()
    if sort_by not in risk.columns:
        sort_by = "monetized_risk_usd"
    out = risk.sort_values(sort_by, ascending=False).head(top_n)
    return {"top_n": top_n, "sort_by": sort_by, "rows": out.to_dict(orient="records")}


@tool(args_schema=WhatIfInput)
def simulate_what_if(id: str, loading_pct: float, defer_years: int, ambient_delta_c: float) -> dict:
    """Recompute risk under loading/temp/deferral what-if scenario."""
    base_features = get_asset_features(id).copy()
    baseline_p5 = get_asset_probability(id, 5)

    scenario = base_features.copy()
    scenario.loc[:, "avg_loading_pct"] = loading_pct
    scenario.loc[:, "ambient_temp_c"] = scenario["ambient_temp_c"] + ambient_delta_c
    scenario.loc[:, "health_index"] = scenario["health_index"] - defer_years * 2.4

    from src.models.failure_prob import load_models, predict_failure_probabilities

    model, aft = load_models()
    scenario_probs = predict_failure_probabilities(scenario, model=model, aft=aft)
    return {
        "id": id,
        "baseline_p5": baseline_p5,
        "scenario_p5": scenario_probs[5],
        "delta_p5": scenario_probs[5] - baseline_p5,
        "curve": scenario_probs,
        "inputs": {"loading_pct": loading_pct, "defer_years": defer_years, "ambient_delta_c": ambient_delta_c},
    }


@tool(args_schema=ReplacementPlanInput)
def recommend_replacement_plan(budget_musd: float, horizon_years: int = 5) -> dict:
    """Recommend budget-constrained replacement plan."""
    risk = _build_risk_table()
    plan = build_replacement_plan(risk, budget_musd, horizon_years)
    return {"budget_musd": budget_musd, "horizon_years": horizon_years, "projects": plan.to_dict(orient="records")}


@tool(args_schema=SpareInput)
def recommend_spare_strategy(voltage_class_kv: int) -> dict:
    """Recommend spares for a voltage class using Monte Carlo confidence intervals."""
    risk = _build_risk_table()
    return monte_carlo_spares(risk, voltage_class_kv)


@tool(args_schema=AssetInput)
def explain_transformer_risk(id: str) -> dict:
    """Generate narrative summary and top factors for transformer risk."""
    features = get_asset_features(id).iloc[0]
    p5 = get_asset_probability(id, 5)
    health = _health(id)["health_index"]
    factors = top_risk_factors(features)
    return {"id": id, "narrative": build_narrative(id, p5, health, factors), "top_factors": factors, "confidence": 0.82}


ALL_TOOLS = [
    get_transformer,
    get_dga_diagnosis,
    get_health_index,
    get_failure_probability,
    get_monetized_risk,
    rank_fleet,
    simulate_what_if,
    recommend_replacement_plan,
    recommend_spare_strategy,
    explain_transformer_risk,
]
