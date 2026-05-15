import pandas as pd

from src.agents.tools import rank_fleet
from src.models.fleet_optimizer import build_replacement_plan


def test_optimizer_respects_budget():
    ranked = pd.DataFrame(rank_fleet.invoke({"top_n": 50, "sort_by": "monetized_risk_usd"})["rows"])
    plan = build_replacement_plan(ranked, budget_musd=5, horizon_years=5)
    assert plan["replacement_cost_usd"].sum() <= 5_000_000 + 1e-6
