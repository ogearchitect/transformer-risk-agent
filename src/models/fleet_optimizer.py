from __future__ import annotations

import numpy as np
import pandas as pd


def build_replacement_plan(risk_table: pd.DataFrame, budget_musd: float, horizon_years: int) -> pd.DataFrame:
    budget = budget_musd * 1_000_000
    df = risk_table.copy().sort_values("risk_reduction_per_usd", ascending=False)
    selected = []
    spent = 0.0

    for _, row in df.iterrows():
        cost = float(row["replacement_cost_usd"])
        if spent + cost <= budget:
            spent += cost
            selected.append(
                {
                    "id": row["id"],
                    "voltage_class_kv": row["voltage_class_kv"],
                    "replacement_year": int(np.ceil(row["p_fail_5y"] * horizon_years)),
                    "replacement_cost_usd": cost,
                    "risk_reduction_usd": float(row["monetized_risk_usd"] * 0.75),
                    "cumulative_spend_usd": spent,
                }
            )
    return pd.DataFrame(selected)


def monte_carlo_spares(risk_table: pd.DataFrame, voltage_class_kv: int, trials: int = 10_000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    subset = risk_table[risk_table["voltage_class_kv"] == voltage_class_kv]
    if subset.empty:
        return {"voltage_class_kv": voltage_class_kv, "recommended_spares": 0, "ci_low": 0, "ci_high": 0}

    probs = subset["p_fail_1y"].clip(0, 1).to_numpy()
    outcomes = rng.binomial(1, probs, size=(trials, len(probs))).sum(axis=1)
    return {
        "voltage_class_kv": voltage_class_kv,
        "recommended_spares": int(np.percentile(outcomes, 85)),
        "ci_low": float(np.percentile(outcomes, 5)),
        "ci_high": float(np.percentile(outcomes, 95)),
    }
