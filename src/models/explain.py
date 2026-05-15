from __future__ import annotations

import pandas as pd


FACTOR_WEIGHTS = {
    "age_years": 0.22,
    "avg_loading_pct": 0.18,
    "health_index": -0.35,
    "through_faults": 0.15,
    "ambient_temp_c": 0.10,
}


def top_risk_factors(feature_row: pd.Series, top_k: int = 3) -> list[dict[str, float]]:
    contribs = []
    for key, w in FACTOR_WEIGHTS.items():
        value = float(feature_row[key])
        baseline = 50 if key == "health_index" else (70 if key == "avg_loading_pct" else 20)
        score = w * (value - baseline)
        contribs.append({"factor": key, "contribution": round(score, 3), "value": round(value, 3)})
    contribs.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contribs[:top_k]


def build_narrative(asset_id: str, p_fail: float, health_index: float, factors: list[dict[str, float]]) -> str:
    badge = "high" if p_fail > 0.35 else "moderate" if p_fail > 0.18 else "low"
    top = ", ".join(f"{f['factor']} ({f['value']})" for f in factors)
    return (
        f"Transformer {asset_id} currently shows {badge} near-term risk with a health index of {health_index:.1f}. "
        f"Key contributors are {top}. Recommended action: prioritize inspection and targeted maintenance within the next cycle."
    )
