from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dga_diagnostics import diagnose_latest


def _bounded_score(value: float, good: float, bad: float, invert: bool = False) -> float:
    if good == bad:
        return 50.0
    ratio = (value - good) / (bad - good)
    score = 100 * (1 - ratio)
    if invert:
        score = 100 - score
    return float(np.clip(score, 0, 100))


def compute_health_index(
    dga_asset: pd.DataFrame,
    pd_asset: pd.DataFrame,
    bushing_asset: pd.DataFrame,
    maintenance_asset: pd.DataFrame,
    age_years: float,
) -> dict[str, float]:
    diagnosis = diagnose_latest(dga_asset)
    dga_score_map = {"low": 92, "moderate": 75, "high": 45, "critical": 20}
    dga_score = dga_score_map[diagnosis.severity]

    pd_latest = float(pd_asset.sort_values("date").iloc[-1]["pd_pc"])
    pd_score = _bounded_score(pd_latest, good=100, bad=1200)

    bushing_latest = bushing_asset.sort_values("date").groupby("phase").tail(1)
    tan_delta = float(bushing_latest["tan_delta_pct"].mean())
    ir = float(bushing_latest["ir_gohm"].mean())
    bushing_score = 0.65 * _bounded_score(tan_delta, good=0.3, bad=2.5) + 0.35 * _bounded_score(ir, good=2500, bad=100, invert=True)

    age_score = _bounded_score(age_years, good=5, bad=70)

    compliance = maintenance_asset[maintenance_asset["event_type"] != "deferral"]["effectiveness"].tail(8).mean()
    compliance = float(compliance) if not np.isnan(compliance) else 0.7
    maintenance_score = float(np.clip(compliance * 100, 0, 100))

    weights = {
        "dga": 0.3,
        "pd": 0.2,
        "bushing": 0.2,
        "age": 0.15,
        "maintenance": 0.15,
    }
    total = (
        weights["dga"] * dga_score
        + weights["pd"] * pd_score
        + weights["bushing"] * bushing_score
        + weights["age"] * age_score
        + weights["maintenance"] * maintenance_score
    )

    return {
        "health_index": float(np.clip(total, 0, 100)),
        "dga_score": float(dga_score),
        "pd_score": float(pd_score),
        "bushing_score": float(bushing_score),
        "age_score": float(age_score),
        "maintenance_score": float(maintenance_score),
    }
