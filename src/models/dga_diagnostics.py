from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


@dataclass
class DGADiagnosis:
    fault_type: str
    severity: str
    trend: str
    ieee_condition: str
    ratios: dict[str, float]


class DuvalCoordinates(NamedTuple):
    ch4_pct: float
    c2h2_pct: float
    c2h4_pct: float


def _ratio(n: float, d: float) -> float:
    return float(n / max(d, 1e-6))


def duval_triangle_1(ch4: float, c2h2: float, c2h4: float) -> str:
    total = max(ch4 + c2h2 + c2h4, 1e-6)
    p_ch4 = 100 * ch4 / total
    p_c2h2 = 100 * c2h2 / total
    p_c2h4 = 100 * c2h4 / total

    if p_c2h2 >= 35 and p_c2h4 >= 20:
        return "D2"
    if p_c2h2 >= 20:
        return "D1"
    if p_c2h4 >= 65:
        return "T3"
    if p_c2h4 >= 45:
        return "T2"
    if p_c2h4 >= 25:
        return "T1"
    if p_ch4 >= 70:
        return "PD"
    return "DT"


def rogers_ratios(h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float) -> dict[str, float]:
    return {
        "r1_ch4_h2": _ratio(ch4, h2),
        "r2_c2h2_c2h4": _ratio(c2h2, c2h4),
        "r3_c2h4_c2h6": _ratio(c2h4, c2h6),
    }


def doernenburg_ratios(h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float) -> dict[str, float]:
    return {
        "dr1_ch4_h2": _ratio(ch4, h2),
        "dr2_c2h2_ch4": _ratio(c2h2, ch4),
        "dr3_c2h2_c2h4": _ratio(c2h2, c2h4),
        "dr4_c2h6_c2h2": _ratio(c2h6, c2h2),
    }


def ieee_c57_104_condition(h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float, co: float) -> str:
    tcg = h2 + ch4 + c2h2 + c2h4 + c2h6 + co
    if tcg < 720:
        return "Condition 1"
    if tcg < 1920:
        return "Condition 2"
    if tcg < 4630:
        return "Condition 3"
    return "Condition 4"


def diagnose_latest(dga_asset_history: pd.DataFrame) -> DGADiagnosis:
    df = dga_asset_history.sort_values("date")
    latest = df.iloc[-1]
    prev = df.iloc[-4] if len(df) >= 4 else df.iloc[0]

    fault = duval_triangle_1(latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"])
    ratios = rogers_ratios(latest["h2_ppm"], latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"], latest["c2h6_ppm"])
    ratios.update(doernenburg_ratios(latest["h2_ppm"], latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"], latest["c2h6_ppm"]))
    condition = ieee_c57_104_condition(
        latest["h2_ppm"], latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"], latest["c2h6_ppm"], latest["co_ppm"]
    )

    growth = (latest["h2_ppm"] + latest["c2h2_ppm"] + latest["c2h4_ppm"]) - (
        prev["h2_ppm"] + prev["c2h2_ppm"] + prev["c2h4_ppm"]
    )
    trend = "increasing" if growth > 15 else "stable" if growth > -10 else "decreasing"

    severity_map = {"Condition 1": "low", "Condition 2": "moderate", "Condition 3": "high", "Condition 4": "critical"}
    severity = severity_map[condition]

    if fault in {"D2", "T3"} and severity in {"high", "critical"}:
        severity = "critical"

    return DGADiagnosis(fault_type=fault, severity=severity, trend=trend, ieee_condition=condition, ratios=ratios)


def duval_coordinates(ch4: float, c2h2: float, c2h4: float) -> DuvalCoordinates:
    total = max(ch4 + c2h2 + c2h4, 1e-9)
    vals = 100 * np.array([ch4, c2h2, c2h4]) / total
    return DuvalCoordinates(float(vals[0]), float(vals[1]), float(vals[2]))
