"""Calibration of per-asset thermal twin coefficients to historical OT data.

For each asset we fit ``(R, n)`` of the IEEE C57.91 top-oil rise equation by
least-squares against the monthly ``thermal_telemetry`` table joined with
``operational_history`` (for the load average that drove the temperature).

Other coefficients keep their ONAF defaults — calibrating all six on 12 monthly
samples would over-fit. ``CalibrationResult`` includes the RMSE so the UI can
honestly show "twin error: ±X °C".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.loader import load_seed_data
from src.twin.thermal import (
    COOLING_TYPE_DEFAULTS,
    ThermalCoefficients,
    _cooling_derate,
    _top_oil_steady_state,
    default_coefficients_for,
)


@dataclass(frozen=True)
class CalibrationResult:
    asset_id: str
    coefficients: ThermalCoefficients
    rmse_c: float
    samples: int
    note: str = ""


def _calibrate_one(
    thermal_df: pd.DataFrame,
    op_df: pd.DataFrame,
    cooling_df: pd.DataFrame,
    base: ThermalCoefficients,
) -> CalibrationResult:
    asset_id = str(thermal_df["id"].iloc[0])
    if thermal_df.empty or op_df.empty:
        return CalibrationResult(asset_id, base, float("nan"), 0, "no data")

    merged = thermal_df[["date", "top_oil_max_c", "top_oil_avg_c"]].merge(
        op_df[["date", "avg_loading_pct", "ambient_temp_c"]],
        on="date", how="inner",
    )
    if not cooling_df.empty:
        merged = merged.merge(
            cooling_df[["date", "availability_pct"]],
            on="date", how="left",
        )
        merged["availability_pct"] = merged["availability_pct"].fillna(100.0)
    else:
        merged["availability_pct"] = 100.0

    merged = merged.dropna().tail(12)
    if len(merged) < 4:
        return CalibrationResult(asset_id, base, float("nan"), int(len(merged)), "<4 samples")

    K = (merged["avg_loading_pct"].to_numpy(dtype=float) / 100.0)
    amb = merged["ambient_temp_c"].to_numpy(dtype=float)
    cool = merged["availability_pct"].to_numpy(dtype=float)
    measured_top_oil = merged["top_oil_max_c"].to_numpy(dtype=float)

    def residual(params: np.ndarray) -> float:
        R, n, theta_rated = float(params[0]), float(params[1]), float(params[2])
        coef = ThermalCoefficients(
            theta_to_rated_c=float(np.clip(theta_rated, 25.0, 75.0)),
            delta_theta_hs_rated_c=base.delta_theta_hs_rated_c,
            R=float(np.clip(R, 1.5, 12.0)),
            n=float(np.clip(n, 0.6, 1.1)),
            m=base.m,
            tau_to_min=base.tau_to_min,
            tau_w_min=base.tau_w_min,
            cooling_loss_factor=base.cooling_loss_factor,
        )
        pred = np.array(
            [
                amb[i] + _top_oil_steady_state(K[i], coef) * _cooling_derate(cool[i], coef)
                for i in range(len(K))
            ]
        )
        return float(np.mean((pred - measured_top_oil) ** 2))

    try:
        res = minimize(
            residual,
            x0=np.array([base.R, base.n, base.theta_to_rated_c]),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 300},
        )
        R_fit, n_fit, theta_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])
        rmse = float(np.sqrt(res.fun))
    except Exception:  # pragma: no cover
        return CalibrationResult(asset_id, base, float("nan"), len(merged), "fit failed")

    coef = ThermalCoefficients(
        theta_to_rated_c=float(np.clip(theta_fit, 25.0, 75.0)),
        delta_theta_hs_rated_c=base.delta_theta_hs_rated_c,
        R=float(np.clip(R_fit, 1.5, 12.0)),
        n=float(np.clip(n_fit, 0.6, 1.1)),
        m=base.m,
        tau_to_min=base.tau_to_min,
        tau_w_min=base.tau_w_min,
        cooling_loss_factor=base.cooling_loss_factor,
    )
    return CalibrationResult(asset_id, coef, rmse, int(len(merged)), "ok")


@lru_cache(maxsize=512)
def calibrate_thermal_twin(asset_id: str) -> CalibrationResult:
    """Fit thermal coefficients to the last ~12 months of OT data for this asset."""
    data = load_seed_data()
    thermal = data.get("thermal_telemetry", pd.DataFrame())
    op = data.get("operational_history", pd.DataFrame())
    cooling = data.get("cooling_status", pd.DataFrame())
    transformers = data.get("transformers", pd.DataFrame())
    base = default_coefficients_for(
        transformers.loc[transformers["id"] == asset_id, "cooling_type"].iloc[0]
        if not transformers.empty and (transformers["id"] == asset_id).any()
        else None
    )
    if thermal.empty or op.empty:
        return CalibrationResult(asset_id, base, float("nan"), 0, "no OT data")
    t = thermal[thermal["id"] == asset_id].sort_values("date")
    o = op[op["id"] == asset_id].sort_values("date")
    c = cooling[cooling["id"] == asset_id].sort_values("date") if not cooling.empty else cooling
    return _calibrate_one(t, o, c, base)
