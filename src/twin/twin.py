"""``TransformerTwin`` — orchestrator that wires the thermal / aging / gas
simulators to a single asset and produces a :class:`TwinState` snapshot.

The twin's main entry point is :meth:`TransformerTwin.run`, which takes a
:class:`ScenarioInput`, integrates the thermal trajectory, and computes the
projected failure-probability uplift by feeding the scenario's peak hotspot
back into the existing :func:`predict_failure_probabilities` (with the
IEEE C57.91 aging multiplier already baked in there).

All fields on ``TwinState`` are JSON-serialisable scalars or short DataFrames
so the result can be passed straight to the Streamlit UI, the agent tool, or
the Azure Digital Twins mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import REFERENCE_YEAR
from src.data.loader import load_seed_data
from src.models.failure_prob import (
    OT_FEATURE_DEFAULTS,
    get_asset_features,
    load_models,
    predict_failure_probabilities,
)
from src.models.health_index import compute_health_index
from src.twin.aging import AgingTwin, aging_acceleration_factor, hours_to_years
from src.twin.calibration import CalibrationResult, calibrate_thermal_twin
from src.twin.gas_evolution import GasEvolutionTwin
from src.twin.scenarios import ScenarioInput, build_scenario
from src.twin.thermal import ThermalCoefficients, ThermalTwin, default_coefficients_for


@dataclass(frozen=True)
class TwinState:
    asset_id: str
    scenario_name: str
    scenario_hours: int
    peak_hotspot_c: float
    peak_top_oil_c: float
    avg_hotspot_c: float
    aging_hours_equivalent: float
    per_unit_life_consumed_event: float
    rul_years_steady_state: float
    rul_years_with_scenario: float
    gas_accumulated_h2_ppm: float
    gas_accumulated_ch4_ppm: float
    gas_accumulated_c2h4_ppm: float
    baseline_p_failure_5y: float
    projected_p_failure_5y: float
    health_index: float
    calibration_rmse_c: float
    calibration_samples: int
    coefficients: ThermalCoefficients
    trajectory: pd.DataFrame = field(repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact dict for chat tools / ADT properties (no DataFrame)."""
        return {
            "asset_id": self.asset_id,
            "scenario_name": self.scenario_name,
            "scenario_hours": self.scenario_hours,
            "peak_hotspot_c": round(self.peak_hotspot_c, 2),
            "peak_top_oil_c": round(self.peak_top_oil_c, 2),
            "avg_hotspot_c": round(self.avg_hotspot_c, 2),
            "aging_hours_equivalent": round(self.aging_hours_equivalent, 2),
            "per_unit_life_consumed_event": round(self.per_unit_life_consumed_event, 6),
            "rul_years_steady_state": round(self.rul_years_steady_state, 2),
            "rul_years_with_scenario": round(self.rul_years_with_scenario, 2),
            "rul_delta_years": round(self.rul_years_steady_state - self.rul_years_with_scenario, 2),
            "gas_h2_ppm": round(self.gas_accumulated_h2_ppm, 3),
            "gas_ch4_ppm": round(self.gas_accumulated_ch4_ppm, 3),
            "gas_c2h4_ppm": round(self.gas_accumulated_c2h4_ppm, 4),
            "baseline_p_failure_5y": round(self.baseline_p_failure_5y, 4),
            "projected_p_failure_5y": round(self.projected_p_failure_5y, 4),
            "projected_p_failure_delta": round(self.projected_p_failure_5y - self.baseline_p_failure_5y, 4),
            "health_index": round(self.health_index, 1),
            "calibration_rmse_c": round(self.calibration_rmse_c, 2) if self.calibration_rmse_c == self.calibration_rmse_c else None,
            "calibration_samples": self.calibration_samples,
        }


class TransformerTwin:
    """Physics digital twin bound to a specific transformer."""

    def __init__(
        self,
        asset_id: str,
        thermal: ThermalTwin,
        aging: AgingTwin,
        gas: GasEvolutionTwin,
        calibration: CalibrationResult,
        asset_row: pd.Series,
        ot_thermal_latest: pd.Series | None,
    ) -> None:
        self.asset_id = asset_id
        self.thermal = thermal
        self.aging = aging
        self.gas = gas
        self.calibration = calibration
        self.asset_row = asset_row
        self.ot_thermal_latest = ot_thermal_latest

    @classmethod
    def from_asset_id(cls, asset_id: str) -> "TransformerTwin":
        data = load_seed_data()
        tx = data["transformers"]
        if not (tx["id"] == asset_id).any():
            raise KeyError(f"Unknown asset_id: {asset_id}")
        row = tx[tx["id"] == asset_id].iloc[0]
        calib = calibrate_thermal_twin(asset_id)
        if not np.isnan(calib.rmse_c):
            coef = calib.coefficients
        else:
            coef = default_coefficients_for(row.get("cooling_type"))
        thermal = ThermalTwin(coefficients=coef)
        latest = None
        therm_df = data.get("thermal_telemetry")
        if therm_df is not None and not therm_df.empty:
            latest_rows = therm_df[therm_df["id"] == asset_id].sort_values("date")
            if not latest_rows.empty:
                latest = latest_rows.iloc[-1]
        return cls(
            asset_id=asset_id,
            thermal=thermal,
            aging=AgingTwin(),
            gas=GasEvolutionTwin(),
            calibration=calib,
            asset_row=row,
            ot_thermal_latest=latest,
        )

    def estimate_consumed_life_pu(self) -> float:
        """Approximate per-unit life consumed to date.

        Uses age × baseline F_AA at the asset's typical hot-spot. This is a
        first-order estimate — sufficient for RUL deltas in the demo.
        """
        install_year = float(self.asset_row.get("vintage_year", REFERENCE_YEAR))
        age_years = max(REFERENCE_YEAR - install_year, 0.0)
        baseline_hotspot = (
            float(self.ot_thermal_latest["hotspot_max_c"])
            if self.ot_thermal_latest is not None
            else 95.0
        )
        faa = float(aging_acceleration_factor(baseline_hotspot))
        consumed_hours = age_years * 8766.0 * faa
        return float(np.clip(consumed_hours / self.aging.normal_life_hours, 0.0, 0.95))

    def run(self, scenario: ScenarioInput | str) -> TwinState:
        if isinstance(scenario, str):
            scenario = build_scenario(scenario)
        traj = self.thermal.simulate(
            load_profile=scenario.load_pu,
            ambient_profile=scenario.ambient_c,
            cooling_profile=scenario.cooling_pct,
        )
        aging_res = self.aging.integrate(traj["hotspot_c"].to_numpy(), traj["t_hours"].to_numpy())
        gas_acc = self.gas.accumulate(traj["hotspot_c"].to_numpy(), traj["t_hours"].to_numpy())

        consumed_pu = self.estimate_consumed_life_pu()
        steady_hs = float(traj["hotspot_c"].mean())
        rul_steady = self.aging.remaining_life_years(consumed_pu, steady_hs)
        rul_scen = self.aging.rul_with_scenario(
            consumed_pu,
            traj["hotspot_c"].to_numpy(),
            traj["t_hours"].to_numpy(),
            recurrence_per_year=12.0,
        )

        baseline_p5, projected_p5 = self._predict_failure(traj)
        hi = self._fetch_health_index()

        return TwinState(
            asset_id=self.asset_id,
            scenario_name=scenario.name,
            scenario_hours=int(traj["t_hours"].iloc[-1]),
            peak_hotspot_c=float(traj["hotspot_c"].max()),
            peak_top_oil_c=float(traj["top_oil_c"].max()),
            avg_hotspot_c=float(traj["hotspot_c"].mean()),
            aging_hours_equivalent=aging_res.aging_hours_equivalent,
            per_unit_life_consumed_event=aging_res.per_unit_life_consumed,
            rul_years_steady_state=rul_steady,
            rul_years_with_scenario=rul_scen,
            gas_accumulated_h2_ppm=gas_acc.h2_ppm,
            gas_accumulated_ch4_ppm=gas_acc.ch4_ppm,
            gas_accumulated_c2h4_ppm=gas_acc.c2h4_ppm,
            baseline_p_failure_5y=baseline_p5,
            projected_p_failure_5y=projected_p5,
            health_index=hi,
            calibration_rmse_c=self.calibration.rmse_c,
            calibration_samples=self.calibration.samples,
            coefficients=self.thermal.coefficients,
            trajectory=traj,
        )

    def _predict_failure(self, trajectory: pd.DataFrame) -> tuple[float, float]:
        model, aft = load_models()
        feats = get_asset_features(self.asset_id)
        baseline = predict_failure_probabilities(feats, model=model, aft=aft)
        bumped = feats.copy()
        bumped.loc[:, "ot_hotspot_max_c"] = float(trajectory["hotspot_c"].max())
        hours_over = float((trajectory["hotspot_c"] > 110.0).sum())
        bumped.loc[:, "ot_hours_above_110c"] = max(
            float(bumped["ot_hours_above_110c"].iloc[0]), hours_over
        )
        projected = predict_failure_probabilities(bumped, model=model, aft=aft)
        return float(baseline[5]), float(projected[5])

    def _fetch_health_index(self) -> float:
        data = load_seed_data()
        age = float(REFERENCE_YEAR - float(self.asset_row.get("vintage_year", REFERENCE_YEAR)))
        try:
            hi = compute_health_index(
                data["dga_history"][data["dga_history"]["id"] == self.asset_id],
                data["pd_history"][data["pd_history"]["id"] == self.asset_id],
                data["bushing_history"][data["bushing_history"]["id"] == self.asset_id],
                data["maintenance_log"][data["maintenance_log"]["id"] == self.asset_id],
                age,
            )
        except Exception:
            return 60.0
        if isinstance(hi, dict):
            return float(hi.get("health_index", 60.0))
        return float(hi)

    def _compose_health_inputs(self, trajectory: pd.DataFrame) -> dict[str, float]:
        peak_hs = float(trajectory["hotspot_c"].max())
        ot_defaults = dict(OT_FEATURE_DEFAULTS)
        return {
            "age_years": max(REFERENCE_YEAR - float(self.asset_row.get("vintage_year", REFERENCE_YEAR)), 0.0),
            "dga_severity": 1.0 + max(peak_hs - 110.0, 0.0) / 20.0,
            "bushing_tan_delta": 0.5,
            "pd_pc": 250.0,
            "loading_avg": 90.0,
            "moisture_pct": ot_defaults.get("ot_h2o_ppm", 10.0) / 1000.0,
        }
