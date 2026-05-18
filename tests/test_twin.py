from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.twin import (
    AgingTwin,
    GasEvolutionTwin,
    ScenarioInput,
    ThermalCoefficients,
    ThermalTwin,
    TransformerTwin,
    build_scenario,
)
from src.twin.adt_sync import NullMirror, adt_mirror, reset_mirror_for_tests
from src.twin.aging import aging_acceleration_factor


def test_thermal_steady_state_load_one_pu_at_30c() -> None:
    twin = ThermalTwin()
    top_oil, hotspot = twin.steady_state(load_pu=1.0, ambient_c=30.0, cooling_pct=100.0)
    assert 70.0 <= top_oil <= 90.0
    assert 100.0 <= hotspot <= 130.0


def test_thermal_simulate_step_overload_rises_monotonically_then_plateaus() -> None:
    twin = ThermalTwin()
    load = np.concatenate([np.full(2, 0.85), np.full(10, 1.30), np.full(12, 0.85)])
    amb = np.full(24, 30.0)
    traj = twin.simulate(load, amb)
    rise = traj[(traj["t_hours"] >= 2.0) & (traj["t_hours"] <= 6.0)]
    assert rise["hotspot_c"].iloc[-1] > rise["hotspot_c"].iloc[0]
    assert float(traj["hotspot_c"].max()) < 180.0


def test_thermal_simulate_validates_shape_match() -> None:
    twin = ThermalTwin()
    with pytest.raises(ValueError):
        twin.simulate([1.0, 1.0], [30.0])


def test_aging_doubles_per_six_degrees() -> None:
    aging = AgingTwin()
    hours = 8.0
    t = np.linspace(0.0, hours, 100)
    flat_110 = np.full_like(t, 110.0)
    flat_116 = np.full_like(t, 116.0)
    a110 = aging.integrate(flat_110, t)
    a116 = aging.integrate(flat_116, t)
    assert a110.aging_hours_equivalent == pytest.approx(8.0, rel=0.02)
    assert a116.aging_hours_equivalent / a110.aging_hours_equivalent == pytest.approx(2.0, rel=0.05)


def test_aging_remaining_life_collapses_at_high_hotspot() -> None:
    aging = AgingTwin()
    short = aging.remaining_life_years(0.10, 140.0)
    long = aging.remaining_life_years(0.10, 100.0)
    assert short < long
    assert short > 0.0


def test_aging_rul_with_scenario_strictly_lt_baseline_for_stressful_scenario() -> None:
    aging = AgingTwin()
    consumed = 0.20
    rul_baseline = aging.remaining_life_years(consumed, 95.0)
    t = np.linspace(0.0, 24.0, 200)
    hot = 95.0 + 50.0 * np.exp(-((t - 12.0) ** 2) / 4.0)
    rul_with = aging.rul_with_scenario(consumed, hot, t, recurrence_per_year=12.0, baseline_hotspot_c=95.0)
    assert rul_with < rul_baseline


def test_gas_h2_monotonic_in_hotspot() -> None:
    gas = GasEvolutionTwin()
    rates = gas.rates_ppm_per_hour(np.array([100.0, 130.0, 150.0, 170.0]))
    assert rates["h2"][0] == 0.0
    assert rates["h2"][1] > 0.0
    assert rates["h2"][2] > rates["h2"][1]
    assert rates["h2"][3] > rates["h2"][2]


def test_gas_zero_below_all_onsets() -> None:
    gas = GasEvolutionTwin()
    g = gas.accumulate(np.full(24, 80.0), np.arange(24, dtype=float))
    assert g.h2_ppm == 0.0
    assert g.ch4_ppm == 0.0
    assert g.c2h4_ppm == 0.0


def test_scenario_library_has_required_presets() -> None:
    for name in ("normal_day", "summer_peak", "heat_wave", "contingency_transfer", "cooling_loss"):
        s = build_scenario(name)
        assert isinstance(s, ScenarioInput)
        assert s.load_pu.size == s.ambient_c.size == s.cooling_pct.size
        assert s.load_pu.size >= 24


def test_scenario_custom_with_repeat_days() -> None:
    s = build_scenario("custom", load_pu=[1.0] * 24, ambient_c=[25.0] * 24, repeat_days=3)
    assert s.load_pu.size == 72


def test_calibrate_thermal_twin_runs_for_real_asset() -> None:
    from src.twin.calibration import calibrate_thermal_twin

    res = calibrate_thermal_twin("T-0001")
    assert res.asset_id == "T-0001"
    assert res.samples >= 4
    assert res.rmse_c == res.rmse_c
    assert res.rmse_c < 15.0


def test_transformer_twin_run_returns_complete_state() -> None:
    twin = TransformerTwin.from_asset_id("T-0042")
    state = twin.run("heat_wave")
    summary = state.to_summary_dict()
    for key in (
        "peak_hotspot_c", "aging_hours_equivalent", "per_unit_life_consumed_event",
        "rul_years_steady_state", "rul_years_with_scenario",
        "gas_h2_ppm", "baseline_p_failure_5y", "projected_p_failure_5y",
        "health_index", "calibration_rmse_c",
    ):
        assert key in summary
    assert summary["peak_hotspot_c"] > 110.0
    assert summary["aging_hours_equivalent"] > 0.0


def test_simulate_twin_scenario_tool_dict_shape() -> None:
    from src.agents.tools import simulate_twin_scenario

    res = simulate_twin_scenario.invoke(
        {"id": "T-0001", "scenario": "summer_peak", "repeat_days": 1, "sync_to_adt": False}
    )
    assert res["id"] == "T-0001"
    assert res["scenario"] == "summer_peak"
    assert "peak_hotspot_c" in res
    assert "adt_sync" in res
    assert res["adt_sync"]["skipped"] is True


def test_adt_mirror_falls_back_to_null_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADT_ENDPOINT", raising=False)
    reset_mirror_for_tests()
    mirror = adt_mirror()
    assert isinstance(mirror, NullMirror)
    assert mirror.enabled is False
    result = mirror.upsert_twin("T-0001", {"hotspotC": 120.0})
    assert result.get("skipped") is True
    assert mirror.get_twin("T-0001") is None
    assert mirror.query("SELECT * FROM digitaltwins") == []


def test_orchestrator_routes_twin_keyword_to_tool() -> None:
    from src.agents.orchestrator import handle_query

    response = handle_query("simulate a heat wave on T-0042")
    tools = [c["tool"] for c in response.tool_calls]
    assert "simulate_twin_scenario" in tools


def test_dtdl_model_is_valid_json_with_expected_properties() -> None:
    import json
    from pathlib import Path

    model = json.loads(Path("dtdl/Transformer.v1.json").read_text())
    assert model["@id"] == "dtmi:tra:Transformer;1"
    property_names = {c["name"] for c in model["contents"] if c["@type"] == "Property"}
    for required in ("assetId", "hotspotC", "topOilC", "rulYears", "lastScenarioName"):
        assert required in property_names
