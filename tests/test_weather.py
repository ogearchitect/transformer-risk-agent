from __future__ import annotations

import json
from unittest import mock

import pytest

from src.data import weather as weather_mod
from src.data.weather import (
    DailyForecast,
    WeatherForecast,
    coordinates_for_asset,
    fetch_forecast,
    forecast_for_asset,
    heat_risk_summary,
    reset_weather_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _clear_weather_cache() -> None:
    reset_weather_cache_for_tests()
    yield
    reset_weather_cache_for_tests()


def test_coordinates_for_known_region() -> None:
    lat, lon, name, region = coordinates_for_asset("T-0001")
    assert -90 <= lat <= 90 and -180 <= lon <= 180
    assert region in {"North", "South", "East", "West", "Central"}
    assert "SS-" in name


def test_coordinates_for_unknown_asset_uses_default() -> None:
    lat, lon, name, region = coordinates_for_asset("T-9999")
    assert region == "Unknown"
    assert abs(lat - 39.83) < 0.01


def test_offline_fallback_when_http_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):  # noqa: ANN001
        raise OSError("simulated network failure")

    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", _boom)
    fc = fetch_forecast(44.97, -93.27, location_name="Test", region="North")
    assert fc.source == "offline-fallback"
    assert len(fc.days) == 7
    for d in fc.days:
        assert d.temperature_max_c >= d.temperature_min_c
        assert 0 <= d.humidity_max_pct <= 100


def test_open_meteo_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "daily": {
            "time": [f"2026-05-{18 + i:02d}" for i in range(7)],
            "temperature_2m_max": [20.0, 22.0, 28.0, 35.0, 39.0, 30.0, 24.0],
            "temperature_2m_min": [10.0, 12.0, 15.0, 20.0, 22.0, 18.0, 14.0],
            "relative_humidity_2m_max": [60, 70, 65, 55, 50, 70, 80],
            "wind_speed_10m_max": [10, 12, 8, 18, 22, 15, 11],
            "weather_code": [0, 1, 2, 3, 95, 61, 0],
        }
    }

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()

    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    fc = fetch_forecast(44.97, -93.27, location_name="Test", region="North")
    assert fc.source == "open-meteo"
    assert len(fc.days) == 7
    assert fc.days[4].temperature_max_c == 39.0
    assert fc.days[4].description.lower().startswith("thunderstorm")


def test_heat_risk_summary_categories() -> None:
    mild = WeatherForecast(
        region="X", location_name="x", latitude=0, longitude=0, source="t",
        days=[DailyForecast(f"d{i}", 18.0, 5.0, 60.0, 10.0, 0, "Clear") for i in range(7)],
    )
    advisory = WeatherForecast(
        region="X", location_name="x", latitude=0, longitude=0, source="t",
        days=[DailyForecast(f"d{i}", 34.0, 22.0, 50.0, 10.0, 1, "Mainly clear") for i in range(7)],
    )
    extreme = WeatherForecast(
        region="X", location_name="x", latitude=0, longitude=0, source="t",
        days=[DailyForecast(f"d{i}", 40.0, 26.0, 40.0, 10.0, 1, "Mainly clear") for i in range(7)],
    )
    assert "Mild" in heat_risk_summary(mild)["alert"]
    assert "Heat advisory" in heat_risk_summary(advisory)["alert"]
    assert "Extreme" in heat_risk_summary(extreme)["alert"]
    assert heat_risk_summary(advisory)["hot_days"] == 7


def test_cache_returns_same_object_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def _fake_open(*a, **k):  # noqa: ANN001
        call_count["n"] += 1

        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({
                    "daily": {
                        "time": ["2026-05-18"],
                        "temperature_2m_max": [25.0],
                        "temperature_2m_min": [12.0],
                        "relative_humidity_2m_max": [60],
                        "wind_speed_10m_max": [10],
                        "weather_code": [0],
                    }
                }).encode()
        return R()

    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", _fake_open)
    a = fetch_forecast(10.0, 10.0)
    b = fetch_forecast(10.0, 10.0)
    assert a is b
    assert call_count["n"] == 1


def test_forecast_for_asset_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    fc = forecast_for_asset("T-0042")
    df = fc.as_dataframe()
    assert list(df.columns)[:3] == ["date", "temperature_max_c", "temperature_min_c"]
    assert len(df) == 7


def test_weather_tool_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    reset_weather_cache_for_tests()
    from src.agents.tools import get_weather_forecast

    result = get_weather_forecast.invoke({"id": "T-0001", "heat_threshold_c": 32.0})
    for key in ("id", "region", "location", "latitude", "longitude", "source", "days", "heat_risk"):
        assert key in result
    assert result["source"] in {"open-meteo", "offline-fallback"}
    assert len(result["days"]) == 7
    assert "alert" in result["heat_risk"]


def test_orchestrator_routes_weather_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    reset_weather_cache_for_tests()
    from src.agents.orchestrator import handle_query

    response = handle_query("what is the weather forecast for T-0042?")
    tools = [c["tool"] for c in response.tool_calls]
    assert "get_weather_forecast" in tools


def test_orchestrator_heat_wave_still_routes_to_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather_mod.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    reset_weather_cache_for_tests()
    from src.agents.orchestrator import handle_query

    response = handle_query("simulate a heat wave on T-0042")
    tools = [c["tool"] for c in response.tool_calls]
    assert "simulate_twin_scenario" in tools
    assert "get_weather_forecast" not in tools


def test_orchestrator_survives_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the LLM backend throws, handle_query must still return a usable answer."""
    from src.agents import orchestrator as orch

    class _BoomLLM:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("simulated AOAI rate limit")

    monkeypatch.setattr(orch, "get_llm", lambda: _BoomLLM())
    resp = orch.handle_query("Top 3 risk")
    assert resp.tool_calls, "tools should still have run before LLM was called"
    assert "unavailable" in resp.answer.lower() or "deterministic" in resp.answer.lower()
