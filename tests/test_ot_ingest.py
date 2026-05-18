from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_seed_data
from src.data.ot_ingest import (
    MQTTSparkplugSource,
    OPCUASource,
    OSIsoftPIWebAPISource,
    OTData,
    SyntheticOTSource,
    generate_ot_data,
)
from src.models.failure_prob import (
    FEATURES,
    OT_FEATURE_DEFAULTS,
    _hotspot_aging_multiplier,
    build_training_frame,
    get_asset_features,
)


@pytest.fixture(scope="module")
def fleet() -> dict:
    return load_seed_data()


def test_synthetic_ot_source_emits_expected_tables(fleet: dict) -> None:
    src = SyntheticOTSource(seed=42)
    out = src.fetch(fleet["transformers"], fleet["operational_history"])
    assert isinstance(out, OTData)
    for name in ("thermal_telemetry", "online_dga", "oltc_telemetry", "cooling_status", "fault_events"):
        df = getattr(out, name)
        assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
        assert not df.empty, f"{name} is empty"
        assert {"id", "date"}.issubset(df.columns), f"{name} missing id/date"


def test_thermal_telemetry_schema_and_ranges(fleet: dict) -> None:
    out = SyntheticOTSource(seed=42).fetch(fleet["transformers"], fleet["operational_history"])
    t = out.thermal_telemetry
    required = {"id", "date", "top_oil_max_c", "top_oil_avg_c", "hotspot_max_c", "hotspot_avg_c", "hours_above_110c"}
    assert required.issubset(t.columns)
    assert (t["top_oil_max_c"].between(20, 200)).all()
    assert (t["hotspot_max_c"].between(20, 220)).all()
    assert (t["hours_above_110c"] >= 0).all()


def test_stress_factor_shifts_signals_upward(fleet: dict) -> None:
    src = SyntheticOTSource(seed=42)
    ids = sorted(fleet["transformers"]["id"].tolist())
    low = pd.Series(np.zeros(len(ids)), index=ids)
    high = pd.Series(np.ones(len(ids)), index=ids)
    low_out = src.fetch(fleet["transformers"], fleet["operational_history"], stress_factor=low)
    high_out = src.fetch(fleet["transformers"], fleet["operational_history"], stress_factor=high)
    assert high_out.thermal_telemetry["hotspot_max_c"].mean() > low_out.thermal_telemetry["hotspot_max_c"].mean()
    assert high_out.online_dga["h2_ppm"].mean() > low_out.online_dga["h2_ppm"].mean()


def test_generate_ot_data_factory(fleet: dict) -> None:
    out = generate_ot_data(fleet["transformers"], fleet["operational_history"], source=None)
    assert isinstance(out, OTData)


def test_real_source_stubs_raise_not_implemented() -> None:
    sources = [
        OSIsoftPIWebAPISource("https://pi.example.com/piwebapi"),
        OPCUASource("opc.tcp://historian.example:4840"),
        MQTTSparkplugSource("mqtts://broker.example:8883", "spBv1.0/utility"),
    ]
    for src in sources:
        with pytest.raises(NotImplementedError):
            src.fetch(pd.DataFrame(), pd.DataFrame())


def test_ot_features_present_in_training_frame() -> None:
    build_training_frame.cache_clear()
    df = build_training_frame()
    for col in OT_FEATURE_DEFAULTS:
        assert col in df.columns, f"{col} missing from training frame"
        assert df[col].notna().any(), f"{col} is entirely NaN"
    assert df[list(OT_FEATURE_DEFAULTS.keys())].notna().all().all()


def test_get_asset_features_returns_ot_columns(fleet: dict) -> None:
    aid = sorted(fleet["transformers"]["id"].tolist())[0]
    feats = get_asset_features(aid)
    assert all(col in feats.columns for col in FEATURES)


def test_hotspot_aging_multiplier_doubles_per_6c() -> None:
    base = _hotspot_aging_multiplier(110.0, 0.0)
    plus6 = _hotspot_aging_multiplier(116.0, 0.0)
    plus12 = _hotspot_aging_multiplier(122.0, 0.0)
    assert base == pytest.approx(1.0)
    assert plus6 == pytest.approx(2.0, rel=0.05)
    assert plus12 == pytest.approx(4.0, rel=0.05)


def test_hotspot_aging_multiplier_capped() -> None:
    assert _hotspot_aging_multiplier(200.0, 1000.0) <= 6.0


def test_hotspot_multiplier_extra_penalty_for_sustained_hours() -> None:
    short = _hotspot_aging_multiplier(115.0, 50.0)
    long = _hotspot_aging_multiplier(115.0, 500.0)
    assert long > short
