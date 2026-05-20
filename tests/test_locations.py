from __future__ import annotations

import pandas as pd

from src.data.loader import load_seed_data
from src.data.locations import (
    DEFAULT_REGION_COORDINATE,
    REGION_COORDINATES,
    coordinates_for_asset,
    coordinates_for_substation,
    fleet_with_coordinates,
)


def test_region_anchors_are_in_continental_us() -> None:
    for region, (lat, lon, city) in REGION_COORDINATES.items():
        assert 24 <= lat <= 50, f"{region} lat out of range: {lat}"
        assert -125 <= lon <= -65, f"{region} lon out of range: {lon}"
        assert city  # non-empty


def test_coordinates_for_asset_is_deterministic() -> None:
    a = coordinates_for_asset("T-0001")
    b = coordinates_for_asset("T-0001")
    assert a == b
    assert len(a) == 4  # (lat, lon, label, region)


def test_unknown_asset_falls_back_to_default() -> None:
    lat, lon, label, region = coordinates_for_asset("T-9999")
    assert (lat, lon, label) == DEFAULT_REGION_COORDINATE
    assert region == "Unknown"


def test_each_substation_has_distinct_anchor() -> None:
    tx = load_seed_data()["transformers"]
    pairs = tx[["region", "substation"]].drop_duplicates()
    seen: set[tuple[float, float]] = set()
    for _, row in pairs.iterrows():
        lat, lon, _ = coordinates_for_substation(row["substation"], row["region"])
        seen.add((round(lat, 4), round(lon, 4)))
    # 172 (region, sub) pairs → at least 170 unique coords (hash collisions allowed but rare)
    assert len(seen) >= len(pairs) - 2


def test_same_substation_assets_get_per_asset_jitter() -> None:
    tx = load_seed_data()["transformers"]
    # Find a (region, substation) with 2+ assets
    grp = tx.groupby(["region", "substation"]).size()
    multi = grp[grp >= 2]
    assert not multi.empty, "expected at least one multi-asset substation"
    region, substation = multi.index[0]
    ids = tx[(tx["region"] == region) & (tx["substation"] == substation)]["id"].tolist()
    coords = [(coordinates_for_asset(i)[0], coordinates_for_asset(i)[1]) for i in ids]
    assert len(set(coords)) == len(coords), "all same-substation assets should have distinct coords"


def test_fleet_with_coordinates_shape() -> None:
    df = fleet_with_coordinates()
    base = load_seed_data()["transformers"]
    assert len(df) == len(base) == 500
    for col in ("latitude", "longitude", "location"):
        assert col in df.columns
    assert df["latitude"].notna().all()
    assert df["longitude"].notna().all()
    # Continental US bounds (with jitter)
    assert df["latitude"].between(24, 50).all()
    assert df["longitude"].between(-125, -65).all()


def test_weather_module_still_exports_coordinates_for_asset() -> None:
    # Backward-compat: weather.py used to own this helper; it should still be importable from there.
    from src.data import weather

    assert hasattr(weather, "coordinates_for_asset")
    lat, lon, label, region = weather.coordinates_for_asset("T-0001")
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert region == "South"
