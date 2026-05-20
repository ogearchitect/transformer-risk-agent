from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.data.loader import load_seed_data

REGION_COORDINATES: dict[str, tuple[float, float, str]] = {
    "North":   (44.9778, -93.2650, "Minneapolis, MN"),
    "South":   (29.7604, -95.3698, "Houston, TX"),
    "East":    (42.3601, -71.0589, "Boston, MA"),
    "West":    (38.5816, -121.4944, "Sacramento, CA"),
    "Central": (39.0997, -94.5786, "Kansas City, MO"),
}

DEFAULT_REGION_COORDINATE = (39.8283, -98.5795, "Lebanon, KS")

REGION_SPREAD_DEG = 0.9
SUBSTATION_SPREAD_DEG = 0.06


def _substation_offset(substation: str) -> tuple[float, float]:
    """Deterministic per-substation jitter so substations don't overlap."""
    h = abs(hash(substation))
    lat = (((h % 401) - 200) / 200.0) * REGION_SPREAD_DEG
    lon = (((h // 401) % 401 - 200) / 200.0) * REGION_SPREAD_DEG
    return lat, lon


def _asset_offset(asset_id: str) -> tuple[float, float]:
    """Small jitter so two transformers at the same substation are distinguishable."""
    h = abs(hash(asset_id + "::asset"))
    lat = (((h % 97) - 48) / 48.0) * SUBSTATION_SPREAD_DEG
    lon = (((h // 97) % 97 - 48) / 48.0) * SUBSTATION_SPREAD_DEG
    return lat, lon


def coordinates_for_substation(substation: str, region: str) -> tuple[float, float, str]:
    """Return (lat, lon, label) for a substation in a region."""
    lat0, lon0, city = REGION_COORDINATES.get(region, DEFAULT_REGION_COORDINATE)
    dlat, dlon = _substation_offset(substation)
    return lat0 + dlat, lon0 + dlon, f"{city} · {substation}"


def coordinates_for_asset(asset_id: str) -> tuple[float, float, str, str]:
    """Return (lat, lon, label, region) for a transformer asset id."""
    df = load_seed_data()["transformers"]
    row = df[df["id"] == asset_id]
    if row.empty:
        lat, lon, name = DEFAULT_REGION_COORDINATE
        return lat, lon, name, "Unknown"
    region = str(row.iloc[0]["region"])
    substation = str(row.iloc[0]["substation"])
    lat, lon, label = coordinates_for_substation(substation, region)
    dlat, dlon = _asset_offset(asset_id)
    return lat + dlat, lon + dlon, label, region


@lru_cache(maxsize=1)
def fleet_with_coordinates() -> pd.DataFrame:
    """Return the transformer fleet enriched with per-asset (lat, lon, location)."""
    df = load_seed_data()["transformers"].copy()
    coords = df.apply(lambda r: pd.Series(coordinates_for_asset(r["id"]),
                                          index=["latitude", "longitude", "location", "region_resolved"]), axis=1)
    return pd.concat([df, coords[["latitude", "longitude", "location"]]], axis=1)
