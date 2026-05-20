from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.data.locations import (
    DEFAULT_REGION_COORDINATE,
    REGION_COORDINATES,
    coordinates_for_asset,
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_S = int(os.getenv("WEATHER_CACHE_TTL_S", "3600"))
WEATHER_TIMEOUT_S = float(os.getenv("WEATHER_TIMEOUT_S", "4.0"))

WMO_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Severe thunderstorm",
}


@dataclass
class DailyForecast:
    date: str
    temperature_max_c: float
    temperature_min_c: float
    humidity_max_pct: float
    wind_max_kmh: float
    weather_code: int
    description: str


@dataclass
class WeatherForecast:
    region: str
    location_name: str
    latitude: float
    longitude: float
    source: str
    days: list[DailyForecast]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self.days])


_CACHE: dict[tuple[float, float], tuple[float, WeatherForecast]] = {}


def reset_weather_cache_for_tests() -> None:
    _CACHE.clear()


def _synthetic_forecast(latitude: float, longitude: float) -> list[DailyForecast]:
    """Deterministic offline fallback when Open-Meteo is unreachable."""
    today = date.today()
    base = 22.0 - 0.4 * (abs(latitude) - 30.0)
    out: list[DailyForecast] = []
    for i in range(7):
        d = today + timedelta(days=i)
        wave = 6.0 * math.sin((i + abs(int(longitude)) % 7) / 1.3)
        tmax = base + wave + 5.0
        tmin = tmax - 9.0
        out.append(
            DailyForecast(
                date=d.isoformat(),
                temperature_max_c=round(tmax, 1),
                temperature_min_c=round(tmin, 1),
                humidity_max_pct=70.0 + 8.0 * math.cos(i / 1.7),
                wind_max_kmh=15.0 + 3.0 * (i % 3),
                weather_code=1 if tmax > 25 else 3,
                description=WMO_CODE_DESCRIPTIONS.get(1, "Mainly clear"),
            )
        )
    return out


def _fetch_open_meteo(latitude: float, longitude: float) -> tuple[list[DailyForecast], str]:
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,wind_speed_10m_max,weather_code",
        "timezone": "auto",
        "forecast_days": "7",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "transformer-risk-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=WEATHER_TIMEOUT_S) as r:
            payload: dict[str, Any] = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return _synthetic_forecast(latitude, longitude), "offline-fallback"

    daily = payload.get("daily") or {}
    days_iso = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    hum = daily.get("relative_humidity_2m_max", [])
    wind = daily.get("wind_speed_10m_max", [])
    codes = daily.get("weather_code", [])
    if not days_iso:
        return _synthetic_forecast(latitude, longitude), "offline-fallback"

    out: list[DailyForecast] = []
    for i, d in enumerate(days_iso):
        code = int(codes[i]) if i < len(codes) else 0
        out.append(
            DailyForecast(
                date=str(d),
                temperature_max_c=float(tmax[i]) if i < len(tmax) and tmax[i] is not None else float("nan"),
                temperature_min_c=float(tmin[i]) if i < len(tmin) and tmin[i] is not None else float("nan"),
                humidity_max_pct=float(hum[i]) if i < len(hum) and hum[i] is not None else float("nan"),
                wind_max_kmh=float(wind[i]) if i < len(wind) and wind[i] is not None else float("nan"),
                weather_code=code,
                description=WMO_CODE_DESCRIPTIONS.get(code, f"Code {code}"),
            )
        )
    return out, "open-meteo"


def fetch_forecast(latitude: float, longitude: float, location_name: str = "", region: str = "") -> WeatherForecast:
    key = (round(latitude, 3), round(longitude, 3))
    now = time.time()
    cached = _CACHE.get(key)
    if cached and (now - cached[0]) < WEATHER_CACHE_TTL_S:
        return cached[1]
    days, source = _fetch_open_meteo(latitude, longitude)
    fc = WeatherForecast(
        region=region or "",
        location_name=location_name or f"{latitude:.2f},{longitude:.2f}",
        latitude=latitude,
        longitude=longitude,
        source=source,
        days=days,
    )
    _CACHE[key] = (now, fc)
    return fc


def forecast_for_asset(asset_id: str) -> WeatherForecast:
    lat, lon, name, region = coordinates_for_asset(asset_id)
    return fetch_forecast(lat, lon, location_name=name, region=region)


def heat_risk_summary(fc: WeatherForecast, heat_threshold_c: float = 32.0) -> dict[str, Any]:
    if not fc.days:
        return {"peak_temp_c": float("nan"), "hot_days": 0, "alert": "No data"}
    tmax_vals = [d.temperature_max_c for d in fc.days if not math.isnan(d.temperature_max_c)]
    if not tmax_vals:
        return {"peak_temp_c": float("nan"), "hot_days": 0, "alert": "No data"}
    peak = max(tmax_vals)
    hot_days = sum(1 for t in tmax_vals if t >= heat_threshold_c)
    if peak >= 38:
        alert = f"Extreme heat: peak {peak:.0f} °C — derate or shed load preemptively"
    elif peak >= heat_threshold_c:
        alert = f"Heat advisory: {hot_days} day(s) ≥ {heat_threshold_c:.0f} °C — expect ~{(peak - 30):.0f}× aging multiplier on stressed assets"
    elif peak >= 25:
        alert = "Warm: monitor cooling availability"
    else:
        alert = "Mild ambient — nominal aging expected"
    return {"peak_temp_c": peak, "hot_days": hot_days, "alert": alert, "heat_threshold_c": heat_threshold_c}
