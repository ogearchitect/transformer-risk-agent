"""Preset scenarios for the digital twin.

A scenario is an hour-resolution tuple of ``(load_pu, ambient_c, cooling_pct)``
arrays, all the same length. ``build_scenario`` is the public factory; it
either looks up a named preset from :data:`SCENARIO_LIBRARY` or builds a custom
one from explicit lists / numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioInput:
    name: str
    load_pu: np.ndarray
    ambient_c: np.ndarray
    cooling_pct: np.ndarray
    description: str = ""

    @property
    def hours(self) -> int:
        return int(self.load_pu.size)


def _daily_load_shape(peak_pu: float, base_pu: float, peak_hour: int = 16) -> np.ndarray:
    hours = np.arange(24)
    diurnal = 0.5 * (1.0 + np.cos(2 * np.pi * (hours - peak_hour) / 24.0))
    return base_pu + (peak_pu - base_pu) * diurnal


def _diurnal_ambient(mean_c: float, swing_c: float, hot_hour: int = 15) -> np.ndarray:
    hours = np.arange(24)
    return mean_c + swing_c * np.cos(2 * np.pi * (hours - hot_hour) / 24.0)


SCENARIO_LIBRARY: dict[str, ScenarioInput] = {
    "normal_day": ScenarioInput(
        name="normal_day",
        load_pu=_daily_load_shape(peak_pu=0.95, base_pu=0.55),
        ambient_c=_diurnal_ambient(mean_c=22.0, swing_c=6.0),
        cooling_pct=np.full(24, 98.0),
        description="Typical loading on a temperate day. Baseline reference.",
    ),
    "summer_peak": ScenarioInput(
        name="summer_peak",
        load_pu=_daily_load_shape(peak_pu=1.15, base_pu=0.75),
        ambient_c=_diurnal_ambient(mean_c=32.0, swing_c=8.0),
        cooling_pct=np.full(24, 96.0),
        description="High summer peak load with 32 °C average ambient.",
    ),
    "heat_wave": ScenarioInput(
        name="heat_wave",
        load_pu=np.tile(_daily_load_shape(peak_pu=1.25, base_pu=0.85), 5),
        ambient_c=np.tile(_diurnal_ambient(mean_c=38.0, swing_c=6.0), 5),
        cooling_pct=np.tile(np.minimum(_diurnal_ambient(mean_c=92.0, swing_c=-6.0), 100.0), 5),
        description="5-day sustained heat wave: 1.25 p.u. peak, 38 °C ambient.",
    ),
    "contingency_transfer": ScenarioInput(
        name="contingency_transfer",
        load_pu=np.concatenate(
            [np.full(2, 0.85), np.full(8, 1.40), np.full(14, 0.95)]
        ),
        ambient_c=np.full(24, 28.0),
        cooling_pct=np.full(24, 100.0),
        description="Adjacent transformer trips; this unit picks up 1.4 p.u. for 8 hours.",
    ),
    "cooling_loss": ScenarioInput(
        name="cooling_loss",
        load_pu=_daily_load_shape(peak_pu=1.0, base_pu=0.7),
        ambient_c=_diurnal_ambient(mean_c=30.0, swing_c=6.0),
        cooling_pct=np.concatenate(
            [np.full(6, 98.0), np.full(10, 60.0), np.full(8, 98.0)]
        ),
        description="One cooling-fan bank fails for 10 hours at midday.",
    ),
}


def build_scenario(
    name: str,
    *,
    load_pu: list[float] | np.ndarray | None = None,
    ambient_c: list[float] | np.ndarray | None = None,
    cooling_pct: list[float] | np.ndarray | None = None,
    description: str = "",
    repeat_days: int = 1,
) -> ScenarioInput:
    """Either look up a preset (case-insensitive) or build a custom scenario.

    Custom scenarios must provide ``load_pu`` and ``ambient_c``; ``cooling_pct``
    defaults to 100. ``repeat_days`` tiles the profile so a 24h shape becomes
    a multi-day scenario.
    """
    key = name.lower().strip().replace(" ", "_").replace("-", "_")
    if load_pu is None and ambient_c is None and cooling_pct is None:
        preset = SCENARIO_LIBRARY.get(key)
        if preset is None:
            raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIO_LIBRARY)}")
        if repeat_days > 1:
            return ScenarioInput(
                name=preset.name,
                load_pu=np.tile(preset.load_pu, repeat_days),
                ambient_c=np.tile(preset.ambient_c, repeat_days),
                cooling_pct=np.tile(preset.cooling_pct, repeat_days),
                description=f"{preset.description} (×{repeat_days} days)",
            )
        return preset

    if load_pu is None or ambient_c is None:
        raise ValueError("Custom scenarios require both load_pu and ambient_c")
    load = np.asarray(list(load_pu), dtype=float)
    amb = np.asarray(list(ambient_c), dtype=float)
    cool = np.asarray(list(cooling_pct), dtype=float) if cooling_pct is not None else np.full_like(load, 100.0)
    if not (load.size == amb.size == cool.size):
        raise ValueError("Custom profiles must have matching lengths")
    if repeat_days > 1:
        load = np.tile(load, repeat_days)
        amb = np.tile(amb, repeat_days)
        cool = np.tile(cool, repeat_days)
    return ScenarioInput(name=name, load_pu=load, ambient_c=amb, cooling_pct=cool, description=description)
