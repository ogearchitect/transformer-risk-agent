"""Arrhenius insulation-aging integrator (IEEE C57.91 §7.2).

Aging acceleration factor:

.. math::

    F_{AA}(\\theta_{hs}) = 2^{(\\theta_{hs} - 110)/6}

Per-unit life consumed over a trajectory is the time-integral of ``F_AA``
divided by the normal insulation life (180 000 hours ≈ 20.5 years, the IEEE
reference for thermally-upgraded paper at 110 °C continuous).

Functions in this module are vectorised — they accept a hotspot trace
(``numpy.ndarray`` or ``pandas.Series``) plus a matching ``t_hours`` index
and return scalar life-consumed in per-unit, or hours of equivalent normal
aging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

normal_insulation_life_hours = 180_000.0


def aging_acceleration_factor(hotspot_c: float | np.ndarray) -> np.ndarray:
    return np.power(2.0, (np.asarray(hotspot_c, dtype=float) - 110.0) / 6.0)


def hours_to_years(hours: float) -> float:
    return float(hours / (365.25 * 24.0))


@dataclass(frozen=True)
class AgingResult:
    aging_hours_equivalent: float
    per_unit_life_consumed: float
    peak_faa: float
    average_faa: float


class AgingTwin:
    """Integrate insulation aging over a thermal trajectory."""

    def __init__(self, normal_life_hours: float = normal_insulation_life_hours) -> None:
        self.normal_life_hours = float(normal_life_hours)

    def integrate(self, hotspot_trace: pd.Series | np.ndarray, t_hours: np.ndarray | pd.Series | None = None) -> AgingResult:
        h = np.asarray(hotspot_trace, dtype=float)
        if t_hours is None:
            if isinstance(hotspot_trace, pd.Series):
                t = np.arange(len(h), dtype=float)
            else:
                t = np.arange(len(h), dtype=float)
        else:
            t = np.asarray(t_hours, dtype=float)
        if h.size != t.size or h.size < 2:
            raise ValueError("hotspot_trace and t_hours must be same length ≥ 2")
        faa = aging_acceleration_factor(h)
        aging_hours = float(np.trapezoid(faa, t))
        return AgingResult(
            aging_hours_equivalent=aging_hours,
            per_unit_life_consumed=float(aging_hours / self.normal_life_hours),
            peak_faa=float(faa.max()),
            average_faa=float(faa.mean()),
        )

    def remaining_life_years(
        self,
        already_consumed_pu_life: float,
        steady_hotspot_c: float,
    ) -> float:
        """Project years remaining if the asset runs continuously at
        ``steady_hotspot_c`` from now."""
        already = float(np.clip(already_consumed_pu_life, 0.0, 0.999))
        faa = float(aging_acceleration_factor(steady_hotspot_c))
        hours_remaining = (1.0 - already) * self.normal_life_hours / max(faa, 1e-6)
        return hours_to_years(hours_remaining)

    def rul_with_scenario(
        self,
        already_consumed_pu_life: float,
        scenario_hotspot_trace: pd.Series | np.ndarray,
        t_hours: np.ndarray | pd.Series | None = None,
        recurrence_per_year: float = 1.0,
        baseline_hotspot_c: float = 95.0,
    ) -> float:
        """Apply ``recurrence_per_year`` occurrences of the scenario plus a
        baseline ``baseline_hotspot_c`` regime for the rest of the year, then
        project remaining years until 100 % per-unit life consumed.
        """
        scenario_aging = self.integrate(scenario_hotspot_trace, t_hours)
        per_event_pu = scenario_aging.per_unit_life_consumed
        event_hours_per_year = (
            float(np.asarray(t_hours)[-1] - np.asarray(t_hours)[0]) if t_hours is not None
            else float(len(np.asarray(scenario_hotspot_trace)) - 1)
        ) * recurrence_per_year
        event_hours_per_year = min(event_hours_per_year, 8766.0)
        baseline_faa = float(aging_acceleration_factor(baseline_hotspot_c))
        baseline_hours = max(8766.0 - event_hours_per_year, 0.0)
        baseline_pu_per_year = baseline_hours * baseline_faa / self.normal_life_hours
        annual_pu = recurrence_per_year * per_event_pu + baseline_pu_per_year
        already = float(np.clip(already_consumed_pu_life, 0.0, 0.999))
        if annual_pu <= 0.0:
            return 100.0
        years = (1.0 - already) / annual_pu
        return float(np.clip(years, 0.0, 100.0))
