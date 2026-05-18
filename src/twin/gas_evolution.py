"""Simplified DGA gas-evolution kinetics for the digital twin.

Drives H₂, CH₄, and C₂H₄ generation rates as Arrhenius-style exponentials of
hot-spot temperature above each gas's onset temperature (IEC 60599 / IEEE
C57.104 thermal-fault regions). Real DGA kinetics are far richer (arcing,
partial discharge, secondary reactions), but for the demo the goal is a
defensible *trajectory* that responds to scenario stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GasKineticsParams:
    h2_onset_c: float = 130.0
    h2_rate_factor: float = 0.04
    """ppm per hour at the onset; doubles per +``half_doubling_c`` °C."""

    ch4_onset_c: float = 220.0
    ch4_rate_factor: float = 0.02

    c2h4_onset_c: float = 280.0
    c2h4_rate_factor: float = 0.01

    half_doubling_c: float = 10.0


def _rate(hotspot_c: float | np.ndarray, onset_c: float, factor: float, half_doubling_c: float) -> np.ndarray:
    h = np.asarray(hotspot_c, dtype=float)
    delta = h - onset_c
    rate = factor * np.power(2.0, delta / half_doubling_c)
    rate = np.where(h >= onset_c, rate, 0.0)
    return rate


@dataclass(frozen=True)
class GasGeneration:
    h2_ppm: float
    ch4_ppm: float
    c2h4_ppm: float


class GasEvolutionTwin:
    def __init__(self, params: GasKineticsParams | None = None) -> None:
        self.params = params or GasKineticsParams()

    def rates_ppm_per_hour(self, hotspot_c: float | np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        return {
            "h2": _rate(hotspot_c, p.h2_onset_c, p.h2_rate_factor, p.half_doubling_c),
            "ch4": _rate(hotspot_c, p.ch4_onset_c, p.ch4_rate_factor, p.half_doubling_c),
            "c2h4": _rate(hotspot_c, p.c2h4_onset_c, p.c2h4_rate_factor, p.half_doubling_c),
        }

    def accumulate(
        self,
        hotspot_trace: pd.Series | np.ndarray,
        t_hours: pd.Series | np.ndarray,
    ) -> GasGeneration:
        h = np.asarray(hotspot_trace, dtype=float)
        t = np.asarray(t_hours, dtype=float)
        if h.size != t.size or h.size < 2:
            raise ValueError("hotspot_trace and t_hours must be same length ≥ 2")
        rates = self.rates_ppm_per_hour(h)
        return GasGeneration(
            h2_ppm=float(np.trapezoid(rates["h2"], t)),
            ch4_ppm=float(np.trapezoid(rates["ch4"], t)),
            c2h4_ppm=float(np.trapezoid(rates["c2h4"], t)),
        )
