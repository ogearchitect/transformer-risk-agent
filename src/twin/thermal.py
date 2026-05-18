"""IEEE C57.91-2011 thermal model for oil-filled power transformers.

Encodes the standard top-oil and hot-spot rise equations as a 2-state ODE that
:func:`ThermalTwin.simulate` integrates with ``scipy.integrate.solve_ivp``.

Variables
---------
- ``θ_to(t)``  – top-oil temperature, °C
- ``Δθ_hs(t)`` – winding-to-top-oil hot-spot rise, °C
- ``θ_hs(t) = θ_to(t) + Δθ_hs(t)`` – winding hot-spot temperature

Steady-state forcing functions (IEEE C57.91 §7, ``ambient`` excluded so the
state is rise-over-ambient — added back at the very end):

.. math::

    \\theta_{to,ss}(K) &= \\theta_{to,rated} \\cdot \\left(\\frac{K^2 R + 1}{R + 1}\\right)^n \\\\
    \\Delta\\theta_{hs,ss}(K) &= \\Delta\\theta_{hs,rated} \\cdot K^{2m}

where ``K`` is the per-unit load, ``R`` is the ratio of load-loss to no-load
loss, and ``n``/``m`` are cooling-mode exponents. Transients relax toward the
steady state with time constants ``τ_to`` (oil, minutes) and ``τ_w`` (winding,
minutes).

Default coefficients are IEEE Table B.3 nominal values for ONAF cooling — the
most common class in our synthetic fleet. ``ThermalCoefficients`` is a
dataclass so per-asset calibration can override any field
(see :mod:`src.twin.calibration`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ThermalCoefficients:
    """IEEE C57.91 thermal coefficients (ONAF defaults)."""

    theta_to_rated_c: float = 45.0
    delta_theta_hs_rated_c: float = 28.0
    R: float = 6.0
    n: float = 0.9
    m: float = 0.9
    tau_to_min: float = 150.0
    tau_w_min: float = 6.0
    cooling_loss_factor: float = 0.85
    """When cooling is partially unavailable, ``τ_to`` increases and steady-state
    top-oil rises. ``cooling_loss_factor`` is the multiplier applied to
    ``θ_to,ss`` when ``cooling_available_pct`` drops to 0 — interpolated
    linearly toward 1.0 at full availability."""


COOLING_TYPE_DEFAULTS: dict[str, ThermalCoefficients] = {
    "ONAN": ThermalCoefficients(
        theta_to_rated_c=55.0, delta_theta_hs_rated_c=23.0,
        R=5.0, n=0.8, m=0.8, tau_to_min=180.0, tau_w_min=7.0,
        cooling_loss_factor=0.85,
    ),
    "ONAF": ThermalCoefficients(
        theta_to_rated_c=45.0, delta_theta_hs_rated_c=28.0,
        R=6.0, n=0.9, m=0.9, tau_to_min=150.0, tau_w_min=6.0,
        cooling_loss_factor=0.80,
    ),
    "OFAF": ThermalCoefficients(
        theta_to_rated_c=40.0, delta_theta_hs_rated_c=28.0,
        R=6.0, n=1.0, m=1.0, tau_to_min=90.0, tau_w_min=5.0,
        cooling_loss_factor=0.70,
    ),
    "ODAF": ThermalCoefficients(
        theta_to_rated_c=40.0, delta_theta_hs_rated_c=28.0,
        R=6.0, n=1.0, m=1.0, tau_to_min=80.0, tau_w_min=5.0,
        cooling_loss_factor=0.65,
    ),
}


def default_coefficients_for(cooling_type: str | None) -> ThermalCoefficients:
    if cooling_type is None:
        return ThermalCoefficients()
    return COOLING_TYPE_DEFAULTS.get(str(cooling_type).upper(), ThermalCoefficients())


def _top_oil_steady_state(K: float, coef: ThermalCoefficients) -> float:
    return coef.theta_to_rated_c * ((K**2 * coef.R + 1.0) / (coef.R + 1.0)) ** coef.n


def _hotspot_rise_steady_state(K: float, coef: ThermalCoefficients) -> float:
    return coef.delta_theta_hs_rated_c * (max(K, 0.0) ** (2.0 * coef.m))


def _cooling_derate(cooling_pct: float, coef: ThermalCoefficients) -> float:
    """Cooling availability ∈ [0, 100] → multiplier on θ_to,ss."""
    pct = float(np.clip(cooling_pct, 0.0, 100.0)) / 100.0
    return float(1.0 + (1.0 / max(coef.cooling_loss_factor, 1e-3) - 1.0) * (1.0 - pct))


@dataclass(frozen=True)
class ThermalTwin:
    """IEEE C57.91 thermal twin for a single transformer."""

    coefficients: ThermalCoefficients = field(default_factory=ThermalCoefficients)

    def with_coefficients(self, **overrides: float) -> "ThermalTwin":
        return ThermalTwin(replace(self.coefficients, **overrides))

    def steady_state(self, load_pu: float, ambient_c: float, cooling_pct: float = 100.0) -> tuple[float, float]:
        coef = self.coefficients
        theta_to = _top_oil_steady_state(load_pu, coef) * _cooling_derate(cooling_pct, coef)
        delta_hs = _hotspot_rise_steady_state(load_pu, coef)
        top_oil = ambient_c + theta_to
        hotspot = top_oil + delta_hs
        return float(top_oil), float(hotspot)

    def simulate(
        self,
        load_profile: Iterable[float],
        ambient_profile: Iterable[float],
        cooling_profile: Iterable[float] | None = None,
        *,
        dt_seconds: float = 300.0,
        initial_top_oil_rise_c: float | None = None,
        initial_hotspot_rise_c: float | None = None,
    ) -> pd.DataFrame:
        """Integrate the thermal ODE over a discrete-time scenario.

        Parameters
        ----------
        load_profile, ambient_profile
            Hour-resolution samples (one value per simulation hour). Both must
            have the same length; the simulator linearly interpolates between
            samples while integrating at ``dt_seconds`` resolution.
        cooling_profile
            Optional cooling-availability percentage per hour (0–100). Defaults
            to 100 (fully available) when omitted.
        dt_seconds
            Integration step (default 5 min = 300 s — well below either time
            constant).
        initial_top_oil_rise_c, initial_hotspot_rise_c
            Optional initial rises over ambient. Defaults to the steady-state
            values for the first ``(load, ambient, cooling)`` sample.

        Returns
        -------
        pandas.DataFrame
            Columns: ``t_hours, load_pu, ambient_c, cooling_pct, top_oil_c,
            hotspot_c, delta_hs_c``.
        """
        load = np.asarray(list(load_profile), dtype=float)
        amb = np.asarray(list(ambient_profile), dtype=float)
        if load.size != amb.size or load.size == 0:
            raise ValueError("load_profile and ambient_profile must be same non-zero length")
        cool = (
            np.full_like(load, 100.0)
            if cooling_profile is None
            else np.asarray(list(cooling_profile), dtype=float)
        )
        if cool.size != load.size:
            raise ValueError("cooling_profile must match load_profile length")

        hours = float(load.size - 1) if load.size > 1 else 1.0
        t_hours = np.arange(load.size, dtype=float)

        def sample(t: float) -> tuple[float, float, float]:
            t = float(np.clip(t, 0.0, hours))
            return (
                float(np.interp(t, t_hours, load)),
                float(np.interp(t, t_hours, amb)),
                float(np.interp(t, t_hours, cool)),
            )

        coef = self.coefficients
        tau_to_h = coef.tau_to_min / 60.0
        tau_w_h = coef.tau_w_min / 60.0

        def deriv(t: float, state: np.ndarray) -> list[float]:
            theta_to_rise, delta_hs = state
            K, _, c_pct = sample(t)
            theta_to_ss = _top_oil_steady_state(K, coef) * _cooling_derate(c_pct, coef)
            delta_hs_ss = _hotspot_rise_steady_state(K, coef)
            return [
                (theta_to_ss - theta_to_rise) / tau_to_h,
                (delta_hs_ss - delta_hs) / tau_w_h,
            ]

        K0, amb0, c0 = sample(0.0)
        if initial_top_oil_rise_c is None:
            initial_top_oil_rise_c = _top_oil_steady_state(K0, coef) * _cooling_derate(c0, coef)
        if initial_hotspot_rise_c is None:
            initial_hotspot_rise_c = _hotspot_rise_steady_state(K0, coef)

        n_steps = max(int(round(hours * 3600.0 / dt_seconds)), 1)
        t_eval = np.linspace(0.0, hours, n_steps + 1)
        sol = solve_ivp(
            deriv,
            (0.0, hours),
            [float(initial_top_oil_rise_c), float(initial_hotspot_rise_c)],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-4,
            max_step=dt_seconds / 3600.0 * 2.0,
        )
        if not sol.success:  # pragma: no cover
            raise RuntimeError(f"Thermal ODE solver failed: {sol.message}")

        amb_interp = np.interp(sol.t, t_hours, amb)
        cool_interp = np.interp(sol.t, t_hours, cool)
        load_interp = np.interp(sol.t, t_hours, load)
        top_oil_rise = sol.y[0]
        delta_hs = sol.y[1]
        top_oil = amb_interp + top_oil_rise
        hotspot = top_oil + delta_hs
        return pd.DataFrame(
            {
                "t_hours": sol.t,
                "load_pu": load_interp,
                "ambient_c": amb_interp,
                "cooling_pct": cool_interp,
                "top_oil_c": top_oil,
                "delta_hs_c": delta_hs,
                "hotspot_c": hotspot,
            }
        )
