"""OT (Operational Technology) telemetry adapters and synthetic generators.

This module separates the *source* of OT telemetry from the *shape* the rest of
the pipeline consumes. Real installations plug in one of the stub adapters
(OSIsoft PI Web API, OPC UA, MQTT/Sparkplug) which must return the same Pandas
DataFrames the synthetic generator produces.

Tables produced (monthly cadence, latest-row features picked up by
``build_training_frame``):

- ``thermal_telemetry``  — top-oil / winding hot-spot temperatures and load-over-
  assigned hours (drives IEEE C57.91 Arrhenius aging).
- ``online_dga``         — continuous H2 / H2O monitor (rate-of-change beats
  the quarterly snapshot DGA in ``dga_history``).
- ``oltc_telemetry``     — on-load tap-changer ops, motor torque, contact wear.
- ``cooling_status``     — cooling-system availability (fan/pump uptime).
- ``fault_events``       — through-fault events with I²t energy (event-based,
  variable cadence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OTData:
    thermal_telemetry: pd.DataFrame
    online_dga: pd.DataFrame
    oltc_telemetry: pd.DataFrame
    cooling_status: pd.DataFrame
    fault_events: pd.DataFrame


class OTSource(Protocol):
    """Contract every OT adapter (synthetic or real) must satisfy."""

    def fetch(
        self,
        transformers: pd.DataFrame,
        operational_history: pd.DataFrame,
    ) -> OTData: ...


class SyntheticOTSource:
    """Generates physically-plausible OT telemetry correlated with the same
    drivers the base fleet generator uses (age, loading, fault pressure).

    Uses ``np.random.default_rng(seed=seed)`` for determinism — mirrors
    ``src/data/generator.py``.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def fetch(
        self,
        transformers: pd.DataFrame,
        operational_history: pd.DataFrame,
        stress_factor: pd.Series | None = None,
    ) -> OTData:
        rng = np.random.default_rng(seed=self._seed)
        op_latest = (
            operational_history.sort_values("date")
            .groupby("id")
            .tail(1)
            .set_index("id")
        )
        op_yearly = (
            operational_history.sort_values("date")
            .groupby("id")
            .tail(12)
        )
        avg_load_12mo = op_yearly.groupby("id")["avg_loading_pct"].mean()
        peak_load_12mo = op_yearly.groupby("id")["peak_loading_pct"].max()
        avg_ambient_12mo = op_yearly.groupby("id")["ambient_temp_c"].mean()
        through_faults_12mo = op_yearly.groupby("id")["through_faults"].sum()
        if stress_factor is None:
            stress_factor = pd.Series(0.0, index=transformers["id"])

        thermal_rows: list[dict] = []
        dga_rows: list[dict] = []
        oltc_rows: list[dict] = []
        cooling_rows: list[dict] = []
        fault_rows: list[dict] = []

        latest_date = pd.to_datetime(operational_history["date"].max())
        monthly_dates = pd.date_range(end=latest_date, periods=12, freq="MS")

        for _, tx in transformers.iterrows():
            aid = tx["id"]
            age = float(2026 - tx["vintage_year"])
            mva = float(tx["mva_rating"])
            cooling_type = str(tx["cooling_type"])
            avg_load = float(avg_load_12mo.get(aid, 70.0))
            peak_load = float(peak_load_12mo.get(aid, 95.0))
            amb = float(avg_ambient_12mo.get(aid, 22.0))
            tfs = int(through_faults_12mo.get(aid, 0))
            crit = float(tx["substation_importance"]) / 100.0
            # Stress factor in [0,1]: high values reflect the *latent* fault
            # pressure that drives the failure label. OT signals encode this
            # otherwise-hidden state, which is exactly why OT data lifts model
            # accuracy in real fleets.
            sf = float(np.clip(stress_factor.get(aid, 0.0), 0.0, 1.0))

            cooling_baseline = {"ONAN": 0.985, "ONAF": 0.975, "OFAF": 0.965}[cooling_type]
            cooling_drift = 0.12 * (age / 60.0) + 0.10 * sf
            cooling_jitter = rng.normal(0, 0.015)
            cooling_avail = float(np.clip(cooling_baseline - cooling_drift + cooling_jitter, 0.55, 1.0))

            top_oil_rise = 32.0 + 0.32 * max(0.0, peak_load - 70.0) + 8.0 * sf
            hotspot_rise_over_top_oil = 28.0 + 0.35 * max(0.0, peak_load - 100.0) + 6.0 * sf
            cooling_penalty = (1.0 - cooling_avail) * 45.0

            for date in monthly_dates:
                month_load_jitter = rng.normal(0, 4.0)
                month_amb_jitter = rng.normal(0, 3.0)
                m_amb = amb + month_amb_jitter
                m_peak_load = peak_load + month_load_jitter
                top_oil_max = m_amb + top_oil_rise * (m_peak_load / 100.0) ** 0.8 + cooling_penalty
                top_oil_avg = top_oil_max - rng.uniform(8, 14)
                hotspot_max = top_oil_max + hotspot_rise_over_top_oil
                hotspot_avg = hotspot_max - rng.uniform(10, 16)
                hrs_over_110 = float(np.clip(
                    (hotspot_max - 110.0) * 6.0 + rng.normal(0, 4), 0.0, 720.0,
                ))
                thermal_rows.append({
                    "id": aid,
                    "date": date,
                    "top_oil_max_c": round(float(top_oil_max), 2),
                    "top_oil_avg_c": round(float(top_oil_avg), 2),
                    "hotspot_max_c": round(float(hotspot_max), 2),
                    "hotspot_avg_c": round(float(hotspot_avg), 2),
                    "hours_above_110c": round(hrs_over_110, 1),
                })

            base_h2 = 35.0 + 0.6 * age + 30.0 * sf + rng.normal(0, 6)
            h2_rate_base = (
                0.10
                + 0.040 * (avg_load - 70.0) / 30.0
                + 0.06 * (1.0 - cooling_avail) * 10.0
                + 0.45 * sf
            )
            for idx, date in enumerate(monthly_dates):
                age_factor = (idx / len(monthly_dates))
                h2 = max(5.0, base_h2 + age_factor * (5 + rng.normal(0, 2)))
                h2_rate = max(0.0, h2_rate_base + age_factor * 0.05 + rng.normal(0, 0.04))
                h2o = float(np.clip(8.0 + (age / 5.0) + rng.normal(0, 2.5), 1.0, 40.0))
                if h2_rate > 0.45 or h2o > 28:
                    status = "warning"
                elif h2_rate > 0.22 or h2o > 20:
                    status = "caution"
                else:
                    status = "normal"
                dga_rows.append({
                    "id": aid,
                    "date": date,
                    "h2_ppm": round(float(h2), 2),
                    "h2o_ppm": round(float(h2o), 2),
                    "h2_rate_ppm_per_day": round(float(h2_rate), 4),
                    "status": status,
                })

            ops_baseline = 18.0 + 6.0 * (peak_load / 100.0) + rng.normal(0, 3)
            torque_baseline = 100.0 + 0.6 * age + rng.normal(0, 5)
            for date in monthly_dates:
                ops = int(max(0, rng.poisson(ops_baseline)))
                torque = float(np.clip(torque_baseline + rng.normal(0, 4), 70.0, 160.0))
                wear = float(np.clip(0.20 + (age / 90.0) + (ops / 600.0), 0.0, 1.0))
                oltc_rows.append({
                    "id": aid,
                    "date": date,
                    "ops_count": ops,
                    "motor_torque_pct_nominal": round(torque, 2),
                    "contact_wear_score": round(wear, 3),
                })

            for date in monthly_dates:
                monthly_jitter = rng.normal(0, 0.008)
                m_avail = float(np.clip(cooling_avail + monthly_jitter, 0.5, 1.0))
                fan_fail = int(max(0, rng.poisson(max(0.05, (1.0 - m_avail) * 6.0))))
                pump_fail = int(max(0, rng.poisson(max(0.0, (1.0 - m_avail) * 1.5))))
                cooling_rows.append({
                    "id": aid,
                    "date": date,
                    "availability_pct": round(m_avail * 100.0, 2),
                    "fan_failures": fan_fail,
                    "pump_failures": pump_fail,
                })

            expected_events = max(0.5, 0.6 * tfs + 0.3 * crit * 4.0 + 3.0 * sf)
            n_events = int(rng.poisson(expected_events))
            for _ in range(n_events):
                evt_date = monthly_dates[int(rng.integers(0, len(monthly_dates)))]
                mag_ka = float(np.clip(rng.lognormal(mean=1.4, sigma=0.55), 0.5, 35.0))
                duration_ms = float(np.clip(rng.lognormal(mean=4.5, sigma=0.4), 30.0, 600.0))
                i2t = (mag_ka * 1000.0) ** 2 * (duration_ms / 1000.0)
                fault_rows.append({
                    "id": aid,
                    "date": evt_date,
                    "magnitude_ka": round(mag_ka, 3),
                    "duration_ms": round(duration_ms, 1),
                    "i2t_a2s": round(float(i2t), 2),
                })

        thermal_df = pd.DataFrame(thermal_rows)
        dga_df = pd.DataFrame(dga_rows)
        oltc_df = pd.DataFrame(oltc_rows)
        cooling_df = pd.DataFrame(cooling_rows)
        fault_df = (
            pd.DataFrame(fault_rows)
            if fault_rows
            else pd.DataFrame(columns=["id", "date", "magnitude_ka", "duration_ms", "i2t_a2s"])
        )
        return OTData(
            thermal_telemetry=thermal_df,
            online_dga=dga_df,
            oltc_telemetry=oltc_df,
            cooling_status=cooling_df,
            fault_events=fault_df,
        )


class OSIsoftPIWebAPISource:
    """Stub adapter for OSIsoft / AVEVA PI Web API.

    Production wiring (left as a TODO so the demo can ship without the
    enterprise dependency):

    1. Auth: Kerberos or OAuth2 against the PI Web API endpoint.
    2. Map each transformer `id` to its PI AF element path via a config table.
    3. For each attribute (top-oil temp, hot-spot temp, online H2, etc.), call
       ``/streams/{webId}/recorded`` or ``/interpolated`` over the lookback
       window and resample to monthly summaries.
    4. Return the same ``OTData`` shape as ``SyntheticOTSource``.
    """

    def __init__(self, base_url: str, lookback_months: int = 12) -> None:
        self.base_url = base_url
        self.lookback_months = lookback_months

    def fetch(self, transformers: pd.DataFrame, operational_history: pd.DataFrame) -> OTData:  # pragma: no cover
        raise NotImplementedError(
            "OSIsoftPIWebAPISource is a documentation stub. Implement the four "
            "steps in the class docstring to wire your PI deployment."
        )


class OPCUASource:
    """Stub adapter for OPC UA historians (asyncua/python-opcua).

    Production wiring:

    1. Connect to the OPC UA server (``asyncua.Client``).
    2. Browse the address space for the asset's node IDs (mapped via config).
    3. Use ``HistoryRead`` to pull raw or aggregated samples for the lookback.
    4. Resample to monthly summaries and return ``OTData``.
    """

    def __init__(self, endpoint: str, lookback_months: int = 12) -> None:
        self.endpoint = endpoint
        self.lookback_months = lookback_months

    def fetch(self, transformers: pd.DataFrame, operational_history: pd.DataFrame) -> OTData:  # pragma: no cover
        raise NotImplementedError(
            "OPCUASource is a documentation stub. Implement HistoryRead-based "
            "fetch using asyncua to wire your OPC UA historian."
        )


class MQTTSparkplugSource:
    """Stub adapter for MQTT / Sparkplug B brokers (Ignition / HiveMQ / EMQX).

    Production wiring:

    1. Subscribe to the asset namespace (``spBv1.0/<group>/DDATA/<edge>/<dev>``)
       and accumulate into a rolling time-series store (Timescale / InfluxDB).
    2. On ``fetch``, query the store for the lookback window per asset.
    3. Resample to monthly summaries and return ``OTData``.
    """

    def __init__(self, broker_url: str, topic_root: str, lookback_months: int = 12) -> None:
        self.broker_url = broker_url
        self.topic_root = topic_root
        self.lookback_months = lookback_months

    def fetch(self, transformers: pd.DataFrame, operational_history: pd.DataFrame) -> OTData:  # pragma: no cover
        raise NotImplementedError(
            "MQTTSparkplugSource is a documentation stub. Implement broker "
            "subscription + time-series persistence to wire your IIoT stack."
        )


def generate_ot_data(
    transformers: pd.DataFrame,
    operational_history: pd.DataFrame,
    source: OTSource | None = None,
    stress_factor: pd.Series | None = None,
) -> OTData:
    """Convenience factory — uses ``SyntheticOTSource`` by default so the demo
    runs offline. ``stress_factor`` (id-indexed Series in [0,1]) lets the base
    fleet generator inject latent failure pressure into the OT signals."""
    src = source or SyntheticOTSource()
    if isinstance(src, SyntheticOTSource):
        return src.fetch(transformers, operational_history, stress_factor=stress_factor)
    return src.fetch(transformers, operational_history)
