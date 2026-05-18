from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REFERENCE_YEAR, SEED_DIR
from src.data.ot_ingest import generate_ot_data


@dataclass(frozen=True)
class FleetData:
    transformers: pd.DataFrame
    dga_history: pd.DataFrame
    pd_history: pd.DataFrame
    bushing_history: pd.DataFrame
    maintenance_log: pd.DataFrame
    operational_history: pd.DataFrame
    thermal_telemetry: pd.DataFrame
    online_dga: pd.DataFrame
    oltc_telemetry: pd.DataFrame
    cooling_status: pd.DataFrame
    fault_events: pd.DataFrame


def _clamp(a: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(a, low), high)


def generate_fleet(seed: int = 42, n_assets: int = 500) -> FleetData:
    rng = np.random.default_rng(seed=seed)
    asset_ids = [f"T-{i:04d}" for i in range(1, n_assets + 1)]
    # Use a fixed mid-year anchor for stable age/seasonality features while avoiding year-boundary skew.
    today = pd.Timestamp(datetime(REFERENCE_YEAR, 7, 1, tzinfo=timezone.utc).date())
    monthly = pd.date_range(end=today, periods=120, freq="MS")
    quarterly = pd.date_range(end=today, periods=40, freq="QS")

    voltage_classes = np.array([69, 138, 230, 345, 500])
    cooling_types = np.array(["ONAN", "ONAF", "OFAF"])
    manufacturers = np.array(["Hitachi", "Siemens", "GE", "ABB", "Mitsubishi"])
    regions = np.array(["North", "South", "East", "West", "Central"])

    vintage = rng.integers(1965, 2025, size=n_assets)
    age = today.year - vintage
    voltage = rng.choice(voltage_classes, size=n_assets, p=[0.22, 0.30, 0.25, 0.15, 0.08])
    mva = _clamp(rng.normal(180, 90, size=n_assets), 30, 900).round(1)
    criticality = _clamp(rng.normal(0.62, 0.18, n_assets), 0.1, 1.0)
    customer_count = rng.integers(3_000, 160_000, size=n_assets)
    consequence = (criticality * customer_count * rng.uniform(40, 90, size=n_assets)).round(0)

    transformers = pd.DataFrame(
        {
            "id": asset_ids,
            "mva_rating": mva,
            "voltage_class_kv": voltage,
            "vintage_year": vintage,
            "manufacturer": rng.choice(manufacturers, size=n_assets),
            "cooling_type": rng.choice(cooling_types, size=n_assets, p=[0.45, 0.35, 0.20]),
            "substation": [f"SS-{i:03d}" for i in rng.integers(1, 180, size=n_assets)],
            "region": rng.choice(regions, size=n_assets),
            "substation_importance": (criticality * 100).round(1),
            "n_minus_one_backup": rng.choice([True, False], size=n_assets, p=[0.7, 0.3]),
            "customers_served": customer_count,
            "failure_consequence_usd": consequence,
        }
    )

    fault_assets = set(rng.choice(asset_ids, size=max(1, int(n_assets * 0.12)), replace=False))

    dga_rows: list[dict] = []
    pd_rows: list[dict] = []
    bushing_rows: list[dict] = []
    maintenance_rows: list[dict] = []
    operational_rows: list[dict] = []

    failure_score = np.zeros(n_assets)

    for i, aid in enumerate(asset_ids):
        asset_age = age[i]
        is_fault = aid in fault_assets
        fault_mode = rng.choice(["thermal", "arcing", "pd"]) if is_fault else "healthy"

        base_h2 = rng.uniform(30, 150)
        base_ch4 = rng.uniform(10, 80)
        base_c2h2 = rng.uniform(0.5, 4.0)
        base_c2h4 = rng.uniform(8, 50)
        base_c2h6 = rng.uniform(6, 60)
        base_co = rng.uniform(200, 900)
        base_co2 = rng.uniform(2500, 10000)

        pd_base = rng.uniform(80, 550)
        tan_delta_base = rng.uniform(0.2, 0.9)
        cap_base = rng.uniform(260, 440)
        ir_base = rng.uniform(450, 2500)

        load_base = rng.uniform(45, 90)
        amb_base = rng.uniform(12, 33)

        fault_pressure = 0.0

        for month_idx, date in enumerate(monthly):
            growth = month_idx / max(len(monthly) - 1, 1)

            h2 = base_h2 + rng.normal(0, 8)
            ch4 = base_ch4 + rng.normal(0, 5)
            c2h2 = max(0.05, base_c2h2 + rng.normal(0, 0.5))
            c2h4 = base_c2h4 + rng.normal(0, 4)
            c2h6 = base_c2h6 + rng.normal(0, 3)

            if is_fault:
                if fault_mode == "thermal":
                    c2h4 += 120 * growth + rng.normal(0, 6)
                    ch4 += 55 * growth
                    c2h6 += 20 * growth
                    fault_pressure += 0.5 * growth
                elif fault_mode == "arcing":
                    c2h2 += 65 * growth + rng.normal(0, 1.5)
                    h2 += 90 * growth
                    c2h4 += 30 * growth
                    fault_pressure += 0.75 * growth
                elif fault_mode == "pd":
                    h2 += 170 * growth
                    ch4 += 25 * growth
                    fault_pressure += 0.4 * growth

            dga_rows.append(
                {
                    "id": aid,
                    "date": date,
                    "h2_ppm": round(max(1.0, h2), 2),
                    "ch4_ppm": round(max(1.0, ch4), 2),
                    "c2h2_ppm": round(max(0.05, c2h2), 3),
                    "c2h4_ppm": round(max(1.0, c2h4), 2),
                    "c2h6_ppm": round(max(1.0, c2h6), 2),
                    "co_ppm": round(max(50.0, base_co + rng.normal(0, 40)), 2),
                    "co2_ppm": round(max(700.0, base_co2 + rng.normal(0, 250)), 2),
                }
            )

            pd_val = pd_base + rng.normal(0, 35) + asset_age * 1.7 + (120 * growth if fault_mode == "pd" else 0)
            pd_rows.append({"id": aid, "date": date, "pd_pc": round(max(5.0, pd_val), 2)})

            load = _clamp(np.array([load_base + rng.normal(0, 8) + (8 * growth if is_fault else 0)]), 15, 135)[0]
            peak = min(160.0, load + rng.uniform(8, 25))
            ambient = amb_base + rng.normal(0, 4)
            humidity = _clamp(np.array([rng.normal(58, 17)]), 15, 98)[0]

            operational_rows.append(
                {
                    "id": aid,
                    "date": date,
                    "avg_loading_pct": round(load, 2),
                    "peak_loading_pct": round(peak, 2),
                    "through_faults": int(max(0, rng.poisson(0.35 + (0.4 if load > 90 else 0)))),
                    "ambient_temp_c": round(ambient, 2),
                    "humidity_pct": round(humidity, 2),
                }
            )

        for q_idx, date in enumerate(quarterly):
            q_growth = q_idx / max(len(quarterly) - 1, 1)
            drift = (asset_age / 60) * q_growth
            tan_delta = tan_delta_base + drift + (0.45 * q_growth if is_fault else 0) + rng.normal(0, 0.03)
            cap = cap_base + (18 * drift) + rng.normal(0, 2)
            ir = max(80.0, ir_base - (200 * drift) - (140 * q_growth if is_fault else 0) + rng.normal(0, 40))
            for phase in ["A", "B", "C"]:
                bushing_rows.append(
                    {
                        "id": aid,
                        "date": date,
                        "phase": phase,
                        "tan_delta_pct": round(max(0.05, tan_delta + rng.normal(0, 0.02)), 4),
                        "capacitance_pf": round(max(50.0, cap + rng.normal(0, 1.5)), 3),
                        "ir_gohm": round(ir + rng.normal(0, 30), 3),
                    }
                )

        events = int(rng.integers(3, 10))
        event_dates = pd.to_datetime(rng.choice(monthly, size=events, replace=False)).sort_values()
        for j, dt in enumerate(event_dates):
            etype = rng.choice(["inspection", "oil_test", "overhaul", "deferral"], p=[0.42, 0.28, 0.18, 0.12])
            score = float(_clamp(np.array([rng.normal(0.78, 0.15)]), 0.2, 1.0)[0])
            if etype == "deferral":
                score *= 0.7
                fault_pressure += 0.2
            maintenance_rows.append(
                {
                    "id": aid,
                    "date": dt,
                    "event_type": etype,
                    "description": f"{etype.replace('_', ' ').title()} event {j + 1}",
                    "effectiveness": round(score, 3),
                }
            )

        age_term = asset_age / 75
        criticality_term = transformers.loc[i, "substation_importance"] / 100
        failure_score[i] = 0.35 * age_term + 0.35 * (fault_pressure / 7) + 0.30 * criticality_term

    failure_prob = 1 / (1 + np.exp(-(failure_score * 5 - 2.6)))
    fail_within_5y = rng.binomial(1, _clamp(failure_prob, 0.01, 0.95)).astype(int)
    duration_years = _clamp(rng.weibull(1.8, n_assets) * (12 - 8 * failure_prob), 0.3, 15)

    transformers["fail_within_5y"] = fail_within_5y
    transformers["event_observed"] = fail_within_5y
    transformers["duration_years"] = duration_years.round(3)

    operational_df = pd.DataFrame(operational_rows)
    # Per-asset latent stress factor (normalised failure_score) — drives the
    # label, AND propagates into OT signals so the OT features are predictive
    # rather than independent noise. This mirrors real fleets where overloaded /
    # degraded units exhibit the OT precursors and ALSO fail more.
    fs_range = max(float(np.ptp(failure_score)), 1e-6)
    stress = pd.Series(
        np.clip((failure_score - failure_score.min()) / fs_range, 0.0, 1.0),
        index=asset_ids,
    )
    ot = generate_ot_data(transformers, operational_df, source=None, stress_factor=stress)

    return FleetData(
        transformers=transformers,
        dga_history=pd.DataFrame(dga_rows),
        pd_history=pd.DataFrame(pd_rows),
        bushing_history=pd.DataFrame(bushing_rows),
        maintenance_log=pd.DataFrame(maintenance_rows),
        operational_history=pd.DataFrame(operational_rows),
        thermal_telemetry=ot.thermal_telemetry,
        online_dga=ot.online_dga,
        oltc_telemetry=ot.oltc_telemetry,
        cooling_status=ot.cooling_status,
        fault_events=ot.fault_events,
    )


def save_seed_data(seed: int = 42, n_assets: int = 500, output_dir: Path | None = None) -> FleetData:
    fleet = generate_fleet(seed=seed, n_assets=n_assets)
    out = output_dir or SEED_DIR
    out.mkdir(parents=True, exist_ok=True)

    fleet.transformers.to_parquet(out / "transformers.parquet", index=False)
    fleet.dga_history.to_parquet(out / "dga_history.parquet", index=False)
    fleet.pd_history.to_parquet(out / "pd_history.parquet", index=False)
    fleet.bushing_history.to_parquet(out / "bushing_history.parquet", index=False)
    fleet.maintenance_log.to_parquet(out / "maintenance_log.parquet", index=False)
    fleet.operational_history.to_parquet(out / "operational_history.parquet", index=False)
    fleet.thermal_telemetry.to_parquet(out / "thermal_telemetry.parquet", index=False)
    fleet.online_dga.to_parquet(out / "online_dga.parquet", index=False)
    fleet.oltc_telemetry.to_parquet(out / "oltc_telemetry.parquet", index=False)
    fleet.cooling_status.to_parquet(out / "cooling_status.parquet", index=False)
    fleet.fault_events.to_parquet(out / "fault_events.parquet", index=False)
    return fleet


if __name__ == "__main__":
    save_seed_data()
    print(f"Seed data generated in {SEED_DIR}")
