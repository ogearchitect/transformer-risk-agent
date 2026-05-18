from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HORIZONS, MODEL_DIR, REFERENCE_YEAR
from src.data.loader import load_seed_data
from src.models.health_index import compute_health_index

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lifelines import WeibullAFTFitter
except Exception:  # pragma: no cover
    WeibullAFTFitter = None


FEATURES = [
    "age_years",
    "mva_rating",
    "voltage_class_kv",
    "avg_loading_pct",
    "through_faults",
    "ambient_temp_c",
    "health_index",
    "substation_importance",
    # OT-injected features (see src/data/ot_ingest.py)
    "ot_hotspot_max_c",
    "ot_hours_above_110c",
    "ot_dh2_per_day",
    "ot_h2o_ppm",
    "ot_cooling_availability_pct",
    "ot_oltc_ops_30d",
    "ot_i2t_total_yr",
]

OT_FEATURE_DEFAULTS = {
    "ot_hotspot_max_c": 95.0,
    "ot_hours_above_110c": 0.0,
    "ot_dh2_per_day": 0.05,
    "ot_h2o_ppm": 10.0,
    "ot_cooling_availability_pct": 97.0,
    "ot_oltc_ops_30d": 20,
    "ot_i2t_total_yr": 0.0,
}


class _FallbackModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FallbackModel":
        self.mean_ = float(np.clip(y.mean(), 0.01, 0.99))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(
            self.mean_
            + 0.002 * (X["age_years"] - X["age_years"].mean())
            - 0.003 * (X["health_index"] - 50)
            + 0.0008 * (X.get("ot_hotspot_max_c", pd.Series([95.0] * len(X))) - 95.0)
            - 0.0008 * (X.get("ot_cooling_availability_pct", pd.Series([97.0] * len(X))) - 97.0),
            0.01,
            0.99,
        )
        return np.column_stack([1 - p, p])

    def save_model(self, path: str) -> None:
        Path(path).write_text(json.dumps({"fallback_mean": self.mean_}), encoding="utf-8")

    def load_model(self, path: str) -> None:
        self.mean_ = json.loads(Path(path).read_text(encoding="utf-8"))["fallback_mean"]


@lru_cache(maxsize=1)
def build_training_frame() -> pd.DataFrame:
    data = load_seed_data()
    tx = data["transformers"].copy()
    dga = data["dga_history"]
    pd_hist = data["pd_history"]
    bushing = data["bushing_history"]
    maintenance = data["maintenance_log"]
    op = data["operational_history"]
    thermal = data.get("thermal_telemetry", pd.DataFrame())
    online_dga = data.get("online_dga", pd.DataFrame())
    oltc = data.get("oltc_telemetry", pd.DataFrame())
    cooling = data.get("cooling_status", pd.DataFrame())
    faults = data.get("fault_events", pd.DataFrame())

    latest_op = op.sort_values("date").groupby("id").tail(1).set_index("id")

    def _latest_by_id(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=cols)
        return (
            df.sort_values("date")
            .groupby("id")
            .tail(1)
            .set_index("id")[cols]
        )

    thermal_latest = _latest_by_id(thermal, ["hotspot_max_c", "hours_above_110c"])
    dga_latest = _latest_by_id(online_dga, ["h2_rate_ppm_per_day", "h2o_ppm"])
    oltc_latest = _latest_by_id(oltc, ["ops_count"])
    cooling_latest = _latest_by_id(cooling, ["availability_pct"])
    if not faults.empty:
        i2t_yr = faults.groupby("id")["i2t_a2s"].sum()
    else:
        i2t_yr = pd.Series(dtype=float)

    rows = []
    for _, row in tx.iterrows():
        aid = row["id"]
        scores = compute_health_index(
            dga[dga["id"] == aid],
            pd_hist[pd_hist["id"] == aid],
            bushing[bushing["id"] == aid],
            maintenance[maintenance["id"] == aid],
            age_years=float(REFERENCE_YEAR - row["vintage_year"]),
        )
        op_last = latest_op.loc[aid]
        rows.append(
            {
                "id": aid,
                "age_years": float(REFERENCE_YEAR - row["vintage_year"]),
                "mva_rating": row["mva_rating"],
                "voltage_class_kv": row["voltage_class_kv"],
                "avg_loading_pct": op_last["avg_loading_pct"],
                "through_faults": op_last["through_faults"],
                "ambient_temp_c": op_last["ambient_temp_c"],
                "health_index": scores["health_index"],
                "substation_importance": row["substation_importance"],
                "ot_hotspot_max_c": float(
                    thermal_latest.loc[aid, "hotspot_max_c"]
                    if aid in thermal_latest.index
                    else OT_FEATURE_DEFAULTS["ot_hotspot_max_c"]
                ),
                "ot_hours_above_110c": float(
                    thermal_latest.loc[aid, "hours_above_110c"]
                    if aid in thermal_latest.index
                    else OT_FEATURE_DEFAULTS["ot_hours_above_110c"]
                ),
                "ot_dh2_per_day": float(
                    dga_latest.loc[aid, "h2_rate_ppm_per_day"]
                    if aid in dga_latest.index
                    else OT_FEATURE_DEFAULTS["ot_dh2_per_day"]
                ),
                "ot_h2o_ppm": float(
                    dga_latest.loc[aid, "h2o_ppm"]
                    if aid in dga_latest.index
                    else OT_FEATURE_DEFAULTS["ot_h2o_ppm"]
                ),
                "ot_cooling_availability_pct": float(
                    cooling_latest.loc[aid, "availability_pct"]
                    if aid in cooling_latest.index
                    else OT_FEATURE_DEFAULTS["ot_cooling_availability_pct"]
                ),
                "ot_oltc_ops_30d": int(
                    oltc_latest.loc[aid, "ops_count"]
                    if aid in oltc_latest.index
                    else OT_FEATURE_DEFAULTS["ot_oltc_ops_30d"]
                ),
                "ot_i2t_total_yr": float(
                    i2t_yr.loc[aid] if aid in i2t_yr.index else OT_FEATURE_DEFAULTS["ot_i2t_total_yr"]
                ),
                "fail_within_5y": row["fail_within_5y"],
                "duration_years": row["duration_years"],
                "event_observed": row["event_observed"],
            }
        )

    return pd.DataFrame(rows)


def _new_xgb() -> object:
    if XGBClassifier is None:
        return _FallbackModel()
    return XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


def train_and_save_models() -> tuple[object, object | None, pd.DataFrame]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df = build_training_frame()

    X = train_df[FEATURES]
    y = train_df["fail_within_5y"]
    model = _new_xgb()
    model.fit(X, y)
    model.save_model(str(MODEL_DIR / "xgb_failure.json"))

    aft = None
    if WeibullAFTFitter is not None:
        aft = WeibullAFTFitter()
        aft.fit(train_df[["duration_years", "event_observed"] + FEATURES], duration_col="duration_years", event_col="event_observed")
        with open(MODEL_DIR / "weibull_aft.pkl", "wb") as f:
            pickle.dump(aft, f)
    else:  # pragma: no cover
        with open(MODEL_DIR / "weibull_aft.pkl", "wb") as f:
            pickle.dump({"fallback": True}, f)

    return model, aft, train_df


@lru_cache(maxsize=1)
def load_models() -> tuple[object, object | None]:
    xgb_path = MODEL_DIR / "xgb_failure.json"
    aft_path = MODEL_DIR / "weibull_aft.pkl"

    if not xgb_path.exists() or not aft_path.exists():
        train_and_save_models()

    model = _new_xgb()
    model.load_model(str(xgb_path))

    with open(aft_path, "rb") as f:
        aft = pickle.load(f)
    return model, aft if hasattr(aft, "predict_survival_function") else None


def _hotspot_aging_multiplier(hotspot_c: float, hours_over_110: float) -> float:
    """IEEE C57.91 thermal aging.

    Insulation aging acceleration factor doubles for every ~6°C above the 110°C
    reference. Returns a multiplier in [1.0, ~6.0] that's applied to the base
    failure probability — capturing the physics the statistical model can only
    learn weakly from monthly averages.
    """
    if hotspot_c <= 110.0:
        return 1.0
    delta = hotspot_c - 110.0
    base_mult = float(2 ** (delta / 6.0))
    if hours_over_110 > 100.0:
        base_mult *= 1.0 + 0.3 * min(1.0, (hours_over_110 - 100.0) / 200.0)
    return float(np.clip(base_mult, 1.0, 6.0))


def predict_failure_probabilities(features: pd.DataFrame, model: object | None = None, aft: object | None = None) -> dict[int, float]:
    model = model or load_models()[0]
    base_p = float(np.clip(model.predict_proba(features[FEATURES])[0, 1], 0.001, 0.999))

    hotspot = float(features["ot_hotspot_max_c"].iloc[0]) if "ot_hotspot_max_c" in features.columns else 95.0
    hrs_over = float(features["ot_hours_above_110c"].iloc[0]) if "ot_hours_above_110c" in features.columns else 0.0
    physics_mult = _hotspot_aging_multiplier(hotspot, hrs_over)

    horizons = {}
    if aft is not None:
        surv = aft.predict_survival_function(features[FEATURES], times=HORIZONS)
        for h in HORIZONS:
            s = float(np.clip(surv.loc[h].iloc[0], 0.0, 1.0))
            p = 1 - s
            blended = 0.5 * p + 0.5 * base_p * (h / 5) ** 0.5
            horizons[h] = float(np.clip(blended * physics_mult, 0.0, 0.999))
    else:
        for h in HORIZONS:
            horizons[h] = float(np.clip(base_p * (h / 5) ** 0.7 * physics_mult, 0.0, 0.999))
    return horizons


def get_asset_features(asset_id: str) -> pd.DataFrame:
    train_df = build_training_frame().set_index("id")
    return train_df.loc[[asset_id], FEATURES]


def get_asset_probability(asset_id: str, horizon_years: int) -> float:
    model, aft = load_models()
    features = get_asset_features(asset_id)
    probs = predict_failure_probabilities(features, model=model, aft=aft)
    return probs[horizon_years]


if __name__ == "__main__":
    train_and_save_models()
    print(f"Saved artifacts to {MODEL_DIR}")
