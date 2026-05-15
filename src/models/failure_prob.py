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
]


class _FallbackModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FallbackModel":
        self.mean_ = float(np.clip(y.mean(), 0.01, 0.99))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.mean_ + 0.002 * (X["age_years"] - X["age_years"].mean()) - 0.003 * (X["health_index"] - 50), 0.01, 0.99)
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

    latest_op = op.sort_values("date").groupby("id").tail(1).set_index("id")

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


def predict_failure_probabilities(features: pd.DataFrame, model: object | None = None, aft: object | None = None) -> dict[int, float]:
    model = model or load_models()[0]
    base_p = float(np.clip(model.predict_proba(features[FEATURES])[0, 1], 0.001, 0.999))

    horizons = {}
    if aft is not None:
        surv = aft.predict_survival_function(features[FEATURES], times=HORIZONS)
        for h in HORIZONS:
            s = float(np.clip(surv.loc[h].iloc[0], 0.0, 1.0))
            p = 1 - s
            horizons[h] = float(np.clip(0.5 * p + 0.5 * base_p * (h / 5) ** 0.5, 0.0, 0.999))
    else:
        for h in HORIZONS:
            horizons[h] = float(np.clip(base_p * (h / 5) ** 0.7, 0.0, 0.999))
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
