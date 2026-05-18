from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import SEED_DIR


@lru_cache(maxsize=1)
def load_seed_data() -> dict[str, pd.DataFrame]:
    base = {
        "transformers": pd.read_parquet(SEED_DIR / "transformers.parquet"),
        "dga_history": pd.read_parquet(SEED_DIR / "dga_history.parquet"),
        "pd_history": pd.read_parquet(SEED_DIR / "pd_history.parquet"),
        "bushing_history": pd.read_parquet(SEED_DIR / "bushing_history.parquet"),
        "maintenance_log": pd.read_parquet(SEED_DIR / "maintenance_log.parquet"),
        "operational_history": pd.read_parquet(SEED_DIR / "operational_history.parquet"),
    }
    ot_tables = {
        "thermal_telemetry": SEED_DIR / "thermal_telemetry.parquet",
        "online_dga": SEED_DIR / "online_dga.parquet",
        "oltc_telemetry": SEED_DIR / "oltc_telemetry.parquet",
        "cooling_status": SEED_DIR / "cooling_status.parquet",
        "fault_events": SEED_DIR / "fault_events.parquet",
    }
    for name, path in ot_tables.items():
        if path.exists():
            base[name] = pd.read_parquet(path)
    return base
