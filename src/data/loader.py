from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import SEED_DIR


@lru_cache(maxsize=1)
def load_seed_data() -> dict[str, pd.DataFrame]:
    return {
        "transformers": pd.read_parquet(SEED_DIR / "transformers.parquet"),
        "dga_history": pd.read_parquet(SEED_DIR / "dga_history.parquet"),
        "pd_history": pd.read_parquet(SEED_DIR / "pd_history.parquet"),
        "bushing_history": pd.read_parquet(SEED_DIR / "bushing_history.parquet"),
        "maintenance_log": pd.read_parquet(SEED_DIR / "maintenance_log.parquet"),
        "operational_history": pd.read_parquet(SEED_DIR / "operational_history.parquet"),
    }
