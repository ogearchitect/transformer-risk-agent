from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SEED_DIR = DATA_DIR / "seed"
MODEL_DIR = ROOT_DIR / "models" / "artifacts"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
REFERENCE_YEAR = int(os.getenv("REFERENCE_YEAR", "2026"))

HORIZONS = (1, 3, 5, 10)
