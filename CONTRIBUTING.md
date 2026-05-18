# Contributing

Thanks for your interest in contributing to Transformer Risk Agent. This repository is a synthetic-data demo of an agentic transformer-risk application, so contributions should preserve reproducibility and local usability.

## Dev setup

- Python 3.11 or newer is required.
- Install dependencies from the repo root:
  ```bash
  pip install -r requirements.txt
  ```

## First-time pipeline

Run the initial workflow from the repo root in this order:

1. `make data` - generate deterministic parquet seed data in `data/seed/`.
2. `make train` - optionally train the failure models and write artifacts to `models/artifacts/`; if skipped, the app auto-trains on first import when artifacts are missing, which is slower.
3. `make demo` - launch the Streamlit app.
4. `make test` - run the test suite.

## Running tests

- `make test`
- `pytest -q`
- Single test example: `pytest tests/test_failure_prob.py::test_failure_probability_bounds -q`
- Invoke `pytest` from the repo root; `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`.

## Conventions

- No linter or formatter is configured; please do not add one without discussion.
- Use `from __future__ import annotations` at the top of every new Python module.
- Asset IDs are always `T-####` (4 digits, zero-padded).
- Import paths and constants from `src/config.py` (`SEED_DIR`, `MODEL_DIR`, `HORIZONS`, `REFERENCE_YEAR`); do not hardcode `2026` or recompute ages.
- The project is deterministic by design: data generation uses `seed=42` and a fixed mid-year anchor. Preserve both.
- Agent code must keep working without an OpenAI key; `src/agents/llm.py` falls back to `MockChatModel` when `OPENAI_API_KEY` is empty, and CI runs in mock mode.
- LangChain tools must use `@tool` with a Pydantic `args_schema` and be registered in `ALL_TOOLS` at the bottom of `src/agents/tools.py`.
- New Streamlit pages go under `src/ui/pages/` and should follow the `N_Page_Name.py` numeric-prefix convention.

## Do not commit

- `.env`
- Locally regenerated `data/seed/*.parquet` or `models/artifacts/*` unless you are deliberately refreshing the committed baselines (the repo intentionally tracks these so CI and `make demo` work without re-running `make data`/`make train`)

## Pull requests

- Branch from `main`.
- Ensure `pytest -q` passes locally before opening or updating the PR.
- Keep PRs scoped to one change or tightly related set of changes.
- Update docs and `.github/copilot-instructions.md` if you change architecture-level conventions.

## Deeper architecture

See `.github/copilot-instructions.md` for the request-flow walkthrough, model layer details, and caching conventions.
