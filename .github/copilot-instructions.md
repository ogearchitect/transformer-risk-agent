# Copilot Instructions — Transformer Risk Agent

Agentic Streamlit demo that predicts transformer failure risk over 1/3/5/10‑year horizons using a synthetic utility fleet, then exposes results through a LangGraph/LangChain agent and a multi‑page UI.

## Setup, run, test

- Requires **Python ≥ 3.11**. `pip install -r requirements.txt`.
- Pipeline order matters the first time:
  1. `make data` → `python -m src.data.generator` writes parquet files to `data/seed/` (deterministic, `seed=42`).
  2. `make train` → `python -m src.models.failure_prob` trains XGBoost + Weibull AFT and writes `models/artifacts/{xgb_failure.json, weibull_aft.pkl}`. `load_models()` auto‑trains on first use if artifacts are missing, so this step is optional but slow on first import.
  3. `make demo` → `streamlit run src/ui/app.py` (entry point; sibling files in `src/ui/pages/` become additional Streamlit pages).
- Tests: `make test` (== `pytest -q`). Run one test with `pytest tests/test_failure_prob.py::test_failure_probability_bounds -q`. `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`, so always invoke pytest from the repo root.
- No linter/formatter is configured — don't add one unless asked.
- Docker: `docker compose up --build` serves the Streamlit app on `:8501`. `.env` is loaded via `env_file` (see `.env.example`).

## Architecture (the parts that span multiple files)

Request flow for the chat panel:
`src/ui/chat_panel.py` → `src/agents/orchestrator.handle_query` → keyword router picks one or two tools from `src/agents/tools.py` → tool calls into `src/models/*` → result string is passed as `tool_context` to `src/agents/llm.get_llm().invoke(...)` for the final natural‑language answer.

- **Orchestrator is keyword‑based, not LLM‑planned.** `handle_query` in `src/agents/orchestrator.py` matches substrings like `"top"+"risk"`, `"replacement"`, `"spare"`, `"what-if"`/`"defer"`, `"failure probability"` and uses small `_extract_*` helpers to pull asset IDs, top‑N, budget, years, and voltage out of the prompt. Adding a new intent means adding a branch here **and** a tool in `tools.py`.
- **`src/agents/graph.py` wraps the orchestrator in a one‑node LangGraph** (`StateGraph(OrchestratorState)`); it is the integration seam for future multi‑node graphs but currently just calls `handle_query`.
- **Tools are LangChain `@tool` functions with Pydantic `args_schema`** (`AssetInput`, `FailureProbInput`, `WhatIfInput`, …). Invoke them with `tool.invoke({...})` — never call the underlying function directly. `ALL_TOOLS` at the bottom of `tools.py` is the canonical registry.
- **Models layer (`src/models/`)** is plain functions over pandas DataFrames: `dga_diagnostics` (Duval Triangle 1 + Rogers/Doernenburg + IEEE C57.104), `health_index` (weighted blend → 0–100), `failure_prob` (XGBoost classifier blended with Weibull AFT survival across `HORIZONS`), `fleet_optimizer` (budget‑constrained ranking + Monte Carlo spares), `explain` (narrative + top factors).
- **LLM is optional.** `src/agents/llm.get_llm()` returns `ChatOpenAI` only when `OPENAI_API_KEY` is set and `LLM_PROVIDER != "mock"`; otherwise it returns `MockChatModel`, which deterministically echoes the tool context. Tests and CI must work without an API key.

## Conventions specific to this codebase

- **Asset IDs are always `T-####`** (4 digits, zero‑padded). The orchestrator's `_extract_asset_id` requires exactly that shape; use it as the source of truth when generating new IDs or prompts.
- **`src/config.py` is the single source for paths and constants.** Import `SEED_DIR`, `MODEL_DIR`, `HORIZONS`, and especially `REFERENCE_YEAR` from there instead of hardcoding `2026` or recomputing ages — `generator.py`, `failure_prob.py`, and `tools.py` all anchor age/feature dates to it.
- **`HORIZONS = (1, 3, 5, 10)`** is assumed throughout (`predict_failure_probabilities`, `_build_risk_table`, the `curve` field returned by tools). If you change it, audit every consumer; `FailureProbInput` also clamps `horizon_years` to `[1, 10]`.
- **Heavy work is memoized with `@lru_cache(maxsize=1)`** in `src/data/loader.load_seed_data`, `src/models/failure_prob.build_training_frame` / `load_models`, and `src/agents/tools._ctx` / `_build_risk_table`. Tests run in a single process so the cache hides regeneration cost — clear caches manually if you mutate seed data mid‑process.
- **Determinism is a feature.** `generator.py` uses `np.random.default_rng(seed=42)` and a fixed mid‑year anchor (`REFERENCE_YEAR-07-01`) to keep ages and seasonality stable. Preserve both when extending data generation.
- **Graceful degradation pattern.** `failure_prob.py` falls back to `_FallbackModel` if `xgboost` is missing and skips Weibull if `lifelines` is missing (`# pragma: no cover` branches). New optional dependencies should follow the same try/except import + fallback pattern so the demo still runs.
- **`from __future__ import annotations`** is used at the top of every module — keep it when adding new files so forward references and `X | None` syntax work on 3.11.
- **Streamlit pages live in `src/ui/pages/`** with the numeric prefix (`1_Asset_Deep_Dive.py`, …) that Streamlit uses for sidebar order. New pages should follow that naming.

## Recommended MCP servers

Copilot CLI reads MCP server config from `~/.copilot/mcp-config.json` (user‑level only — there is no project‑level file). For productive work on this repo, configure at minimum:

- **`microsoft-learn`** (HTTP, `https://learn.microsoft.com/api/mcp`) — first‑party docs lookups for Azure, .NET, Python SDKs, and the Azure services backing the demo.
- **`azure`** (stdio, `npx -y @azure/mcp@latest server start`) — query/manage Azure resources for deployment (`scripts/deploy_azure.sh`, container image, App Service, etc.). Requires `az login` before the server starts so the MCP can acquire tokens via `DefaultAzureCredential`.

Example entries to merge into `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "microsoft-learn": {
      "type": "http",
      "url": "https://learn.microsoft.com/api/mcp"
    },
    "azure": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@azure/mcp@latest", "server", "start"]
    }
  }
}
```

After editing, restart Copilot CLI (`/restart`) and verify with `/mcp`.

## Things to avoid

- Don't call `predict_failure_probabilities` or model artifacts directly from UI/agent code — go through the `tools.py` wrappers so caching and feature assembly stay consistent.
- Don't read parquet files with raw paths; use `load_seed_data()` so the LRU cache stays warm.
- Don't commit `.env`. `data/seed/*.parquet` and `models/artifacts/*` ARE committed on purpose so CI and `make demo` work without re-running `make data`/`make train`; only re-commit them when you are deliberately refreshing the baseline.
