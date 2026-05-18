from __future__ import annotations

import os
from collections.abc import Mapping

import plotly.graph_objects as go
import streamlit as st

from src.agents.scenario_agent import ScenarioState, new_state, run_scenario_turn
from src.data.loader import load_seed_data
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header, footer_badge, inject_global_styles, metric_row

st.set_page_config(page_title="Scenario Simulator", page_icon="⚡", layout="wide")

HORIZONS = (1, 3, 5, 10)
DEFAULT_LOADING_PCT = 95
DEFAULT_DEFER_YEARS = 0
DEFAULT_AMBIENT_DELTA_C = 0


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sync_controls_from_state(state: ScenarioState, *, force: bool = False) -> None:
    inputs = state.get("inputs") if isinstance(state.get("inputs"), Mapping) else {}
    control_values = {
        "scenario_loading_pct": int(round(_coerce_float(inputs.get("loading_pct")) or DEFAULT_LOADING_PCT)),
        "scenario_defer_years": int(round(_coerce_float(inputs.get("defer_years")) or DEFAULT_DEFER_YEARS)),
        "scenario_ambient_delta_c": int(round(_coerce_float(inputs.get("ambient_delta_c")) or DEFAULT_AMBIENT_DELTA_C)),
    }
    control_values["scenario_loading_pct"] = min(max(control_values["scenario_loading_pct"], 20), 150)
    control_values["scenario_defer_years"] = min(max(control_values["scenario_defer_years"], 0), 10)
    control_values["scenario_ambient_delta_c"] = min(max(control_values["scenario_ambient_delta_c"], -10), 15)
    for key, value in control_values.items():
        if force or key not in st.session_state:
            st.session_state[key] = value


def _apply_state(state: ScenarioState) -> None:
    st.session_state.scenario_state = state
    asset_id = state.get("asset_id")
    if asset_id:
        st.session_state["scenario_asset_picker"] = asset_id
    _sync_controls_from_state(state, force=True)


def _curve_values(payload: Mapping[str, object] | None) -> list[float] | None:
    if not isinstance(payload, Mapping):
        return None
    curve = payload.get("curve")
    if not isinstance(curve, Mapping):
        return None

    values: list[float] = []
    for horizon in HORIZONS:
        raw = curve.get(horizon, curve.get(str(horizon)))
        value = _coerce_float(raw)
        if value is None:
            return None
        values.append(value)
    return values


def _curve_chart(state: ScenarioState) -> go.Figure | None:
    baseline_values = _curve_values(state.get("baseline_curve"))
    scenario_values = _curve_values(state.get("last_result"))
    if not baseline_values or not scenario_values:
        return None

    labels = [f"{h}y" for h in HORIZONS]
    fig = go.Figure()
    fig.add_bar(
        name="Baseline",
        x=labels,
        y=baseline_values,
        marker_color="#22d3ee",
        hovertemplate="%{x}<br>Baseline: %{y:.3f}<extra></extra>",
    )
    fig.add_bar(
        name="Scenario",
        x=labels,
        y=scenario_values,
        marker_color="#a78bfa",
        hovertemplate="%{x}<br>Scenario: %{y:.3f}<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        height=300,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        title="Baseline vs scenario risk curve",
    )
    fig.update_yaxes(title_text="Failure probability", tickformat=".1%")
    fig.update_xaxes(title_text="Horizon")
    return fig


def _run_turn(user_msg: str) -> None:
    with st.spinner("Running scenario…"):
        next_state, _ = run_scenario_turn(st.session_state.scenario_state, user_msg)
    _apply_state(next_state)
    st.rerun()


def _render_results(state: ScenarioState) -> None:
    last_result = state.get("last_result")
    if not isinstance(last_result, Mapping):
        return

    with st.container(border=True):
        st.markdown("#### Scenario results")
        inputs = state.get("inputs") if isinstance(state.get("inputs"), Mapping) else {}
        asset_id = state.get("asset_id") or inputs.get("asset_id") or "Unknown asset"
        loading = int(round(_coerce_float(inputs.get("loading_pct")) or DEFAULT_LOADING_PCT))
        defer = int(round(_coerce_float(inputs.get("defer_years")) or DEFAULT_DEFER_YEARS))
        ambient = int(round(_coerce_float(inputs.get("ambient_delta_c")) or DEFAULT_AMBIENT_DELTA_C))
        st.caption(f"{asset_id} • {loading}% loading • defer {defer} years • ambient {ambient:+}°C")

        if last_result.get("error"):
            st.error(str(last_result["error"]))

        scenario_p5 = _coerce_float(last_result.get("scenario_p5"))
        delta_p5 = _coerce_float(last_result.get("delta_p5"))
        if scenario_p5 is not None and delta_p5 is not None:
            metric_row(
                [
                    ("Scenario P(failure, 5y)", f"{scenario_p5:.3f}", "5-year horizon"),
                    ("Δ vs baseline", f"{delta_p5:+.3f}", f"{delta_p5:+.3f}"),
                ]
            )
        else:
            st.info("Run a scenario to populate 5-year risk metrics.")

        chart = _curve_chart(state)
        if chart is None:
            st.info("Run a scenario to see the chart.")
        else:
            st.plotly_chart(chart, use_container_width=True)


def main() -> None:
    inject_global_styles()
    branded_header(
        "Scenario Simulator",
        "Chat with the agent to model load, deferred maintenance, and ambient stress",
    )

    asset_ids = sorted(load_seed_data()["transformers"]["id"].tolist())
    if not asset_ids:
        st.error("No transformer assets are available.")
        footer_badge("Mock LLM")
        render_chat_panel()
        return

    st.session_state.setdefault("scenario_asset_picker", asset_ids[0])

    left_col, right_col = st.columns([35, 65], gap="large")

    with left_col:
        st.markdown("### Scenario inputs")
        with st.container(border=True):
            selected_asset = st.selectbox(
                "Transformer asset",
                asset_ids,
                key="scenario_asset_picker",
                help="Switching assets resets the simulator to a fresh scenario state.",
            )
            current_state = st.session_state.get("scenario_state")
            if not isinstance(current_state, dict) or current_state.get("asset_id") != selected_asset:
                _apply_state(new_state(selected_asset))
            else:
                _sync_controls_from_state(current_state)

            loading = st.slider("Loading %", 20, 150, key="scenario_loading_pct")
            defer = st.slider("Defer years", 0, 10, key="scenario_defer_years")
            ambient = st.slider("Ambient Δ°C", -10, 15, key="scenario_ambient_delta_c")

            st.caption("Use the sliders for quick studies, then ask follow-up questions in the simulator chat.")
            if st.button("▶ Run scenario", use_container_width=True):
                prompt = f"Simulate {selected_asset} at {loading}% loading deferred {defer} years with ambient {ambient:+}C"
                _run_turn(prompt)
            if st.button("♻ Reset", use_container_width=True):
                _apply_state(new_state(selected_asset))
                st.rerun()

    state: ScenarioState = st.session_state.scenario_state

    with right_col:
        _render_results(state)
        st.markdown("### Scenario conversation")
        if state.get("history"):
            for message in state["history"]:
                role = message.get("role", "assistant")
                with st.chat_message(role):
                    st.markdown(str(message.get("content", "")))
        else:
            st.info("Run the sliders or ask the simulator for a what-if analysis to get started.")

        user_msg = st.chat_input("Ask the simulator anything…", key="scenario_page_chat_input")
        if user_msg:
            _run_turn(user_msg)

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")
    render_chat_panel()


if __name__ == "__main__":
    main()
