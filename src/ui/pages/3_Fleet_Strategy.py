from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.tools import rank_fleet, recommend_replacement_plan, recommend_spare_strategy
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header, footer_badge, inject_global_styles, metric_row

st.set_page_config(page_title="Fleet Strategy", page_icon="⚡", layout="wide")

VOLTAGE_CLASSES = (69, 138, 230, 345, 500)
PARETO_TOP_N = 50  # rank_fleet max per Pydantic schema


def _safe_invoke(label: str, fn, payload: dict):
    try:
        return fn.invoke(payload), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{label}: {type(exc).__name__}: {exc}"


def _render_replacement_plan(plan: Mapping, budget_musd: float) -> None:
    projects = plan.get("projects") or []
    plan_df = pd.DataFrame(projects)

    if plan_df.empty:
        st.info("No projects fit the selected budget — try increasing the cap.")
        return

    total_capex_musd = float(plan_df["replacement_cost_usd"].sum()) / 1e6
    total_risk_reduction_musd = float(plan_df["risk_reduction_usd"].sum()) / 1e6
    avg_roi = (
        plan_df["risk_reduction_usd"].sum() / plan_df["replacement_cost_usd"].sum()
        if plan_df["replacement_cost_usd"].sum() > 0
        else 0.0
    )

    metric_row(
        [
            ("Projects", f"{len(plan_df):,}", "Within budget"),
            ("CapEx", f"${total_capex_musd:,.1f}M", f"of ${budget_musd:.0f}M cap"),
            ("Risk reduction", f"${total_risk_reduction_musd:,.1f}M", "monetised over horizon"),
            ("Avg ROI", f"{avg_roi:,.2f}×", "$ risk avoided / $ spent"),
        ]
    )

    year_voltage = (
        plan_df.assign(capex_musd=plan_df["replacement_cost_usd"] / 1e6)
        .groupby(["replacement_year", "voltage_class_kv"], as_index=False)["capex_musd"]
        .sum()
    )
    year_voltage["voltage_class_kv"] = year_voltage["voltage_class_kv"].astype(str) + " kV"
    capex_fig = px.bar(
        year_voltage,
        x="replacement_year",
        y="capex_musd",
        color="voltage_class_kv",
        template=PLOTLY_TEMPLATE,
        labels={"replacement_year": "Replacement year (offset)", "capex_musd": "CapEx ($M)", "voltage_class_kv": "Voltage"},
        title="CapEx schedule by replacement year",
    )
    capex_fig.update_layout(barmode="stack", height=320, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    st.plotly_chart(capex_fig, use_container_width=True)

    with st.expander(f"Show all {len(plan_df)} selected projects"):
        display_df = plan_df.assign(
            replacement_cost_musd=lambda d: d["replacement_cost_usd"] / 1e6,
            risk_reduction_musd=lambda d: d["risk_reduction_usd"] / 1e6,
            cumulative_spend_musd=lambda d: d["cumulative_spend_usd"] / 1e6,
        )[
            [
                "id",
                "voltage_class_kv",
                "replacement_year",
                "replacement_cost_musd",
                "risk_reduction_musd",
                "cumulative_spend_musd",
            ]
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "voltage_class_kv": st.column_config.NumberColumn("Voltage (kV)"),
                "replacement_year": st.column_config.NumberColumn("Year", format="%d"),
                "replacement_cost_musd": st.column_config.NumberColumn("CapEx ($M)", format="%.2f"),
                "risk_reduction_musd": st.column_config.NumberColumn("Risk reduced ($M)", format="%.2f"),
                "cumulative_spend_musd": st.column_config.NumberColumn("Cumulative ($M)", format="%.2f"),
            },
        )


def _render_spare_strategy() -> None:
    rows: list[dict] = []
    errors: list[str] = []
    for kv in VOLTAGE_CLASSES:
        row, err = _safe_invoke(f"spare {kv}kV", recommend_spare_strategy, {"voltage_class_kv": kv})
        if err:
            errors.append(err)
            continue
        if row:
            rows.append(row)

    if errors:
        for err in errors:
            st.warning(err)
    if not rows:
        st.info("No spare-strategy data available.")
        return

    spare_df = pd.DataFrame(rows)
    spare_df["voltage_label"] = spare_df["voltage_class_kv"].astype(str) + " kV"
    spare_df["err_low"] = (spare_df["recommended_spares"] - spare_df["ci_low"]).clip(lower=0)
    spare_df["err_high"] = (spare_df["ci_high"] - spare_df["recommended_spares"]).clip(lower=0)

    fig = go.Figure(
        go.Bar(
            x=spare_df["voltage_label"],
            y=spare_df["recommended_spares"],
            marker_color="#a78bfa",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": spare_df["err_high"],
                "arrayminus": spare_df["err_low"],
                "color": "#22d3ee",
                "thickness": 2,
            },
            hovertemplate="%{x}<br>P85: %{y}<br>CI 5–95: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
            customdata=spare_df[["ci_low", "ci_high"]].to_numpy(),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Recommended spares with 5–95% Monte Carlo CI",
        yaxis_title="Spares",
        height=300,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        spare_df[["voltage_class_kv", "recommended_spares", "ci_low", "ci_high"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_pareto() -> None:
    ranked_payload, err = _safe_invoke(
        "rank_fleet", rank_fleet, {"top_n": PARETO_TOP_N, "sort_by": "monetized_risk_usd"}
    )
    if err or not ranked_payload:
        st.warning(err or "rank_fleet returned no rows.")
        return

    ranked = pd.DataFrame(ranked_payload.get("rows") or [])
    if ranked.empty or "replacement_cost_usd" not in ranked.columns:
        st.info("No ranked-fleet data available.")
        return

    ranked = ranked.sort_values("monetized_risk_usd", ascending=False).reset_index(drop=True)
    ranked["capex_musd"] = ranked["replacement_cost_usd"].cumsum() / 1e6
    ranked["risk_musd"] = ranked["monetized_risk_usd"].cumsum() / 1e6
    ranked["rank"] = ranked.index + 1

    fig = px.line(
        ranked,
        x="capex_musd",
        y="risk_musd",
        markers=True,
        template=PLOTLY_TEMPLATE,
        labels={"capex_musd": "Cumulative CapEx ($M)", "risk_musd": "Cumulative monetised risk addressed ($M)"},
        hover_data={"rank": True, "id": True, "voltage_class_kv": True},
        title=f"CapEx vs risk Pareto (top {PARETO_TOP_N} assets by monetised risk)",
    )
    fig.update_traces(line={"color": "#22d3ee"}, marker={"color": "#a78bfa", "size": 7})
    fig.update_layout(height=340, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    inject_global_styles()
    branded_header(
        "Fleet Strategy",
        "Plan replacements, size spares, and trade off CapEx against monetised risk",
    )

    with st.container(border=True):
        c1, c2 = st.columns(2)
        budget = c1.slider("Budget cap ($M)", 5.0, 60.0, 15.0, step=1.0, key="fleet_budget_musd")
        horizon = c2.slider("Planning horizon (years)", 1, 15, 5, key="fleet_horizon_years")

    plan, plan_err = _safe_invoke(
        "recommend_replacement_plan",
        recommend_replacement_plan,
        {"budget_musd": float(budget), "horizon_years": int(horizon)},
    )

    st.markdown("### Replacement plan")
    if plan_err:
        st.error(plan_err)
    elif plan is not None:
        _render_replacement_plan(plan, float(budget))

    left, right = st.columns([55, 45], gap="large")
    with left:
        st.markdown("### Spare strategy")
        _render_spare_strategy()
    with right:
        st.markdown("### CapEx vs risk Pareto")
        _render_pareto()

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")
    render_chat_panel()


if __name__ == "__main__":
    main()
