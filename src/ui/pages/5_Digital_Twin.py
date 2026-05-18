from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_seed_data
from src.twin import SCENARIO_LIBRARY, TransformerTwin, build_scenario
from src.twin.adt_sync import DEFAULT_MODEL_ID, adt_mirror
from src.ui.chat_panel import render_chat_panel
from src.ui.components import (
    PLOTLY_TEMPLATE,
    branded_header,
    footer_badge,
    inject_global_styles,
    metric_row,
)

st.set_page_config(page_title="Digital Twin", page_icon="🪞", layout="wide")


def _trajectory_figure(traj: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=traj["t_hours"], y=traj["ambient_c"],
        name="Ambient", mode="lines", line={"color": "#64748b", "width": 1.5, "dash": "dot"},
    )
    fig.add_scatter(
        x=traj["t_hours"], y=traj["top_oil_c"],
        name="Top-oil", mode="lines", line={"color": "#22d3ee", "width": 2.5},
    )
    fig.add_scatter(
        x=traj["t_hours"], y=traj["hotspot_c"],
        name="Hot-spot", mode="lines", line={"color": "#a78bfa", "width": 2.5},
        fill="tonexty", fillcolor="rgba(167,139,250,0.10)",
    )
    fig.add_hline(y=110, line={"color": "#f87171", "dash": "dash"}, annotation_text="110 °C ref")
    fig.add_hline(y=140, line={"color": "#dc2626", "dash": "dash"}, annotation_text="140 °C limit")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        title="Twin thermal trajectory",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        xaxis_title="Hours",
        yaxis_title="°C",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def _load_chart(traj: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=traj["t_hours"], y=traj["load_pu"],
        name="Load p.u.", mode="lines", line={"color": "#facc15", "width": 2},
        fill="tozeroy", fillcolor="rgba(250,204,21,0.15)",
    )
    fig.add_hline(y=1.0, line={"color": "#94a3b8", "dash": "dot"}, annotation_text="1.0 p.u.")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=220,
        title="Load + cooling profile",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        yaxis={"title": "p.u.", "range": [0, max(1.8, float(traj["load_pu"].max()) + 0.1)]},
    )
    fig.add_scatter(
        x=traj["t_hours"], y=traj["cooling_pct"] / 100.0,
        name="Cooling avail. (×0.01)", mode="lines",
        line={"color": "#22d3ee", "width": 1.5, "dash": "dash"},
        yaxis="y2",
    )
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0.5, 1.05], "showgrid": False, "title": "Cooling"})
    return fig


def _aging_chart(traj: pd.DataFrame) -> go.Figure:
    from src.twin.aging import aging_acceleration_factor

    faa = aging_acceleration_factor(traj["hotspot_c"].to_numpy())
    cum_aging = np.cumsum(faa) * float(np.diff(traj["t_hours"]).mean())
    fig = go.Figure()
    fig.add_scatter(
        x=traj["t_hours"], y=faa,
        name="F_AA (instantaneous)", mode="lines",
        line={"color": "#f87171", "width": 2},
    )
    fig.add_scatter(
        x=traj["t_hours"], y=cum_aging,
        name="Cumulative aging hours", mode="lines",
        line={"color": "#a78bfa", "width": 2}, yaxis="y2",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=240,
        title="Insulation aging (IEEE C57.91)",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        yaxis_title="F_AA (×normal)",
        yaxis2={"overlaying": "y", "side": "right", "title": "Cum. equivalent hours", "showgrid": False},
    )
    return fig


def _adt_panel(state, twin: TransformerTwin) -> None:
    mirror = adt_mirror()
    endpoint = os.environ.get("ADT_ENDPOINT", "").strip()
    if mirror.enabled and endpoint:
        st.success(f"Azure Digital Twins mirror connected — {endpoint}")
    else:
        st.warning("Azure Digital Twins mirror is in local-only mode. Set `ADT_ENDPOINT` and grant the container app's MSI 'Azure Digital Twins Data Owner' to enable.")

    cols = st.columns([1, 1, 2])
    sync_clicked = cols[0].button("Sync twin to ADT", type="primary", disabled=not mirror.enabled)
    query_clicked = cols[1].button("Run sample ADT query", disabled=not mirror.enabled)
    sample_query = cols[2].text_input(
        "ADT query",
        value=f"SELECT T FROM digitaltwins T WHERE T.$metadata.$model = '{DEFAULT_MODEL_ID}' AND T.hotspotC > 130",
        label_visibility="collapsed",
    )

    if sync_clicked and mirror.enabled:
        payload = {
            "assetId": twin.asset_id,
            "voltageClassKv": int(twin.asset_row.get("voltage_class_kv", 0)),
            "mvaRating": float(twin.asset_row.get("mva_rating", 0.0)),
            "installYear": int(twin.asset_row.get("vintage_year", 0)),
            "coolingType": str(twin.asset_row.get("cooling_type", "ONAF")),
            "healthIndex": state.health_index,
            "hotspotC": state.peak_hotspot_c,
            "topOilC": state.peak_top_oil_c,
            "rulYears": state.rul_years_with_scenario,
            "agingPerUnitLife": state.per_unit_life_consumed_event,
            "lastCalibrationRmseC": float(state.calibration_rmse_c) if state.calibration_rmse_c == state.calibration_rmse_c else 0.0,
            "lastScenarioName": state.scenario_name,
            "projectedPFailure5y": state.projected_p_failure_5y,
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
        }
        result = mirror.upsert_twin(
            twin.asset_id,
            payload,
            telemetry={
                "h2Ppm": state.gas_accumulated_h2_ppm,
                "ch4Ppm": state.gas_accumulated_ch4_ppm,
                "c2h4Ppm": state.gas_accumulated_c2h4_ppm,
            },
        )
        if result.get("ok"):
            st.success(f"Synced twin **{twin.asset_id}** to ADT.")
        else:
            st.error(f"Sync failed: {result}")

    if query_clicked and mirror.enabled:
        results = mirror.query(sample_query)
        if not results:
            st.info("No twins matched (sync some first).")
        else:
            st.dataframe(pd.DataFrame(results), use_container_width=True)


def main() -> None:
    inject_global_styles()
    branded_header(
        "Digital Twin",
        "IEEE C57.91 physics simulator + Azure Digital Twins mirror",
    )

    data = load_seed_data()
    asset_ids = sorted(data["transformers"]["id"].tolist())

    pick_col, scen_col, days_col = st.columns([2, 2, 1])
    aid = pick_col.selectbox("Transformer", asset_ids, index=asset_ids.index("T-0042") if "T-0042" in asset_ids else 0, key="twin_asset")
    scenario_name = scen_col.selectbox(
        "Scenario",
        list(SCENARIO_LIBRARY.keys()),
        index=list(SCENARIO_LIBRARY.keys()).index("heat_wave"),
        format_func=lambda n: SCENARIO_LIBRARY[n].description[:60] + (" …" if len(SCENARIO_LIBRARY[n].description) > 60 else ""),
        key="twin_scenario",
    )
    repeat_days = days_col.number_input("Repeat days", min_value=1, max_value=14, value=1, step=1, key="twin_repeat")

    with st.spinner("Calibrating twin against last-12-mo OT data and integrating thermal ODE…"):
        twin = TransformerTwin.from_asset_id(aid)
        scenario = build_scenario(scenario_name, repeat_days=int(repeat_days))
        state = twin.run(scenario)

    summary = state.to_summary_dict()
    metric_row(
        [
            ("Peak hotspot", f"{summary['peak_hotspot_c']:.1f} °C", "≥110 °C accelerates aging" if summary["peak_hotspot_c"] >= 110 else "Normal"),
            ("Aging (equiv. hrs)", f"{summary['aging_hours_equivalent']:.0f} h", f"×{state.per_unit_life_consumed_event * 100:.3f}% life used"),
            ("RUL (baseline)", f"{summary['rul_years_steady_state']:.1f} y", "If continuous baseline"),
            ("RUL (with scenario)", f"{summary['rul_years_with_scenario']:.1f} y", f"Δ {summary['rul_delta_years']:+.2f} y"),
            ("Projected p_fail 5y", f"{summary['projected_p_failure_5y']:.1%}", f"Δ {summary['projected_p_failure_delta']:+.1%}"),
            ("Twin calibration", (f"±{summary['calibration_rmse_c']:.1f} °C" if summary['calibration_rmse_c'] is not None else "n/a"), f"{summary['calibration_samples']} samples"),
        ]
    )

    st.markdown(f"**Scenario:** _{scenario.description}_  (×{int(repeat_days)} day{'s' if repeat_days > 1 else ''}, {scenario.hours * int(repeat_days)} hours simulated)")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(_trajectory_figure(state.trajectory), use_container_width=True, key="twin_trajectory")
    with right:
        st.plotly_chart(_load_chart(state.trajectory), use_container_width=True, key="twin_load")
        st.plotly_chart(_aging_chart(state.trajectory), use_container_width=True, key="twin_aging")

    gas_col, coef_col = st.columns([2, 1])
    with gas_col:
        gas_df = pd.DataFrame(
            {
                "Gas": ["H₂", "CH₄", "C₂H₄"],
                "Accumulated (ppm)": [
                    state.gas_accumulated_h2_ppm,
                    state.gas_accumulated_ch4_ppm,
                    state.gas_accumulated_c2h4_ppm,
                ],
                "Onset °C": [130, 220, 280],
            }
        )
        st.markdown("#### Projected DGA gas generation")
        st.dataframe(gas_df, use_container_width=True, hide_index=True)
    with coef_col:
        c = state.coefficients
        st.markdown("#### Calibrated coefficients")
        st.markdown(
            f"""
            | Field | Value |
            |---|---|
            | θ_to,rated | {c.theta_to_rated_c:.1f} °C |
            | Δθ_hs,rated | {c.delta_theta_hs_rated_c:.1f} °C |
            | R | {c.R:.2f} |
            | n | {c.n:.2f} |
            | τ_to | {c.tau_to_min:.0f} min |
            | τ_w | {c.tau_w_min:.0f} min |
            """
        )

    st.markdown("### Azure Digital Twins mirror")
    _adt_panel(state, twin)

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")
    render_chat_panel()


if __name__ == "__main__":
    main()
