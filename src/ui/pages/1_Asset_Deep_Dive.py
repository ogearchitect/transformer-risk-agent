from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.tools import explain_transformer_risk, get_failure_probability, get_health_index
from src.data.loader import load_seed_data
from src.models.dga_diagnostics import diagnose_latest
from src.models.failure_prob import (
    FEATURES,
    _hotspot_aging_multiplier,
    get_asset_features,
    load_models,
    predict_failure_probabilities,
)
from src.ui.chat_panel import render_chat_panel
from src.ui.components import (
    PLOTLY_TEMPLATE,
    branded_header,
    duval_triangle_plot,
    footer_badge,
    health_gauge,
    inject_global_styles,
    metric_row,
)

st.set_page_config(page_title="Asset Deep Dive", page_icon="⚡", layout="wide")


def _ot_lift_predictions(asset_id: str) -> dict[str, float]:
    """Return baseline (no-OT) vs OT-enriched 5-year probability for the asset.

    Baseline uses the same trained model but with OT features held at their
    fleet-wide defaults — so the only difference is the actual OT telemetry
    for THIS asset. Highlights the "what would we miss without OT data" story.
    """
    from src.models.failure_prob import OT_FEATURE_DEFAULTS

    feats = get_asset_features(asset_id)
    model, aft = load_models()
    enriched = predict_failure_probabilities(feats, model=model, aft=aft)
    baseline_feats = feats.copy()
    for key, value in OT_FEATURE_DEFAULTS.items():
        if key in baseline_feats.columns:
            baseline_feats[key] = value
    baseline = predict_failure_probabilities(baseline_feats, model=model, aft=aft)
    return {"baseline_p5": baseline[5], "ot_p5": enriched[5]}


def _render_ot_section(asset_id: str, data: Mapping[str, pd.DataFrame]) -> None:
    thermal = data.get("thermal_telemetry", pd.DataFrame())
    online_dga = data.get("online_dga", pd.DataFrame())
    cooling = data.get("cooling_status", pd.DataFrame())
    oltc = data.get("oltc_telemetry", pd.DataFrame())
    faults = data.get("fault_events", pd.DataFrame())

    if thermal.empty and online_dga.empty:
        st.caption("OT telemetry not available — run `make data` to regenerate seed.")
        return

    thermal_a = thermal[thermal["id"] == asset_id].sort_values("date") if not thermal.empty else pd.DataFrame()
    dga_a = online_dga[online_dga["id"] == asset_id].sort_values("date") if not online_dga.empty else pd.DataFrame()
    cool_a = cooling[cooling["id"] == asset_id].sort_values("date") if not cooling.empty else pd.DataFrame()
    oltc_a = oltc[oltc["id"] == asset_id].sort_values("date") if not oltc.empty else pd.DataFrame()
    faults_a = faults[faults["id"] == asset_id].sort_values("date") if not faults.empty else pd.DataFrame()

    latest_thermal = thermal_a.iloc[-1] if not thermal_a.empty else None
    latest_dga = dga_a.iloc[-1] if not dga_a.empty else None
    latest_cool = cool_a.iloc[-1] if not cool_a.empty else None
    latest_oltc = oltc_a.iloc[-1] if not oltc_a.empty else None
    i2t_yr = float(faults_a["i2t_a2s"].sum()) if not faults_a.empty else 0.0

    if latest_thermal is not None:
        hotspot = float(latest_thermal["hotspot_max_c"])
        hrs110 = float(latest_thermal["hours_above_110c"])
        mult = _hotspot_aging_multiplier(hotspot, hrs110)
        hotspot_delta = f"{mult:.1f}× aging" if mult > 1.0 else "Normal aging"
    else:
        hotspot, hrs110, mult, hotspot_delta = 0.0, 0.0, 1.0, "—"

    metric_row(
        [
            ("Hotspot (peak)", f"{hotspot:.1f} °C", hotspot_delta),
            ("Hours > 110 °C / mo", f"{hrs110:.0f} h", "Insulation stress"),
            (
                "Online H₂ rate",
                f"{float(latest_dga['h2_rate_ppm_per_day']):.2f} ppm/day" if latest_dga is not None else "—",
                str(latest_dga["status"]).title() if latest_dga is not None else "",
            ),
            (
                "Cooling availability",
                f"{float(latest_cool['availability_pct']):.1f}%" if latest_cool is not None else "—",
                "Higher is better",
            ),
            (
                "OLTC ops / month",
                f"{int(latest_oltc['ops_count'])}" if latest_oltc is not None else "—",
                "Contact wear leading indicator",
            ),
            ("I²t YTD", f"{i2t_yr / 1e6:.2f} MA²s", f"{len(faults_a)} through-faults"),
        ]
    )

    try:
        from src.twin import TransformerTwin
        from src.twin.calibration import calibrate_thermal_twin

        calib = calibrate_thermal_twin(asset_id)
        twin = TransformerTwin.from_asset_id(asset_id)
        twin_state = twin.run("normal_day")
        rmse_str = f"±{calib.rmse_c:.1f} °C" if calib.samples > 0 and calib.rmse_c == calib.rmse_c else "n/a"
        aging_str = f"{twin_state.aging_hours_equivalent * 365.0:.0f} h / yr"
        rul_str = f"{twin_state.rul_years_with_scenario:.1f} y"
        metric_row(
            [
                ("Twin calibration RMSE", rmse_str, f"{calib.samples} OT samples"),
                ("Twin: projected aging / yr", aging_str, "Normal-day scenario × 365"),
                ("Twin: RUL projection", rul_str, "Run page 5 for scenarios"),
            ]
        )
    except Exception:
        pass

    try:
        lift = _ot_lift_predictions(asset_id)
        delta = lift["ot_p5"] - lift["baseline_p5"]
        msg = (
            f"**Without OT** the model predicts **{lift['baseline_p5']:.1%}** 5-year failure probability; "
            f"**with this asset's OT telemetry** it predicts **{lift['ot_p5']:.1%}** "
            f"(**Δ = {delta:+.1%}**)."
        )
        (st.error if delta > 0.05 else st.success if delta < -0.05 else st.info)(msg)
    except Exception:
        pass

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not thermal_a.empty:
            fig = go.Figure()
            fig.add_scatter(
                x=thermal_a["date"], y=thermal_a["top_oil_max_c"],
                name="Top-oil max", mode="lines+markers",
                line={"color": "#22d3ee", "width": 2},
            )
            fig.add_scatter(
                x=thermal_a["date"], y=thermal_a["hotspot_max_c"],
                name="Hot-spot max", mode="lines+markers",
                line={"color": "#a78bfa", "width": 2},
            )
            fig.add_hline(y=110, line={"color": "#f87171", "dash": "dash"}, annotation_text="110 °C ref")
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=280,
                title="Thermal telemetry (12 mo)",
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
                yaxis_title="°C",
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if not dga_a.empty:
            fig = go.Figure()
            fig.add_scatter(
                x=dga_a["date"], y=dga_a["h2_ppm"],
                name="H₂ ppm", mode="lines+markers", yaxis="y1",
                line={"color": "#22d3ee", "width": 2},
            )
            fig.add_scatter(
                x=dga_a["date"], y=dga_a["h2_rate_ppm_per_day"],
                name="ΔH₂ /day", mode="lines+markers", yaxis="y2",
                line={"color": "#facc15", "width": 2, "dash": "dot"},
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=280,
                title="Online DGA (continuous monitor)",
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
                yaxis={"title": "H₂ ppm"},
                yaxis2={"title": "ΔH₂ ppm/day", "overlaying": "y", "side": "right"},
            )
            st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        if not cool_a.empty:
            fig = px.area(
                cool_a, x="date", y="availability_pct",
                template=PLOTLY_TEMPLATE, title="Cooling availability",
            )
            fig.update_traces(line={"color": "#22d3ee"}, fillcolor="rgba(34,211,238,0.15)")
            fig.update_yaxes(range=[60, 101])
            fig.update_layout(height=240, margin={"l": 0, "r": 0, "t": 40, "b": 0}, yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)
    with chart_col4:
        if not oltc_a.empty:
            fig = go.Figure()
            fig.add_bar(
                x=oltc_a["date"], y=oltc_a["ops_count"],
                name="OLTC ops", marker_color="#a78bfa",
            )
            fig.add_scatter(
                x=oltc_a["date"], y=oltc_a["contact_wear_score"] * 50,
                name="Wear score ×50", mode="lines+markers",
                line={"color": "#facc15", "width": 2}, yaxis="y2",
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=240,
                title="OLTC operations + wear",
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
                yaxis_title="Ops / month",
                yaxis2={"overlaying": "y", "side": "right", "showgrid": False, "range": [0, 60]},
            )
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    inject_global_styles()
    branded_header(
        "Asset Deep Dive",
        "Per-transformer health, DGA diagnostics, and live OT telemetry",
    )

    data = load_seed_data()
    aid = st.selectbox(
        "Select transformer",
        sorted(data["transformers"]["id"].tolist()),
        index=0,
        key="asset_deep_dive_picker",
    )

    hi = get_health_index.invoke({"id": aid})
    probs = {h: get_failure_probability.invoke({"id": aid, "horizon_years": h})["p_failure"] for h in [1, 3, 5, 10]}

    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(health_gauge(hi["health_index"]), use_container_width=True)
    prob_df = pd.DataFrame({"horizon": [f"{h}y" for h in probs.keys()], "p_failure": list(probs.values())})
    bar = px.bar(prob_df, x="horizon", y="p_failure", template=PLOTLY_TEMPLATE, title="Failure probability by horizon")
    bar.update_traces(marker_color="#a78bfa")
    bar.update_yaxes(tickformat=".1%", range=[0, 1])
    bar.update_layout(height=300, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    c2.plotly_chart(bar, use_container_width=True)

    st.markdown("### Live OT telemetry")
    _render_ot_section(aid, data)

    st.markdown("### Offline DGA (lab-sampled)")
    dga = data["dga_history"][data["dga_history"]["id"] == aid].sort_values("date")
    if not dga.empty:
        diag = diagnose_latest(dga)
        st.info(f"DGA diagnosis: **{diag.fault_type}**, severity **{diag.severity}**, trend **{diag.trend}**")
        fig = px.line(dga, x="date", y=["h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm"], title="DGA trend (10 years)", template=PLOTLY_TEMPLATE)
        fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 40, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        latest = dga.iloc[-1]
        st.plotly_chart(
            duval_triangle_plot(latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"]),
            use_container_width=True,
        )

    st.markdown("### Partial discharge and bushing")
    pd_col, bush_col = st.columns(2)
    pd_hist = data["pd_history"][data["pd_history"]["id"] == aid]
    pd_col.plotly_chart(
        px.line(pd_hist, x="date", y="pd_pc", title="Partial discharge (pC)", template=PLOTLY_TEMPLATE).update_layout(
            height=280, margin={"l": 0, "r": 0, "t": 40, "b": 0}
        ),
        use_container_width=True,
    )
    bush = data["bushing_history"][data["bushing_history"]["id"] == aid]
    bush_col.plotly_chart(
        px.line(bush, x="date", y="tan_delta_pct", color="phase", title="Bushing tan δ by phase", template=PLOTLY_TEMPLATE).update_layout(
            height=280, margin={"l": 0, "r": 0, "t": 40, "b": 0}
        ),
        use_container_width=True,
    )

    maint = data["maintenance_log"][data["maintenance_log"]["id"] == aid].sort_values("date")
    if not maint.empty:
        st.markdown("### Maintenance history")
        maint_fig = px.scatter(
            maint, x="date", y="event_type", color="event_type", size="effectiveness",
            template=PLOTLY_TEMPLATE, title="Maintenance events",
            hover_data={"description": True, "effectiveness": ":.2f"},
        )
        maint_fig.update_layout(height=240, margin={"l": 0, "r": 0, "t": 40, "b": 0})
        st.plotly_chart(maint_fig, use_container_width=True)

    expl = explain_transformer_risk.invoke({"id": aid})
    st.markdown("### Agent narrative")
    st.write(expl.get("narrative", ""))

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")
    render_chat_panel()


if __name__ == "__main__":
    main()
