from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.tools import explain_transformer_risk, get_failure_probability, get_health_index
from src.data.loader import load_seed_data
from src.models.dga_diagnostics import diagnose_latest
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header, duval_triangle_plot, health_gauge

st.set_page_config(page_title="Asset Deep Dive", page_icon="⚡", layout="wide")


def main() -> None:
    branded_header()
    render_chat_panel()
    data = load_seed_data()
    aid = st.selectbox("Select transformer", sorted(data["transformers"]["id"].tolist()), index=0)

    hi = get_health_index.invoke({"id": aid})
    probs = {h: get_failure_probability.invoke({"id": aid, "horizon_years": h})["p_failure"] for h in [1, 3, 5, 10]}

    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(health_gauge(hi["health_index"]), use_container_width=True)
    c2.bar_chart(pd.DataFrame({"horizon": list(probs.keys()), "p_failure": list(probs.values())}).set_index("horizon"))

    dga = data["dga_history"][data["dga_history"]["id"] == aid].sort_values("date")
    diag = diagnose_latest(dga)
    st.info(f"DGA diagnosis: **{diag.fault_type}**, severity **{diag.severity}**, trend **{diag.trend}**")

    fig = px.line(dga, x="date", y=["h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm"], title="DGA Trend")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=360)
    st.plotly_chart(fig, use_container_width=True)

    latest = dga.iloc[-1]
    st.plotly_chart(duval_triangle_plot(latest["ch4_ppm"], latest["c2h2_ppm"], latest["c2h4_ppm"]), use_container_width=True)

    pd_hist = data["pd_history"][data["pd_history"]["id"] == aid]
    st.plotly_chart(px.line(pd_hist, x="date", y="pd_pc", title="Partial Discharge Trend", template=PLOTLY_TEMPLATE), use_container_width=True)

    bush = data["bushing_history"][data["bushing_history"]["id"] == aid]
    st.plotly_chart(px.line(bush, x="date", y="tan_delta_pct", color="phase", title="Bushing tan δ by phase", template=PLOTLY_TEMPLATE), use_container_width=True)

    maint = data["maintenance_log"][data["maintenance_log"]["id"] == aid].sort_values("date")
    mt = px.timeline(maint, x_start="date", x_end="date", y="event_type", color="event_type", title="Maintenance Timeline", template=PLOTLY_TEMPLATE)
    st.plotly_chart(mt, use_container_width=True)

    expl = explain_transformer_risk.invoke({"id": aid})
    st.markdown("### Agent Narrative")
    st.write(expl["narrative"])


if __name__ == "__main__":
    main()
