from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


PLOTLY_TEMPLATE = "plotly_dark"


def branded_header() -> None:
    st.markdown("## ⚡ Transformer Risk Agent — Agentic AI for Asset Health & Fleet Strategy")


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def health_gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Health Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3ddc97"},
                "steps": [
                    {"range": [0, 40], "color": "#8b0000"},
                    {"range": [40, 70], "color": "#f39c12"},
                    {"range": [70, 100], "color": "#2ecc71"},
                ],
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=250)
    return fig


def duval_triangle_plot(ch4: float, c2h2: float, c2h4: float) -> go.Figure:
    total = max(ch4 + c2h2 + c2h4, 1e-9)
    a, b, c = [100 * v / total for v in (ch4, c2h2, c2h4)]
    fig = go.Figure(
        go.Scatterternary(
            a=[a], b=[b], c=[c], mode="markers+text", text=["Current"], marker={"size": 10, "color": "cyan"}
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        ternary={
            "sum": 100,
            "aaxis": {"title": "CH4%"},
            "baxis": {"title": "C2H2%"},
            "caxis": {"title": "C2H4%"},
        },
        height=360,
        title="Duval Triangle 1 (current point)",
    )
    return fig
