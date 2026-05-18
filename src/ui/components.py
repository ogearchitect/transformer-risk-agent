from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


_BRAND_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#4ade80"]
PLOTLY_TEMPLATE = "transformer_risk_dark"


def _register_plotly_template() -> None:
    if PLOTLY_TEMPLATE in pio.templates:
        return

    template = go.layout.Template(pio.templates["plotly_dark"])
    axis_style = {
        "gridcolor": "rgba(148, 163, 184, 0.16)",
        "linecolor": "rgba(148, 163, 184, 0.24)",
        "zerolinecolor": "rgba(148, 163, 184, 0.24)",
        "title_standoff": 10,
    }
    template.layout.colorway = _BRAND_COLORS
    template.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"
    template.layout.plot_bgcolor = "rgba(0, 0, 0, 0)"
    template.layout.font = {"color": "#e2e8f0", "family": "Inter, Segoe UI, sans-serif"}
    template.layout.legend = {
        "orientation": "h",
        "bgcolor": "rgba(0, 0, 0, 0)",
        "borderwidth": 0,
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    }
    template.layout.xaxis = axis_style
    template.layout.yaxis = axis_style
    pio.templates[PLOTLY_TEMPLATE] = template


_register_plotly_template()


def inject_global_styles() -> None:
    if st.session_state.get("_global_styles_injected"):
        return

    css_path = Path(__file__).resolve().parents[2] / ".streamlit" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.session_state["_global_styles_injected"] = True


def branded_header(
    title: str = "Transformer Risk Agent",
    subtitle: str = "Agentic AI for fleet resilience, consequence-aware risk, and capital planning.",
) -> None:
    inject_global_styles()
    today = datetime.now().strftime("%b %d, %Y")
    left, right = st.columns([0.76, 0.24])
    left.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-eyebrow">Fleet Command Center</div>
          <div class="hero-title">⚡ {escape(title)}</div>
          <div class="hero-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    right.markdown(
        f"""
        <div class="hero-meta">
          <span class="status-pill">● Live</span>
          <div class="hero-timestamp">As of: {escape(today)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _parse_numeric(text: str) -> float | None:
    cleaned = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = float(match.group())
    upper = cleaned.upper()
    if "B" in upper:
        value *= 1e9
    elif "M" in upper:
        value *= 1e6
    elif "K" in upper:
        value *= 1e3
    return value


def _sparkline_series(value: str, delta: str) -> list[float] | None:
    numeric_value = _parse_numeric(value)
    if numeric_value is None:
        return None

    if delta.strip().startswith("-"):
        multipliers = [1.18, 1.11, 1.07, 1.03, 1.0]
    elif delta.strip().startswith("+"):
        multipliers = [0.78, 0.86, 0.91, 0.96, 1.0]
    else:
        multipliers = [0.94, 0.98, 1.01, 0.99, 1.0]
    return [max(numeric_value * m, 0.0) for m in multipliers]


def _sparkline_figure(series: Sequence[float], delta: str) -> go.Figure:
    line_color = "#4ade80" if delta.strip().startswith("+") else "#f472b6" if delta.strip().startswith("-") else "#22d3ee"
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(series))),
            y=list(series),
            mode="lines",
            line={"color": line_color, "width": 2.4},
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.12)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=72,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for idx, (col, item) in enumerate(zip(cols, metrics)):
        label, value, delta = item[:3]
        sparkline = item[3] if len(item) > 3 else _sparkline_series(value, delta)
        tone = "negative" if delta.strip().startswith("-") else "positive" if delta.strip().startswith("+") else "neutral"
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{escape(label)}</div>
                  <div class="kpi-value">{escape(value)}</div>
                  <div class="kpi-delta {tone}">{escape(delta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if sparkline:
                slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)[:32]
                st.plotly_chart(
                    _sparkline_figure(sparkline, delta),
                    use_container_width=True,
                    config={"displayModeBar": False, "staticPlot": True},
                    key=f"kpi_spark_{slug}_{idx}_{abs(hash((label, value, delta))) % 10_000_000}",
                )


def footer_badge(model_label: str) -> None:
    st.markdown(f'<div class="footer-badge">Powered by {escape(model_label)}</div>', unsafe_allow_html=True)


def health_gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Health Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22d3ee"},
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#16a34a"},
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
            a=[a],
            b=[b],
            c=[c],
            mode="markers+text",
            text=["Current"],
            marker={"size": 10, "color": "#22d3ee"},
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
