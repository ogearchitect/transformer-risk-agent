from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.tools import _build_risk_table, rank_fleet
from src.config import REFERENCE_YEAR
from src.data.loader import load_seed_data
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header, footer_badge, inject_global_styles, metric_row

st.set_page_config(page_title="Transformer Risk Agent", page_icon="⚡", layout="wide")

_BRAND_SCALE = ["#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#4ade80"]


def _fallback_risk_table(tx: pd.DataFrame) -> pd.DataFrame:
    age_years = REFERENCE_YEAR - tx["vintage_year"]
    age_score = (age_years - age_years.min()) / max(age_years.max() - age_years.min(), 1)
    importance_score = tx["substation_importance"] / max(tx["substation_importance"].max(), 1)
    consequence_score = tx["failure_consequence_usd"] / max(tx["failure_consequence_usd"].max(), 1)

    p_fail_1y = (0.04 + 0.32 * age_score + 0.18 * importance_score + 0.11 * consequence_score).clip(0.02, 0.85)
    p_fail_3y = (p_fail_1y * 1.45).clip(0.03, 0.92)
    p_fail_5y = (p_fail_1y * 1.75).clip(0.05, 0.96)
    p_fail_10y = (p_fail_1y * 2.15).clip(0.08, 0.98)
    monetized_risk = p_fail_1y * tx["failure_consequence_usd"]
    replacement_cost = tx["mva_rating"] * 10_500

    return pd.DataFrame(
        {
            "id": tx["id"],
            "p_fail_1y": p_fail_1y,
            "p_fail_3y": p_fail_3y,
            "p_fail_5y": p_fail_5y,
            "p_fail_10y": p_fail_10y,
            "monetized_risk_usd": monetized_risk,
            "replacement_cost_usd": replacement_cost,
            "risk_reduction_per_usd": (monetized_risk * 0.75) / replacement_cost.clip(lower=1.0),
        }
    )


def _load_dashboard_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = load_seed_data()["transformers"].copy()
    ranked_preview = pd.DataFrame()
    try:
        ranked_preview = pd.DataFrame(rank_fleet.invoke({"top_n": 50, "sort_by": "monetized_risk_usd"})["rows"])
        risk_table = _build_risk_table()
    except Exception:
        risk_table = _fallback_risk_table(tx)
        ranked_preview = risk_table.sort_values("monetized_risk_usd", ascending=False).head(50)

    merged = tx.merge(
        risk_table[
            [
                "id",
                "p_fail_1y",
                "p_fail_3y",
                "p_fail_5y",
                "p_fail_10y",
                "monetized_risk_usd",
                "replacement_cost_usd",
                "risk_reduction_per_usd",
            ]
        ],
        on="id",
        how="left",
    )
    return merged, ranked_preview


def main() -> None:
    inject_global_styles()
    branded_header(
        subtitle="Operational intelligence for transformer health, fleet exposure, and capital allocation in one control room."
    )
    render_chat_panel()

    merged, ranked_preview = _load_dashboard_frame()
    avg_hi = 100 - merged["p_fail_5y"].mean() * 100
    high_risk = int((merged["p_fail_1y"] > 0.25).sum())
    expected_fail_1y = merged["p_fail_1y"].sum()
    monetized_risk = merged["monetized_risk_usd"].sum()

    metric_row(
        [
            ("Fleet Size", f"{len(merged)}", "seeded"),
            ("Avg Health Index", f"{avg_hi:.1f}", "+0.0"),
            ("High Risk Assets", f"{high_risk}", "1y > 25%"),
            ("1y Expected Failures", f"{expected_fail_1y:.1f}", "sum P(fail)"),
            ("Total Monetized Risk", f"${monetized_risk / 1e6:.1f}M", "1-year"),
        ]
    )

    if not ranked_preview.empty:
        lead = ranked_preview.iloc[0]
        st.caption(
            f"Live watchlist: {lead['id']} currently leads the fleet by 1-year monetized risk (${lead['monetized_risk_usd'] / 1e6:.2f}M)."
        )

    left, right = st.columns([2, 1])
    with left:
        heatmap = px.scatter(
            merged,
            x="substation_importance",
            y="p_fail_1y",
            size="mva_rating",
            color="p_fail_5y",
            size_max=30,
            hover_name="id",
            hover_data={
                "voltage_class_kv": True,
                "region": True,
                "mva_rating": ":.1f",
                "p_fail_1y": ":.1%",
                "p_fail_5y": ":.1%",
                "substation_importance": ":.0f",
            },
            color_continuous_scale=_BRAND_SCALE,
            title="Fleet Risk Heatmap — Criticality vs. Near-Term Failure Risk",
        )
        heatmap.update_layout(template=PLOTLY_TEMPLATE, height=440, coloraxis_colorbar_title="5y risk")
        heatmap.update_xaxes(title_text="Substation importance score")
        heatmap.update_yaxes(title_text="1-year failure probability", tickformat=".0%")
        st.plotly_chart(heatmap, use_container_width=True)

    with right:
        risk_by_region = (
            merged.assign(high_risk=merged["p_fail_1y"] > 0.25)
            .groupby("region", as_index=False)["high_risk"]
            .sum()
            .rename(columns={"high_risk": "high_risk_assets"})
            .sort_values("high_risk_assets", ascending=True)
        )
        region_bar = px.bar(
            risk_by_region,
            x="high_risk_assets",
            y="region",
            orientation="h",
            text="high_risk_assets",
            color="high_risk_assets",
            color_continuous_scale=_BRAND_SCALE,
            title="Risk by Region — Assets Above 25% 1-Year Risk",
        )
        region_bar.update_layout(template=PLOTLY_TEMPLATE, height=440, coloraxis_showscale=False)
        region_bar.update_xaxes(title_text="High-risk asset count")
        region_bar.update_yaxes(title_text="")
        st.plotly_chart(region_bar, use_container_width=True)

    region_mix = (
        merged.assign(voltage_band=merged["voltage_class_kv"].map(lambda kv: f"{int(kv)} kV"))
        .groupby(["region", "voltage_band"], as_index=False)
        .agg(asset_count=("id", "count"), avg_p_fail_5y=("p_fail_5y", "mean"))
    )
    treemap = px.treemap(
        region_mix,
        path=["region", "voltage_band"],
        values="asset_count",
        color="avg_p_fail_5y",
        color_continuous_scale=_BRAND_SCALE,
        title="Region Distribution — Fleet Composition by Voltage Class",
        hover_data={"asset_count": True, "avg_p_fail_5y": ":.1%"},
    )
    treemap.update_layout(template=PLOTLY_TEMPLATE, height=430, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    st.plotly_chart(treemap, use_container_width=True)

    bubble = px.scatter(
        merged,
        x="substation_importance",
        y="monetized_risk_usd",
        size="mva_rating",
        color="p_fail_5y",
        size_max=34,
        hover_name="id",
        hover_data={
            "voltage_class_kv": True,
            "region": True,
            "mva_rating": ":.1f",
            "monetized_risk_usd": ":,.0f",
            "p_fail_5y": ":.1%",
            "substation_importance": ":.0f",
        },
        color_continuous_scale=_BRAND_SCALE,
        title="Brushable Risk Surface — Importance, Exposure, and Capacity",
    )
    bubble.update_layout(template=PLOTLY_TEMPLATE, height=470, dragmode="select", coloraxis_colorbar_title="5y risk")
    bubble.update_xaxes(title_text="Substation importance score")
    bubble.update_yaxes(title_text="Monetized risk (USD, log scale)", type="log", tickprefix="$")
    st.plotly_chart(bubble, use_container_width=True)
    st.page_link("pages/1_Asset_Deep_Dive.py", label="Open Asset Deep Dive →", icon="📈")

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")


if __name__ == "__main__":
    main()
