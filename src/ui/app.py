from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.tools import rank_fleet
from src.data.loader import load_seed_data
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header, metric_row

st.set_page_config(page_title="Transformer Risk Agent", page_icon="⚡", layout="wide")


def main() -> None:
    branded_header()
    render_chat_panel()

    data = load_seed_data()
    tx = data["transformers"]
    ranked = pd.DataFrame(rank_fleet.invoke({"top_n": 500, "sort_by": "monetized_risk_usd"})["rows"])
    merged = tx.merge(ranked, on="id", how="left")

    avg_hi = merged["id"].head(100).apply(lambda x: 100 - merged.loc[merged["id"] == x, "p_fail_5y"].iloc[0] * 100).mean()
    high_risk = int((merged["p_fail_1y"] > 0.25).sum())
    expected_fail_1y = merged["p_fail_1y"].sum()
    monetized_risk = merged["monetized_risk_usd"].sum()

    metric_row(
        [
            ("Fleet Size", f"{len(merged)}", "seeded"),
            ("Avg Health Index", f"{avg_hi:.1f}", "+0.0"),
            ("High Risk Assets", f"{high_risk}", "1y > 25%"),
            ("1y Expected Failures", f"{expected_fail_1y:.1f}", "sum P(fail)"),
            ("Total Monetized Risk", f"${monetized_risk/1e6:.1f}M", "1-year"),
        ]
    )

    fig = px.scatter(
        merged,
        x="substation_importance",
        y="p_fail_1y",
        color="p_fail_1y",
        size="mva_rating",
        hover_data=["id", "voltage_class_kv", "region"],
        color_continuous_scale="Turbo",
        title="Fleet Risk Heatmap",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    table = merged[["id", "voltage_class_kv", "mva_rating", "p_fail_1y", "p_fail_5y", "monetized_risk_usd"]].sort_values(
        "monetized_risk_usd", ascending=False
    )
    st.subheader("Risk-Ranked Fleet")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.page_link("pages/1_Asset_Deep_Dive.py", label="Open Asset Deep Dive →", icon="📈")


if __name__ == "__main__":
    main()
