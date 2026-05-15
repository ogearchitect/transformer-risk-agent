from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.tools import rank_fleet, recommend_replacement_plan, recommend_spare_strategy
from src.ui.chat_panel import render_chat_panel
from src.ui.components import PLOTLY_TEMPLATE, branded_header

st.set_page_config(page_title="Fleet Strategy", page_icon="⚡", layout="wide")


def main() -> None:
    branded_header()
    render_chat_panel()

    budget = st.slider("Budget cap ($M)", 5.0, 60.0, 15.0)
    horizon = st.slider("Planning horizon (years)", 1, 15, 5)

    plan = recommend_replacement_plan.invoke({"budget_musd": budget, "horizon_years": horizon})
    plan_df = pd.DataFrame(plan["projects"])

    st.subheader("Replacement Plan")
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    if not plan_df.empty:
        g = px.timeline(plan_df, x_start="replacement_year", x_end="replacement_year", y="id", color="voltage_class_kv", template=PLOTLY_TEMPLATE)
        st.plotly_chart(g, use_container_width=True)

    st.subheader("Spare Strategy")
    spare_rows = [recommend_spare_strategy.invoke({"voltage_class_kv": kv}) for kv in [69, 138, 230, 345, 500]]
    st.dataframe(pd.DataFrame(spare_rows), use_container_width=True, hide_index=True)

    ranked = pd.DataFrame(rank_fleet.invoke({"top_n": 150, "sort_by": "monetized_risk_usd"})["rows"])
    ranked["capex_musd"] = ranked["replacement_cost_usd"].cumsum() / 1e6
    ranked["risk_musd"] = ranked["monetized_risk_usd"].cumsum() / 1e6
    p = px.line(ranked, x="capex_musd", y="risk_musd", title="CapEx vs Risk Pareto", template=PLOTLY_TEMPLATE)
    st.plotly_chart(p, use_container_width=True)


if __name__ == "__main__":
    main()
