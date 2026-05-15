from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agents.tools import get_failure_probability, simulate_what_if
from src.data.loader import load_seed_data
from src.ui.chat_panel import render_chat_panel
from src.ui.components import branded_header

st.set_page_config(page_title="What-If", page_icon="⚡", layout="wide")


def main() -> None:
    branded_header()
    render_chat_panel()
    ids = sorted(load_seed_data()["transformers"]["id"].tolist())
    aid = st.selectbox("Transformer", ids, index=0)

    loading = st.slider("Loading %", 20, 150, 95)
    ambient = st.slider("Ambient delta °C", -10, 15, 2)
    defer = st.slider("Defer next overhaul (years)", 0, 6, 2)

    baseline = get_failure_probability.invoke({"id": aid, "horizon_years": 5})["p_failure"]
    scenario = simulate_what_if.invoke({"id": aid, "loading_pct": loading, "defer_years": defer, "ambient_delta_c": ambient})

    df = pd.DataFrame({"Case": ["Baseline", "Scenario"], "P(failure,5y)": [baseline, scenario["scenario_p5"]]})
    c1, c2 = st.columns(2)
    c1.bar_chart(df.set_index("Case"))
    c2.metric("Δ 5-year risk", f"{scenario['delta_p5']:+.3f}")


if __name__ == "__main__":
    main()
