from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents.tools import _build_risk_table
from src.data.locations import REGION_COORDINATES, fleet_with_coordinates
from src.ui.chat_panel import render_chat_panel
from src.ui.components import (
    PLOTLY_TEMPLATE,
    branded_header,
    footer_badge,
    inject_global_styles,
    metric_row,
)

st.set_page_config(page_title="Fleet Map", page_icon="🗺️", layout="wide")


COLOR_DIMENSIONS = {
    "Failure probability (5-yr)": ("p_fail_5y", "Reds", "%5y"),
    "Monetized risk ($)":          ("monetized_risk_usd", "OrRd", "$"),
    "Voltage class (kV)":          ("voltage_class_kv", "Viridis", "kV"),
    "Vintage year":                ("vintage_year", "Cividis", "yr"),
}

SIZE_DIMENSIONS = {
    "MVA rating":           "mva_rating",
    "Customers served":     "customers_served",
    "Failure consequence":  "failure_consequence_usd",
}


@st.cache_data(ttl=600, show_spinner=False)
def _enriched_fleet() -> pd.DataFrame:
    fleet = fleet_with_coordinates()
    risk = _build_risk_table()[["id", "p_fail_1y", "p_fail_3y", "p_fail_5y", "p_fail_10y", "monetized_risk_usd"]]
    return fleet.merge(risk, on="id", how="left")


def _risk_band(p: float) -> str:
    if p >= 0.15: return "Critical"
    if p >= 0.08: return "High"
    if p >= 0.03: return "Medium"
    return "Low"


def _kpis(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        st.warning("No assets match the current filters.")
        return
    crit = int((df["p_fail_5y"] >= 0.15).sum())
    high = int(((df["p_fail_5y"] >= 0.08) & (df["p_fail_5y"] < 0.15)).sum())
    total_mva = float(df["mva_rating"].sum())
    total_customers = int(df["customers_served"].sum())
    total_monetized = float(df["monetized_risk_usd"].sum())
    metric_row(
        [
            ("Assets visible",        f"{n:,}",                    f"of {len(_enriched_fleet()):,} total"),
            ("Critical (≥15% 5y)",   f"{crit:,}",                  "Immediate attention"),
            ("High (8–15% 5y)",      f"{high:,}",                  "Plan within 12 mo"),
            ("Total MVA",             f"{total_mva/1000:.1f} GVA", "Installed capacity"),
            ("Customers served",      f"{total_customers/1e6:.2f} M", "Cumulative"),
            ("Monetized risk",        f"${total_monetized/1e6:.1f} M", "1-yr exposure"),
        ]
    )


def _map_figure(
    df: pd.DataFrame,
    color_col: str,
    color_scale: str,
    size_col: str,
    style: str = "open-street-map",
) -> go.Figure:
    sizes = df[size_col].astype(float).clip(lower=1)
    s_min, s_max = sizes.min(), sizes.max()
    norm = (sizes - s_min) / max(s_max - s_min, 1.0)
    marker_size = 8 + norm * 22

    customdata = np.stack(
        [
            df["id"].values,
            df["substation"].values,
            df["region"].values,
            df["voltage_class_kv"].values,
            df["mva_rating"].values,
            df["manufacturer"].values,
            df["vintage_year"].values,
            df["customers_served"].values,
            (df["p_fail_5y"].fillna(0) * 100).values,
            (df["monetized_risk_usd"].fillna(0) / 1e6).values,
        ],
        axis=-1,
    )
    hovertemplate = (
        "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
        "Region: %{customdata[2]} · %{customdata[3]} kV · %{customdata[4]:.0f} MVA<br>"
        "Manufacturer: %{customdata[5]} · Vintage: %{customdata[6]}<br>"
        "Customers: %{customdata[7]:,.0f}<br>"
        "5-yr p(fail): %{customdata[8]:.1f}% · Monetized risk: $%{customdata[9]:.2f} M"
        "<extra></extra>"
    )

    fig = go.Figure(
        go.Scattermap(
            lat=df["latitude"], lon=df["longitude"],
            mode="markers",
            marker={
                "size": marker_size,
                "color": df[color_col],
                "colorscale": color_scale,
                "showscale": True,
                "colorbar": {"title": color_col, "x": 1.02, "thickness": 12, "len": 0.65},
                "opacity": 0.85,
            },
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="",
        )
    )
    fig.update_layout(
        map={
            "style": style,
            "zoom": 3.2,
            "center": {"lat": 39.5, "lon": -98.5},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=620,
        template=PLOTLY_TEMPLATE,
        showlegend=False,
    )
    return fig


def _region_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("region")
          .agg(
              assets=("id", "count"),
              total_mva=("mva_rating", "sum"),
              avg_p5=("p_fail_5y", "mean"),
              monetized_risk_musd=("monetized_risk_usd", lambda s: s.sum() / 1e6),
              customers=("customers_served", "sum"),
          )
          .reset_index()
          .sort_values("monetized_risk_musd", ascending=False)
    )
    agg["avg_p5"] = (agg["avg_p5"] * 100).round(1)
    agg["total_mva"] = agg["total_mva"].round(0).astype(int)
    agg["monetized_risk_musd"] = agg["monetized_risk_musd"].round(2)
    agg.columns = ["Region", "Assets", "Total MVA", "Avg p(fail 5y) %", "Monetized risk ($M)", "Customers served"]
    return agg


def main() -> None:
    inject_global_styles()
    branded_header(
        "Fleet Map",
        "Geographic view of every transformer in the fleet — colored by risk, sized by impact",
    )

    fleet = _enriched_fleet()
    fleet["risk_band"] = fleet["p_fail_5y"].fillna(0).apply(_risk_band)

    with st.sidebar:
        st.markdown("### 🗺️ Map controls")
        color_choice = st.selectbox("Color by", list(COLOR_DIMENSIONS.keys()), index=0, key="map_color")
        size_choice = st.selectbox("Size by", list(SIZE_DIMENSIONS.keys()), index=0, key="map_size")
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        regions = sorted(fleet["region"].unique().tolist())
        sel_regions = st.multiselect("Region", regions, default=regions, key="map_regions")
        kvs = sorted(fleet["voltage_class_kv"].unique().tolist())
        sel_kvs = st.multiselect("Voltage class (kV)", kvs, default=kvs, key="map_kv")
        mfrs = sorted(fleet["manufacturer"].unique().tolist())
        sel_mfrs = st.multiselect("Manufacturer", mfrs, default=mfrs, key="map_mfrs")
        bands = ["Critical", "High", "Medium", "Low"]
        sel_bands = st.multiselect("Risk band (5-yr)", bands, default=bands, key="map_bands")

    color_col, color_scale, _suffix = COLOR_DIMENSIONS[color_choice]
    size_col = SIZE_DIMENSIONS[size_choice]

    filtered = fleet[
        fleet["region"].isin(sel_regions)
        & fleet["voltage_class_kv"].isin(sel_kvs)
        & fleet["manufacturer"].isin(sel_mfrs)
        & fleet["risk_band"].isin(sel_bands)
    ].copy()

    _kpis(filtered)

    map_col, side_col = st.columns([3, 1])
    with map_col:
        if filtered.empty:
            st.info("No transformers match — widen your filters in the sidebar.")
        else:
            fig = _map_figure(filtered, color_col, color_scale, size_col)
            st.plotly_chart(fig, use_container_width=True, key="fleet_map")
            st.caption(
                f"Showing **{len(filtered):,}** of {len(fleet):,} transformers across "
                f"**{filtered['substation'].nunique()}** substations in "
                f"**{filtered['region'].nunique()}** regions. Coordinates are simulated "
                "(region capital + deterministic per-substation/asset jitter); swap for a real GIS feed in production."
            )

    with side_col:
        st.markdown("#### Region anchors")
        anchors = pd.DataFrame(
            [{"Region": r, "Anchor city": c, "Lat": lat, "Lon": lon}
             for r, (lat, lon, c) in REGION_COORDINATES.items()]
        )
        st.dataframe(anchors, use_container_width=True, hide_index=True)

        st.markdown("#### Risk band counts")
        band_counts = filtered["risk_band"].value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0).astype(int)
        st.dataframe(band_counts.rename_axis("Band").reset_index().rename(columns={"risk_band": "Assets"}),
                     use_container_width=True, hide_index=True)

    st.markdown("### Regional rollup")
    summary = _region_summary(filtered)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("No data to summarize.")

    with st.expander(f"🔎 Asset table ({len(filtered):,} rows)", expanded=False):
        cols = [
            "id", "substation", "region", "voltage_class_kv", "mva_rating", "manufacturer",
            "vintage_year", "customers_served", "p_fail_5y", "monetized_risk_usd", "latitude", "longitude",
        ]
        show = filtered[cols].copy()
        show["p_fail_5y"] = (show["p_fail_5y"] * 100).round(1)
        show["monetized_risk_usd"] = (show["monetized_risk_usd"] / 1e6).round(2)
        show.columns = ["Asset", "Substation", "Region", "kV", "MVA", "Manufacturer",
                        "Vintage", "Customers", "p(fail 5y) %", "Monetized ($M)", "Lat", "Lon"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")
    render_chat_panel()


if __name__ == "__main__":
    main()
