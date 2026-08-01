from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Designing the Autonomous Curb", page_icon="🚏", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_av_pudo_events.csv"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_MAP = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
PUDO_COLORS = {"Pickup": "#C62828", "Dropoff": "#F28E9C", "Net PUDO balance": "#6A1B9A"}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    required = {
        "pullover_type", "day_of_week", "hour", "adjusted_point",
        "desired_point", "curb_geo", "polygon_type",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["pullover_type"] = df["pullover_type"].astype(str).str.strip().str.title()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    numeric_days = pd.to_numeric(df["day_of_week"], errors="coerce")
    labels = numeric_days.map(DAY_MAP)
    original = df["day_of_week"].astype(str).str.strip().str.title()
    df["day_of_week"] = pd.Categorical(labels.fillna(original), categories=DAY_ORDER, ordered=True)
    df["polygon_type"] = df["polygon_type"].astype(str).str.strip()
    df["period"] = pd.cut(
        df["hour"].astype(float),
        bins=[-1, 5, 10, 15, 18, 22, 24],
        labels=["Overnight", "Morning", "Midday", "Afternoon", "Evening peak", "Late evening"],
    )
    return df


def parse_point(value):
    if pd.isna(value):
        return np.nan, np.nan
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(nums) < 2:
        return np.nan, np.nan
    a, b = map(float, nums[:2])
    return (b, a) if abs(a) > 90 else (a, b)


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["adjusted_point"].apply(parse_point)
    out["latitude"] = parsed.str[0]
    out["longitude"] = parsed.str[1]
    return out


def event_balance(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Return pickup, dropoff, and signed pickup-minus-dropoff counts by group."""
    counts = (
        df.groupby(group_cols + ["pullover_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Pickup", "Dropoff"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["Net PUDO balance"] = counts["Pickup"] - counts["Dropoff"]
    return counts


def build_adaptive_allocation(hourly: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"hour": range(24)}).merge(hourly, on="hour", how="left").fillna(0)
    peak = max(float(base["events"].max()), 1.0)
    demand_index = base["events"] / peak
    base["AV PUDO"] = 10 + 30 * demand_index
    base["Loading/delivery"] = 10 + 12 * np.exp(-0.5 * ((base["hour"] - 9) / 3.0) ** 2)
    base["Transit & accessible access"] = 18.0
    base["Parking/storage"] = 100 - base["AV PUDO"] - base["Loading/delivery"] - base["Transit & accessible access"]
    return base


st.title("Designing the Autonomous Curb")
st.caption("Interactive archive of the synthetic pick-up and drop-off analysis used in the TRB paper.")

try:
    raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"The dashboard cannot find the dataset. Expected path: `{DATA_PATH}`")
    st.stop()
except Exception as exc:
    st.error(f"The dataset was found but could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    selected_days = st.multiselect("Day of week", DAY_ORDER, default=DAY_ORDER)
    base_types = sorted(raw["pullover_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Underlying event records", base_types, default=base_types)
    hour_range = st.slider("Hour range", 0, 23, (0, 23))
    polygon_values = sorted(raw["polygon_type"].dropna().astype(str).unique().tolist())
    selected_polygons = st.multiselect("Analytical polygon", polygon_values, default=polygon_values)

filtered = raw[
    raw["day_of_week"].astype(str).isin(selected_days)
    & raw["pullover_type"].isin(selected_types)
    & raw["hour"].between(hour_range[0], hour_range[1])
    & raw["polygon_type"].astype(str).isin(selected_polygons)
].copy()

pickups = int((filtered["pullover_type"] == "Pickup").sum())
dropoffs = int((filtered["pullover_type"] == "Dropoff").sum())
net_balance = pickups - dropoffs
peak_hour = filtered.groupby("hour").size().idxmax() if not filtered.empty else None
peak_count = filtered.groupby("hour").size().max() if not filtered.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Synthetic events", f"{len(filtered):,}")
m2.metric("Pick-ups", f"{pickups:,}")
m3.metric("Drop-offs", f"{dropoffs:,}")
m4.metric("Net PUDO balance", f"{net_balance:+,}", help="Pick-ups minus drop-offs. Positive values indicate a pick-up surplus; negative values indicate a drop-off surplus.")
m5.metric("Peak filtered hour", "—" if peak_hour is None else f"{int(peak_hour):02d}:00", f"{int(peak_count):,} events" if peak_hour is not None else None)

st.info(
    "These are synthetic, privacy-preserving observations calibrated to generalized spatial and temporal patterns. "
    "Net PUDO balance is calculated as pick-ups minus drop-offs within the selected grouping."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Temporal sorting", "Spatial sorting", "Adaptive curb scenario", "Data explorer"
])

with tab1:
    c1, c2 = st.columns(2)
    by_type = pd.DataFrame({
        "Event type": ["Pickup", "Dropoff", "Net PUDO balance"],
        "Events": [pickups, dropoffs, net_balance],
    })
    fig_type = px.bar(
        by_type, x="Event type", y="Events", color="Event type",
        color_discrete_map=PUDO_COLORS, title="Synthetic PUDO activity and net balance",
    )
    fig_type.add_hline(y=0, line_color="black", line_width=1)
    fig_type.update_layout(showlegend=False)
    c1.plotly_chart(fig_type, use_container_width=True)

    by_polygon = filtered.groupby("polygon_type", observed=True).size().reset_index(name="Events")
    by_polygon["Share"] = by_polygon["Events"] / max(by_polygon["Events"].sum(), 1)
    fig_poly = px.bar(
        by_polygon, x="polygon_type", y="Events",
        text=by_polygon["Share"].map(lambda x: f"{x:.1%}"),
        title="Spatial concentration by analytical polygon",
        labels={"polygon_type": "Analytical polygon"},
    )
    c2.plotly_chart(fig_poly, use_container_width=True)

    day_balance = event_balance(filtered, ["day_of_week"])
    day_long = day_balance.melt(
        id_vars="day_of_week",
        value_vars=["Pickup", "Dropoff", "Net PUDO balance"],
        var_name="Event type", value_name="Events",
    )
    fig_day = px.bar(
        day_long, x="day_of_week", y="Events", color="Event type",
        color_discrete_map=PUDO_COLORS, barmode="group",
        title="Pick-ups, drop-offs, and net balance by day",
        labels={"day_of_week": "Day"},
    )
    fig_day.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_day, use_container_width=True)

with tab2:
    hourly_balance = event_balance(filtered, ["hour"])
    hourly_long = hourly_balance.melt(
        id_vars="hour",
        value_vars=["Pickup", "Dropoff", "Net PUDO balance"],
        var_name="Event type", value_name="Events",
    )
    selected_series = st.multiselect(
        "Temporal event series",
        ["Pickup", "Dropoff", "Net PUDO balance"],
        default=["Pickup", "Dropoff", "Net PUDO balance"],
        key="temporal_series",
    )
    hourly_plot = hourly_long[hourly_long["Event type"].isin(selected_series)]
    fig_hour = px.line(
        hourly_plot, x="hour", y="Events", color="Event type", markers=True,
        color_discrete_map=PUDO_COLORS, title="Hourly PUDO activity and net balance",
        labels={"hour": "Hour of day"},
    )
    fig_hour.add_hline(y=0, line_color="black", line_width=1)
    fig_hour.update_xaxes(dtick=1)
    st.plotly_chart(fig_hour, use_container_width=True)
    st.caption("Net PUDO balance = pick-ups − drop-offs. Positive values indicate more pick-ups; negative values indicate more drop-offs.")

    balance_heat = event_balance(filtered, ["day_of_week", "hour"])
    metric = st.radio(
        "Heat-map event type",
        ["Total events", "Pickup", "Dropoff", "Net PUDO balance"],
        horizontal=True,
    )
    if metric == "Total events":
        balance_heat[metric] = balance_heat["Pickup"] + balance_heat["Dropoff"]
    pivot = balance_heat.pivot(index="day_of_week", columns="hour", values=metric).reindex(DAY_ORDER).fillna(0)
    color_scale = "RdBu" if metric == "Net PUDO balance" else "Reds"
    midpoint = 0 if metric == "Net PUDO balance" else None
    fig_heat = px.imshow(
        pivot, aspect="auto", color_continuous_scale=color_scale,
        color_continuous_midpoint=midpoint,
        labels={"x": "Hour", "y": "Day", "color": metric},
        title=f"Day-hour heat map: {metric}",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    period_counts = filtered.groupby("period", observed=True).size().reset_index(name="Events")
    st.dataframe(period_counts, use_container_width=True, hide_index=True)

with tab3:
    mapped = add_coordinates(filtered).dropna(subset=["latitude", "longitude"])
    if mapped.empty:
        st.warning("The adjusted-point field could not be parsed into map coordinates. Polygon summaries remain available below.")
    else:
        fig_map = px.scatter_map(
            mapped, lat="latitude", lon="longitude", color="pullover_type",
            color_discrete_map=PUDO_COLORS,
            hover_data=["day_of_week", "hour", "polygon_type"],
            zoom=14, height=600, title="Synthetic adjusted PUDO locations",
        )
        fig_map.update_layout(map_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Spatial density of synthetic adjusted PUDO locations")
        heat_radius = st.slider("Heatmap radius", 5, 40, 18, key="spatial_heat_radius")
        fig_density = px.density_map(
            mapped, lat="latitude", lon="longitude", radius=heat_radius,
            center={"lat": float(mapped["latitude"].mean()), "lon": float(mapped["longitude"].mean())},
            zoom=14, height=600,
            hover_data=["pullover_type", "day_of_week", "hour", "polygon_type"],
            title="Heat map of synthetic adjusted PUDO activity",
        )
        fig_density.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=55, b=0))
        st.plotly_chart(fig_density, use_container_width=True)
        st.caption("The heat map shows relative concentration within the filtered synthetic dataset, not observed citywide demand.")

    polygon_balance = event_balance(filtered, ["polygon_type"])
    polygon_long = polygon_balance.melt(
        id_vars="polygon_type",
        value_vars=["Pickup", "Dropoff", "Net PUDO balance"],
        var_name="Event type", value_name="Events",
    )
    spatial_series = st.multiselect(
        "Polygon event series",
        ["Pickup", "Dropoff", "Net PUDO balance"],
        default=["Pickup", "Dropoff", "Net PUDO balance"],
        key="polygon_series",
    )
    fig_cross = px.bar(
        polygon_long[polygon_long["Event type"].isin(spatial_series)],
        x="polygon_type", y="Events", color="Event type",
        color_discrete_map=PUDO_COLORS, barmode="group",
        title="PUDO activity and net balance by polygon",
        labels={"polygon_type": "Analytical polygon"},
    )
    fig_cross.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_cross, use_container_width=True)

with tab4:
    hourly_total = filtered.groupby("hour").size().reset_index(name="events")
    allocation = build_adaptive_allocation(hourly_total)
    long = allocation.melt(
        id_vars=["hour", "events"],
        value_vars=["Parking/storage", "Loading/delivery", "Transit & accessible access", "AV PUDO"],
        var_name="Curb function", value_name="Share",
    )
    polar_colors = {
        "Parking/storage": "#2F78B7", "Loading/delivery": "#F5A623",
        "Transit & accessible access": "#3A9D3D", "AV PUDO": "#C62828",
    }
    fig_rose = go.Figure()
    for function in polar_colors:
        part = long[long["Curb function"] == function]
        fig_rose.add_trace(go.Barpolar(
            r=part["Share"], theta=part["hour"] * 15, width=[12] * len(part),
            name=function, marker_color=polar_colors[function],
            marker_line_color="white", marker_line_width=0.7,
        ))
    fig_rose.update_layout(
        title="Data-calibrated adaptive curb allocation scenario", barmode="stack", height=700,
        polar=dict(
            radialaxis=dict(range=[0, 100], ticksuffix="%"),
            angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                             tickvals=[0, 90, 180, 270], ticktext=["12 AM", "6 AM", "12 PM", "6 PM"]),
        ),
    )
    st.plotly_chart(fig_rose, use_container_width=True)
    st.caption("Planning scenario only: AV PUDO rises with filtered demand; loading is morning-weighted; transit/accessibility is protected; parking is residual.")

with tab5:
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_synthetic_av_pudo_events.csv", mime="text/csv",
    )
