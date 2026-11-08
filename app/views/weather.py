import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import MUTED, SEASON_COLORS, card, load_weather_daily, load_zone_day, page_header, show, style, zone_key

page_header("Weather and AC use",
            "External weather (Open-Meteo) mapped to the 24 AC-metered (C1) zone-days. Regional reanalysis, not "
            "measured at the building, so every link here is a correlation, not a cause.",
            eyebrow="Question A · external drivers")

wx = load_weather_daily()
if wx is None:
    st.warning("No weather data. Re-run `scripts/build_dataset.py` with internet access.")
    st.stop()
zd = load_zone_day()
m1 = zd[(zd["group"] == "C1") & zd["usable"]].copy()

FEATURES = {
    "wx_work_temp_mean": ("Mean temperature 08–18h", "°C"),
    "wx_temp_mean": ("Daily mean temperature", "°C"),
    "wx_temp_max": ("Daily max temperature", "°C"),
    "wx_cdh": ("Cooling degree-hours (base 24 °C)", "°C·h"),
    "wx_work_cdh": ("Cooling degree-hours 08–18h", "°C·h"),
    "wx_rh_mean": ("Mean relative humidity", "%"),
    "wx_dew_point_mean": ("Mean dew point", "°C"),
    "wx_radiation_kwh_m2": ("Solar radiation", "kWh/m²"),
}

# ---------------------------------------------------------------- shared filters
f1, f2, _ = st.columns([3, 3, 6], gap="small")
day_types = f1.segmented_control(
    "Day type", ["Workday", "Weekend/holiday"], selection_mode="multi",
    default=["Workday", "Weekend/holiday"],
) or ["Workday", "Weekend/holiday"]
years = f2.segmented_control("Year", [2018, 2019], selection_mode="multi", default=[2019]) or [2019]

selected_dates = pd.to_datetime(
    m1.loc[m1["day_type"].isin(day_types) & m1["year"].isin(years), "date"]
).dt.normalize().drop_duplicates()
weather_dates = pd.to_datetime(wx["date"]).dt.normalize()
w = wx[weather_dates.isin(selected_dates)].copy()
if w.empty:
    st.warning("No weather data for this filter.")
    st.stop()

# ---------------------------------------------------------------- weather timeline
hot = w.loc[w["temp_max"].idxmax()]
with card(f"Outdoor temperature, {w['date'].min():%d %b %Y} – {w['date'].max():%d %b %Y}",
          f"The hot season peaks in April–May; the warmest day reached {hot['temp_max']:.1f} °C on "
          f"{hot['date']:%d %b %Y}."):
    fig = go.Figure()
    fig.add_scatter(x=w["date"], y=w["temp_max"], name="Daily max", mode="lines",
                    line=dict(width=1.2, color="#eb6834"), hovertemplate="Max %{y:.1f} °C<extra></extra>")
    fig.add_scatter(x=w["date"], y=w["temp_mean"], name="Daily mean", mode="lines",
                    line=dict(width=2, color="#2a78d6"), hovertemplate="Mean %{y:.1f} °C<extra></extra>")
    fig.update_layout(yaxis_title="°C", hovermode="x unified")
    fig.update_xaxes(hoverformat="%a %d %b %Y")
    show(style(fig, height=260))

# ---------------------------------------------------------------- AC vs weather
f1, f2, _ = st.columns([3, 3, 6], gap="small")
feat = f1.selectbox("Weather feature", list(FEATURES), format_func=lambda f: FEATURES[f][0])
s = m1[m1["day_type"].isin(day_types) & m1["year"].isin(years)].dropna(subset=[feat, "ac_kwh"])
zones = sorted(s["zone_id"].unique(), key=zone_key)
zone = f2.selectbox("Zone", ["All C1 zones (mean per zone)"] + zones)
label, unit = FEATURES[feat]

if zone.startswith("All"):
    pts = s.groupby("date").agg(ac_kwh=("ac_kwh", "mean"), x=(feat, "first"), season=("season", "first")).reset_index()
else:
    pts = s[s["zone_id"] == zone].rename(columns={feat: "x"})[["date", "ac_kwh", "x", "season"]]
if len(pts) < 10:
    st.warning("Too few usable days for this filter.")
    st.stop()
slope, icpt = np.polyfit(pts["x"], pts["ac_kwh"], 1)
r = np.corrcoef(pts["x"], pts["ac_kwh"])[0, 1]
strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"

with card("AC energy per zone-day against weather",
          f"Each extra {unit} goes with <b>{slope:+.1f} kWh</b> of AC per zone-day, a {strength} "
          f"relationship (r = {r:.2f}, {len(pts):,} days).",
          "Simple what-if from a linear fit: a sanity check for the XGBoost expected-consumption model, not a "
          "causal estimate (no occupancy or set-point data)."):
    fig = px.scatter(pts, x="x", y="ac_kwh", color="season", color_discrete_map=SEASON_COLORS,
                     category_orders={"season": ["Hot", "Rainy", "Cool"]},
                     labels={"x": f"{label} ({unit})", "ac_kwh": "AC kWh per zone-day", "season": ""},
                     custom_data=["date"])
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.9)")),
                      hovertemplate="%{customdata[0]|%a %d %b %Y}<br>%{x:.1f} " + unit
                                    + " → %{y:,.0f} kWh<extra></extra>")
    xs = np.linspace(pts["x"].min(), pts["x"].max(), 50)
    fig.add_scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Linear fit",
                    line=dict(width=2, color=MUTED, dash="dot"), hoverinfo="skip")
    show(style(fig, height=400))

# ---------------------------------------------------------------- per-zone sensitivity
rows = []
for zid, g in s.groupby("zone_id"):
    if len(g) >= 20 and g[feat].std() > 0:
        b, _ = np.polyfit(g[feat], g["ac_kwh"], 1)
        rows.append({"zone_id": zid, "slope": b, "r": np.corrcoef(g[feat], g["ac_kwh"])[0, 1],
                     "mean_ac": g["ac_kwh"].mean(), "days": len(g)})
sens = pd.DataFrame(rows).sort_values("slope", ascending=False)
if len(sens):
    with card("Weather sensitivity by zone",
              f"<b>{sens.iloc[0]['zone_id']}</b> is the most weather-sensitive zone "
              f"({sens.iloc[0]['slope']:+.1f} kWh per {unit}).",
              "Slope relative to the zone's own mean shows which zones respond most to heat, not just which "
              "are largest."):
        sens["rel"] = 100 * sens["slope"] / sens["mean_ac"]
        st.dataframe(sens, hide_index=True, width="stretch", height=320, column_config={
            "zone_id": st.column_config.TextColumn("Zone", width="small"),
            "slope": st.column_config.NumberColumn(f"kWh per {unit}", format="%+.2f"),
            "r": st.column_config.ProgressColumn("Correlation r", format="%.2f", min_value=0, max_value=1),
            "mean_ac": st.column_config.NumberColumn("Mean AC kWh / day", format="%.0f"),
            "rel": st.column_config.NumberColumn(f"% of mean per {unit}", format="%.1f%%"),
            "days": st.column_config.NumberColumn("Days"),
        })
