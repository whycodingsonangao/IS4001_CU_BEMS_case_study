import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    DEEMPH,
    FLAG_COLOR,
    MONTHS,
    SEQ_BLUE,
    card,
    heat_text,
    load_hourly,
    load_zone_day,
    load_zones,
    page_header,
    show,
    style,
    zone_key,
)

from cubems import config
from cubems.calendar import holiday_table

page_header("Data quality register",
            f"An hour is valid with ≥{config.MIN_HOUR_MINUTES}/60 valid minutes (a minute is valid only if every circuit "
            "of that end use has a reading). A zone-day is complete when all 24 hours are valid for every end use of "
            "its group. Gaps are dropped, never filled with zeros.", eyebrow="Data")

zd = load_zone_day()
y19 = zd[zd["year"] == 2019]
groups = st.segmented_control("Zone groups", ["C1", "C2", "Lighting only"], selection_mode="multi",
                              default=["C1", "C2", "Lighting only"]) or ["C1", "C2", "Lighting only"]
y19 = y19[y19["group"].isin(groups)]
zdg = zd[zd["group"].isin(groups)]


def windows(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "zones": d["zone_id"].nunique(),
        "complete": 100 * d["usable"].mean(),
        "strict": 100 * d["usable_strict"].mean(),
        "dev": 100 * d.loc[d["month"] <= 8, "usable"].mean(),
        "final": 100 * d.loc[d["month"] >= 9, "usable"].mean(),
    })


tbl = y19.groupby(["group", "floor"]).apply(windows, include_groups=False).reset_index()
tbl["floor"] = "Floor " + tbl["floor"].astype(str)
tot = y19.groupby("group").apply(windows, include_groups=False).reset_index().assign(floor="All floors")
tbl = pd.concat([tbl, tot]).sort_values(["group", "floor"], key=lambda s: s.str.replace("All", "~"))
pct = dict(format="%.1f%%", min_value=0, max_value=100)
c1n = y19[y19["group"] == "C1"]
with card("Complete zone-days, 2019",
          f"C1: <b>{int(c1n['usable'].sum()):,}</b> of {len(c1n):,} zone-days complete under 57/60 vs "
          f"{int(c1n['usable_strict'].sum()):,} under 60/60. Floor 6 data stops 22 Oct 13:27 (19.3% of the year absent)."
          if len(c1n) else None):
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "group": "Group", "floor": "Floor", "zones": st.column_config.NumberColumn("Zones", format="%d"),
        "complete": st.column_config.ProgressColumn("Jan–Dec (57/60)", **pct),
        "strict": st.column_config.ProgressColumn("Jan–Dec (60/60)", **pct),
        "dev": st.column_config.ProgressColumn("Jan–Aug", **pct),
        "final": st.column_config.ProgressColumn("Sep–Dec", **pct)})

c1, _ = st.columns([1, 4])
yr = c1.segmented_control("Year", [2019, 2018], default=2019) or 2019
d = zdg[zdg["year"] == yr]
piv = 100 - d.pivot_table(index="zone_id", columns="month", values="usable", aggfunc="mean") * 100
piv = piv.loc[sorted(piv.index, key=zone_key)]
piv.columns = [MONTHS[m - 1] for m in piv.columns]
worst = piv.stack().idxmax()
with card("Days lost per zone and month",
          f"Blank cells lost nothing. The worst cell is <b>{worst[0]} in {worst[1]}</b> ({piv.stack().max():.0f}% of days "
          "incomplete)."):
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=SEQ_BLUE, zmin=0, zmax=100,
                               xgap=2, ygap=2, text=heat_text(piv), texttemplate="%{text}",
                               colorbar=dict(title="% incomplete", thickness=12),
                               hovertemplate="%{y} · %{x}<br>%{z:.0f}% of days incomplete<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, height=max(360, 20 * len(piv)), legend=False))

zones = sorted(zdg["zone_id"].unique(), key=zone_key)
with card("Valid hours per day for one zone"):
    zone = st.selectbox("Zone", zones, index=zones.index("F6Z1") if "F6Z1" in zones else 0)
    z = zdg[zdg["zone_id"] == zone].sort_values("date")
    uses = config.GROUP_USES[z["group"].iloc[0]]
    z = z.assign(min_hours=z[[f"{u.lower()}_hours" for u in uses]].min(axis=1))
    st.markdown(f'<div class="takeaway">{zone} has <b>{int((~z["usable"]).sum())}</b> incomplete day(s) out of '
                f'{len(z)}.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for ok, name, colr in [(True, "Complete", DEEMPH), (False, "Incomplete (dropped)", FLAG_COLOR)]:
        s = z[z["usable"] == ok]
        fig.add_bar(x=s["date"], y=s["min_hours"], name=name, marker_color=colr, marker_line_width=0,
                    hovertemplate="%{x|%a %d %b %Y}<br>%{y} of 24 hours valid (worst end use)<extra></extra>")
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(range=[0, 25], title="Valid hours (worst end use)")
    show(style(fig, height=240))


@st.cache_data
def holiday_check() -> pd.DataFrame:
    """Observed AC on each listed date vs the zone's normal workday (median zone ratio, 09-17h)."""
    h = load_hourly()
    h = h[(h["timestamp"].dt.year == 2019) & (h["ac_minutes"] >= config.MIN_HOUR_MINUTES)]
    h = h[h["timestamp"].dt.hour.between(9, 16)].assign(date=h["timestamp"].dt.normalize())
    zday = h.groupby(["floor", "zone", "date"]).agg(kwh=("ac_kwh", "sum"), n=("ac_kwh", "size")).reset_index()
    zday = zday[zday["n"] == 8]
    wk = zday[zday["date"].dt.dayofweek < 5].groupby(["floor", "zone"])["kwh"].median().rename("med")
    zday = zday.join(wk, on=["floor", "zone"])
    zday = zday[zday["med"] > 5]
    r = (zday["kwh"] / zday["med"]).groupby(zday["date"]).median().rename("load_vs_normal_workday")
    t = holiday_table().merge(r, left_on="date", right_index=True, how="left")
    return t


with card("Official 2019 holiday list (cited) checked against the building",
          "Used as model flags only where <b>use as holiday</b> is ticked. Labour Day and the ASEAN Summit days are "
          "listed but not flagged: the building ran normally.",
          "Load = median over zones of 09–17h AC vs that zone's normal workday. Blank = no complete data that day "
          "(e.g. 16 Sep, a possible institution closure: open item)."):
    st.dataframe(holiday_check(), hide_index=True, width="stretch", height=320, column_config={
        "date": st.column_config.DateColumn("Date", format="ddd D MMM"), "name": "Holiday", "category": "Category",
        "use_as_holiday": st.column_config.CheckboxColumn("Use as holiday"),
        "load_vs_normal_workday": st.column_config.ProgressColumn("Building AC load", format="percent", min_value=0,
                                                                   max_value=1.4),
        "source": st.column_config.TextColumn("Source", width="large")})

c1, c2 = st.columns(2)
with c1, card("Extreme readings", "Flagged, kept, never deleted automatically.",
              f"Any power channel above {config.EXTREME_KW:.0f} kW in a minute."):
    ex = zdg.loc[zdg["extreme_minutes"] > 0, ["zone_id", "date", "extreme_minutes"]]
    st.dataframe(ex, hide_index=True, width="stretch", column_config={
        "zone_id": "Zone", "date": st.column_config.DateColumn("Date"),
        "extreme_minutes": st.column_config.NumberColumn("Minutes flagged")})
with c2, card("Indoor sensors, 2019", "Not model inputs: 0% coverage in Jan–Feb on every floor; no columns on Floor 1."):
    env = (y19[y19["temp_minutes"].notna()].groupby("zone_id")["env_coverage"]
           .apply(lambda s: 100 * (s >= 0.95).mean()).rename("pct").reset_index())
    st.dataframe(env, hide_index=True, width="stretch", height=260, column_config={
        "zone_id": "Zone", "pct": st.column_config.ProgressColumn("Days with ≥95% sensor minutes", **pct)})

with st.expander("Zone registry"):
    st.dataframe(load_zones(), hide_index=True, width="stretch", column_config={
        "zone_id": "Zone", "group": "Group", "n_ac_units": "AC units", "has_ac": "Has AC",
        "has_env": "Indoor sensors", "channels": "Power channels"})
