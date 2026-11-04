import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    DEEMPH,
    FLAG_COLOR,
    MONTHS,
    MUTED,
    SEQ_BLUE,
    card,
    heat_text,
    load_zone_day,
    load_zones,
    page_header,
    show,
    style,
    zone_key,
)

from cubems import config

page_header("Data quality register",
            f"A zone-day is usable when at least {config.MIN_DAY_COVERAGE:.0%} of its minutes have every power "
            "channel recorded. Gap days are dropped, never filled with zeros.", eyebrow="Data")

zd = load_zone_day()
scope = st.segmented_control("Zone groups", ["Model 1", "Model 2", "Out of scope"], selection_mode="multi",
                             default=["Model 1", "Model 2"]) or ["Model 1", "Model 2"]
zd = zd[zd["group"].isin(scope)]
y19 = zd[zd["year"] == 2019]


def windows(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "zones": d["zone_id"].nunique(),
        "qa": 100 * d["usable"].mean(),
        "base": 100 * d.loc[d["month"] <= 8, "usable"].mean(),
        "hold": 100 * d.loc[d["month"] >= 9, "usable"].mean(),
    })


tbl = y19.groupby(["group", "floor"]).apply(windows, include_groups=False).reset_index()
tbl["floor"] = "Floor " + tbl["floor"].astype(str)
tot = y19.groupby("group").apply(windows, include_groups=False).reset_index().assign(floor="All floors")
tbl = pd.concat([tbl, tot]).sort_values(["group", "floor"], key=lambda s: s.str.replace("All", "~"))
pct = dict(format="%.1f%%", min_value=0, max_value=100)
with card("Usable zone-days by analysis window",
          "Recomputes the proposal's scope table. Floor 6's holdout collapses to 41% because its data ends "
          "22 Oct 2019."):
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "group": st.column_config.TextColumn("Group"),
        "floor": st.column_config.TextColumn("Floor"),
        "zones": st.column_config.NumberColumn("Zones", format="%d"),
        "qa": st.column_config.ProgressColumn("Q-A · Jan–Dec", **pct),
        "base": st.column_config.ProgressColumn("Q-B baseline · Jan–Aug", **pct),
        "hold": st.column_config.ProgressColumn("Q-B holdout · Sep–Dec", **pct),
    })

c1, c2 = st.columns([1, 4])
yr = c1.segmented_control("Year", [2019, 2018], default=2019) or 2019
d = zd[zd["year"] == yr]
piv = 100 - d.pivot_table(index="zone_id", columns="month", values="usable", aggfunc="mean") * 100
piv = piv.loc[sorted(piv.index, key=zone_key)]
piv.columns = [MONTHS[m - 1] for m in piv.columns]
worst = piv.stack().idxmax()
with card("Days lost to gaps, by zone and month",
          f"Blank cells lost nothing. The worst cell is <b>{worst[0]} in {worst[1]}</b> "
          f"({piv.stack().max():.0f}% of days unusable)."):
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=SEQ_BLUE, zmin=0, zmax=100,
                               xgap=2, ygap=2, text=heat_text(piv), texttemplate="%{text}",
                               colorbar=dict(title="% unusable", thickness=12),
                               hovertemplate="%{y} · %{x}<br>%{z:.0f}% of days unusable<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, height=max(360, 20 * len(piv)), legend=False))

zones = sorted(zd["zone_id"].unique(), key=zone_key)
with card("Daily coverage for one zone"):
    zone = st.selectbox("Zone", zones, index=zones.index("F6Z1") if "F6Z1" in zones else 0)
    z = zd[zd["zone_id"] == zone].sort_values("date")
    lost = int((~z["usable"]).sum())
    st.markdown(f'<div class="takeaway">{zone} has <b>{lost}</b> unusable day(s) out of {len(z)} recorded.</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for ok, name, colr in [(True, "Usable", DEEMPH), (False, f"Below {config.MIN_DAY_COVERAGE:.0%} (dropped)",
                                                       FLAG_COLOR)]:
        s = z[z["usable"] == ok]
        fig.add_bar(x=s["date"], y=s["coverage"], name=name, marker_color=colr, marker_line_width=0,
                    hovertemplate="%{x|%a %d %b %Y}<br>Coverage %{y:.1%}<extra></extra>")
    fig.add_hline(y=config.MIN_DAY_COVERAGE, line=dict(dash="dot", color=MUTED, width=1))
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.04], title="Joint-meter coverage")
    show(style(fig, height=240))

c1, c2 = st.columns(2)
with c1, card("Extreme readings", "Flagged, kept, never deleted automatically.",
              f"Any power channel above {config.EXTREME_KW:.0f} kW in a minute. Test analyses with and without "
              "these days."):
    ex = zd.loc[zd["extreme_minutes"] > 0, ["zone_id", "date", "extreme_minutes"]]
    st.dataframe(ex, hide_index=True, width="stretch", column_config={
        "zone_id": "Zone", "date": st.column_config.DateColumn("Date"),
        "extreme_minutes": st.column_config.NumberColumn("Minutes flagged")})
with c2, card("Indoor sensor coverage, 2019", "Context only, never used as clustering input."):
    env = (y19[y19["temp_minutes"].notna()].groupby("zone_id")["env_coverage"]
           .apply(lambda s: 100 * (s >= config.MIN_DAY_COVERAGE).mean()).rename("pct").reset_index())
    st.dataframe(env, hide_index=True, width="stretch", height=260, column_config={
        "zone_id": "Zone", "pct": st.column_config.ProgressColumn("Days with ≥95% sensor data", **pct)})

with st.expander("Zone registry"):
    st.dataframe(load_zones(), hide_index=True, width="stretch", column_config={
        "zone_id": "Zone", "group": "Group", "n_ac_units": "AC units", "has_ac": "Has AC",
        "has_env": "Indoor sensors", "channels": "Power channels"})
