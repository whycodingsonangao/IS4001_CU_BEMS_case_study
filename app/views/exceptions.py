import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    BAND_FILL,
    EMPHASIS,
    EPISODE_FILL,
    FLAG_COLOR,
    MUTED,
    card,
    load_qb,
    load_qb_decision,
    need,
    page_header,
    show,
    style,
    zone_key,
)

page_header("Exceptions: AC use unusual given the forecast",
            "Sep–Dec hours where actual AC falls outside the forecast's expected range (prediction + development "
            "residual 5th–95th percentiles, locked before the final test). Candidates for a person to check, not faults.",
            eyebrow="Question B · exception layer")

dec = need(load_qb_decision(), "run_question_b.py")
scored, ep, k = load_qb("scored"), load_qb("episodes"), load_qb("kpis")
n_hours = max(len(scored), 1)

c = st.columns(4)
c[0].metric("Candidate episodes", f"{len(ep):,}", border=True,
            help=f"≥{dec['exception_min_hours']} consecutive hours more than {dec['exception_margin_kwh']} kWh outside the range")
c[1].metric("Zones with episodes", f"{ep['zone_id'].nunique()} of {k['zone_id'].nunique()}", border=True)
c[2].metric("Event rate per 100 h", f"{100 * len(ep) / n_hours:.2f}", border=True,
            help="Valid episodes / valid monitored hours × 100")
c[3].metric("Duration burden", f"{100 * ep['duration_h'].sum() / n_hours:.1f}%", border=True,
            help="Total deviation hours / valid monitored hours × 100")

top = k.iloc[0]
c1, c2 = st.columns([5, 7])
with c1, card("Zone triage", f"Check <b>{top['zone_id']}</b> first: {top['duration_burden']:.1f}% of its monitored "
              "hours are outside the expected range.", "Floor 6 is monitored only until 22 Oct; report it separately."):
    st.dataframe(k[["zone_id", "episodes", "event_rate", "duration_burden"]], hide_index=True, width="stretch",
                 height=420, column_config={
                     "zone_id": st.column_config.TextColumn("Zone", width="small"),
                     "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                     "event_rate": st.column_config.NumberColumn("Per 100 h", format="%.2f"),
                     "duration_burden": st.column_config.ProgressColumn(
                         "Duration burden", format="%.1f%%", min_value=0, max_value=float(max(k["duration_burden"].max(), 1)))})

days = ep.assign(day=ep["start"].dt.normalize()).groupby("day")["zone_id"].nunique().sort_values(ascending=False)
busy = days[days >= 4]
with c2, card("Exception list: candidates for management review",
              ("Building-wide days (≥4 zones): " + ", ".join(f"<b>{d:%a %d %b}</b> ({n})" for d, n in busy.head(4).items())
               + ". Check the calendar first: these look like closures or events, not equipment.") if len(busy) else None,
              "Ordered by duration, then peak departure (distance outside the range, not a savings estimate)."):
    st.dataframe(ep[["zone_id", "start", "duration_h", "direction", "peak_departure_kw"]], hide_index=True, width="stretch",
                 height=372, column_config={
                     "zone_id": st.column_config.TextColumn("Zone", width="small"),
                     "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM, HH:mm"),
                     "duration_h": st.column_config.NumberColumn("Hours", format="%d h"),
                     "direction": "Direction",
                     "peak_departure_kw": st.column_config.NumberColumn("Peak departure", format="%.1f kWh")})
    st.download_button("Download exception list (CSV)", ep.to_csv(index=False), "exception_list.csv", "text/csv",
                       icon=":material/download:")

zones = sorted(scored["zone_id"].unique(), key=zone_key)
default = ep["zone_id"].iloc[0] if len(ep) else zones[0]
with card("Inspect a zone against its expected range"):
    f1, f2 = st.columns([1, 2])
    zone = f1.selectbox("Zone", zones, index=zones.index(default))
    zs = scored[scored["zone_id"] == zone]
    zep = ep[ep["zone_id"] == zone]
    first = zep["start"].min().normalize().date() if len(zep) else zs["timestamp"].min().date()
    pick = f2.date_input("Start date (shows 3 days)", first, min_value=zs["timestamp"].min().date(),
                         max_value=zs["timestamp"].max().date())
    t0 = pd.Timestamp(pick)
    grid = pd.DataFrame({"timestamp": pd.date_range(t0, t0 + pd.Timedelta(hours=71), freq="h")})
    win = grid.merge(zs, on="timestamp", how="left")
    fig = go.Figure()
    run = win["lo"].isna().cumsum()
    first_band = True
    for _, seg in win.dropna(subset=["lo", "hi"]).groupby(run):
        xs = pd.concat([seg["timestamp"], seg["timestamp"][::-1]])
        ys = pd.concat([seg["hi"], seg["lo"][::-1]])
        fig.add_scatter(x=xs, y=ys, fill="toself", fillcolor=BAND_FILL, line=dict(width=0), mode="lines",
                        name="Expected range", legendgroup="band", showlegend=first_band, hoverinfo="skip")
        first_band = False
    fig.add_scatter(x=win["timestamp"], y=win["pred"], mode="lines", name="Forecast",
                    line=dict(width=1.2, color=MUTED, dash="dot"), hovertemplate="Forecast %{y:.1f} kWh<extra></extra>")
    fig.add_scatter(x=win["timestamp"], y=win["y"], mode="lines", name="Actual", line=dict(width=2, color=EMPHASIS),
                    hovertemplate="Actual %{y:.1f} kWh<extra></extra>")
    fl = win[win["flag"].fillna(False).astype(bool)]
    fig.add_scatter(x=fl["timestamp"], y=fl["y"], mode="markers", name="Outside range",
                    marker=dict(size=9, color=FLAG_COLOR, symbol="x-thin", line=dict(width=2.5, color=FLAG_COLOR)),
                    customdata=np.abs(fl["departure"]), hovertemplate="Outside by %{customdata:.1f} kWh<extra></extra>")
    for _, e in zep.iterrows():
        if e["end"] > grid["timestamp"].min() and e["start"] <= grid["timestamp"].max():
            fig.add_vrect(x0=e["start"], x1=e["end"], fillcolor=EPISODE_FILL, line_width=0, layer="below")
    fig.update_layout(yaxis_title="AC kWh per hour", hovermode="x unified")
    fig.update_xaxes(hoverformat="%a %d %b, %H:00", tickformat="%a %d %b\n%H:00")
    show(style(fig, height=400))
    st.caption("Because the forecast already uses yesterday, last week, the calendar and the day's weather, a hot day "
               "alone no longer triggers an exception.")
