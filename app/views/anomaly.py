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
    page_header,
    run_anomaly,
    show,
    style,
    zone_key,
)

from cubems import config
from cubems.anomaly import kpis

page_header("Anomaly explorer",
            "Which zones show persistent, unexpected changes in operation relative to their usual pattern? "
            "These are candidates for a person to check, not confirmed faults.",
            eyebrow="Question B · zone exceptions")

with st.expander("Detection settings", icon=":material/tune:"):
    s1, s2, s3, s4 = st.columns(4)
    use = s1.segmented_control("Channel", ["ac", "light", "plug"], default="ac", format_func=str.upper) or "ac"
    base = s2.date_input("Baseline window", (pd.Timestamp(config.QB_BASELINE[0]), pd.Timestamp(config.QB_BASELINE[1])))
    test = s3.date_input("Replay window", (pd.Timestamp(config.QB_HOLDOUT[0]), pd.Timestamp(config.QB_HOLDOUT[1])))
    q_lo, q_hi = s4.slider("Band percentiles", 0, 100, (5, 95))
    s5, s6, s7, _ = st.columns(4)
    margin = s5.number_input("Margin outside band (kW)", 0.0, 10.0, 0.25, 0.05)
    min_h = s6.slider("Min consecutive hours", 1, 8, 2)
    min_days = s7.slider("Min baseline days per hour cell", 3, 30, 10)
    st.caption("Lock these before scoring the holdout. Tuning them on Sep–Dec would leak test information.")
if len(base) != 2 or len(test) != 2:
    st.stop()

scored, ep = run_anomaly(use, str(base[0]), str(base[1]), str(test[0]), str(test[1]), q_lo, q_hi, margin, min_h,
                         min_days)
k = kpis(scored, ep)
n_hours = max(len(scored), 1)

c = st.columns(4)
c[0].metric("Candidate episodes", f"{len(ep):,}", border=True)
c[1].metric("Zones with episodes", f"{ep['zone_id'].nunique()} of {k['zone_id'].nunique()}", border=True)
c[2].metric("Event rate per 100 h", f"{100 * len(ep) / n_hours:.2f}", border=True,
            help="Valid episodes / valid monitored hours × 100")
c[3].metric("Duration burden", f"{100 * ep['duration_h'].sum() / n_hours:.1f}%", border=True,
            help="Total deviation hours / valid monitored hours × 100")

top = k.iloc[0] if len(k) else None
c1, c2 = st.columns([5, 7])
with c1, card("Zone triage",
              f"Check <b>{top['zone_id']}</b> first: {top['duration_burden']:.1f}% of its monitored hours are "
              "off-pattern." if top is not None else None,
              "Floor 6 data ends 22 Oct 2019, so its replay window is about 7 weeks. Report it separately."):
    st.dataframe(k[["zone_id", "episodes", "event_rate", "duration_burden"]], hide_index=True,
                 width="stretch", height=420, column_config={
                     "zone_id": st.column_config.TextColumn("Zone", width="small"),
                     "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                     "deviation_hours": st.column_config.NumberColumn("Dev. hours", format="%d"),
                     "event_rate": st.column_config.NumberColumn("Per 100 h", format="%.2f"),
                     "duration_burden": st.column_config.ProgressColumn(
                         "Duration burden", format="%.1f%%", min_value=0,
                         max_value=float(max(k["duration_burden"].max(), 1))),
                 })

with c2, card("Exception list: candidates for management review",
              f"Longest episode: <b>{ep.iloc[0]['zone_id']}</b>, {ep.iloc[0]['duration_h']} h "
              f"{ep.iloc[0]['direction'].lower()} from {ep.iloc[0]['start']:%a %d %b %H:%M}." if len(ep) else None,
              "Ordered by duration, then peak departure. Peak departure is the largest distance outside the band, "
              "not a savings estimate."):
    show_ep = ep.assign(dow=ep["start"].dt.strftime("%a"))
    st.dataframe(show_ep[["zone_id", "start", "duration_h", "direction", "peak_departure_kw"]],
                 hide_index=True, width="stretch", height=372, column_config={
                     "zone_id": st.column_config.TextColumn("Zone", width="small"),
                     "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM, HH:mm"),
                     "duration_h": st.column_config.NumberColumn("Hours", format="%d h"),
                     "direction": st.column_config.TextColumn("Direction"),
                     "peak_departure_kw": st.column_config.NumberColumn("Peak departure", format="%.1f kW"),
                     "min_coverage": st.column_config.ProgressColumn("Data coverage", format="percent",
                                                                     min_value=0, max_value=1),
                 })
    st.download_button("Download exception list (CSV)", ep.to_csv(index=False), "exception_list.csv", "text/csv",
                       icon=":material/download:")

# ---------------------------------------------------------------- band chart
zones = sorted(scored["zone_id"].unique(), key=zone_key)
default = ep["zone_id"].iloc[0] if len(ep) else zones[0]
with card("Inspect a zone against its historical band"):
    f1, f2 = st.columns([1, 2])
    zone = f1.selectbox("Zone", zones, index=zones.index(default))
    zs = scored[scored["zone_id"] == zone]
    zep = ep[ep["zone_id"] == zone]
    days = sorted(zs["date"].dt.date.unique())
    pick = f2.date_input("Start date (shows 3 days)", zep["start"].min().date() if len(zep) else days[0],
                         min_value=days[0], max_value=days[-1])
    t0 = pd.Timestamp(pick)
    grid = pd.DataFrame({"timestamp": pd.date_range(t0, t0 + pd.Timedelta(hours=71), freq="h")})
    win = grid.merge(zs, on="timestamp", how="left")  # missing hours stay NaN -> visible breaks

    fig = go.Figure()
    # band as closed polygons per contiguous run (robust to gaps, unlike fill='tonexty')
    run = win["lo"].isna().cumsum()
    first = True
    for _, seg in win.dropna(subset=["lo", "hi"]).groupby(run):
        xs = pd.concat([seg["timestamp"], seg["timestamp"][::-1]])
        ys = pd.concat([seg["hi"], seg["lo"][::-1]])
        fig.add_scatter(x=xs, y=ys, fill="toself", fillcolor=BAND_FILL, line=dict(width=0), mode="lines",
                        name=f"Usual range (P{q_lo}–P{q_hi})", legendgroup="band", showlegend=first, hoverinfo="skip")
        first = False
    fig.add_scatter(x=win["timestamp"], y=win["median"], mode="lines", name="Usual (median)",
                    line=dict(width=1.2, color=MUTED, dash="dot"), hovertemplate="Usual %{y:.1f} kW<extra></extra>")
    fig.add_scatter(x=win["timestamp"], y=win["kw"], mode="lines", name="Observed", connectgaps=False,
                    line=dict(width=2, color=EMPHASIS), hovertemplate="Observed %{y:.1f} kW<extra></extra>")
    fl = win[win["flag"].fillna(False).astype(bool)]
    fig.add_scatter(x=fl["timestamp"], y=fl["kw"], mode="markers", name="Outside range",
                    marker=dict(size=9, color=FLAG_COLOR, symbol="x-thin", line=dict(width=2.5, color=FLAG_COLOR)),
                    customdata=np.abs(fl["departure"]), hovertemplate="Outside by %{customdata:.1f} kW<extra></extra>")
    n_in = 0
    for _, e in zep.iterrows():
        if e["end"] > grid["timestamp"].min() and e["start"] <= grid["timestamp"].max():
            fig.add_vrect(x0=e["start"], x1=e["end"], fillcolor=EPISODE_FILL, line_width=0, layer="below")
            n_in += 1
    fig.update_layout(yaxis_title=f"{use.upper()} power (kW)", hovermode="x unified")
    fig.update_xaxes(hoverformat="%a %d %b, %H:00", tickformat="%a %d %b\n%H:00")
    show(style(fig, height=400))
    st.caption(f"{n_in} sustained episode(s) shaded in this window. The range compares the same zone, hour and day "
               "type (workday vs weekend/holiday). The weather-aware expected-consumption model will replace it to cut "
               "false alarms on hot days.")
