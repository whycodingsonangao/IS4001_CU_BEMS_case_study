"""Daily AC deviations for retrospective facilities review."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    BAND_FILL,
    EMPHASIS,
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

page_header("Daily AC deviations",
            "Review complete zone-days whose measured AC total falls outside the expected daily range. "
            "These are candidates for follow-up, not confirmed equipment faults; the result is known after the day ends.",
            eyebrow="Question B · review")

dec = need(load_qb_decision(), "run_question_b.py")
scored = need(load_qb("scored"), "run_question_b.py")
exceptions = need(load_qb("exceptions"), "run_question_b.py")
k = need(load_qb("kpis"), "run_question_b.py")

c = st.columns(3)
c[0].metric("Exception zone-days", f"{len(exceptions):,}", border=True)
c[1].metric("Zones with deviations", f"{exceptions.zone_id.nunique()} of {k.zone_id.nunique()}", border=True)
c[2].metric("Monitored zone-days", f"{len(scored):,}", border=True)

top = k.iloc[0]
with card("Zones for retrospective review",
          f"<b>{top.zone_id}</b> has the highest share of out-of-range days: {top.exception_rate:.1f}%.",
          "Rates use only complete days with an available expected range. Floor 6 ends in October."):
    st.dataframe(k[["zone_id", "exception_days", "monitored_days", "exception_rate", "total_departure_kwh"]],
                 hide_index=True, width="stretch", height=360, column_config={
                     "zone_id": "Zone", "exception_days": "Out-of-range days", "monitored_days": "Monitored days",
                     "exception_rate": st.column_config.NumberColumn("Rate", format="%.1f%%"),
                     "total_departure_kwh": st.column_config.NumberColumn("Total outside range · kWh", format="%.1f")})

busy = exceptions.groupby("date").zone_id.nunique().sort_values(ascending=False)
with card("Exception list", "Check building-wide dates against the calendar before attributing them to equipment.",
          "The range uses each zone and day type's 5th–95th percentile development residuals; it is fixed before "
          "scoring the final test."):
    st.dataframe(exceptions[["zone_id", "date", "direction", "y", "pred", "lo", "hi", "departure_kwh"]],
                 hide_index=True, width="stretch", height=400, column_config={
                     "zone_id": "Zone", "date": st.column_config.DateColumn("Date"), "direction": "Direction",
                     "y": st.column_config.NumberColumn("Actual kWh", format="%.1f"),
                     "pred": st.column_config.NumberColumn("Forecast kWh", format="%.1f"),
                     "lo": st.column_config.NumberColumn("Range low", format="%.1f"),
                     "hi": st.column_config.NumberColumn("Range high", format="%.1f"),
                     "departure_kwh": st.column_config.NumberColumn("Outside range · kWh", format="%+.1f")})
    st.download_button("Download daily exceptions (CSV)", exceptions.to_csv(index=False),
                       "daily_ac_exceptions.csv", "text/csv", icon=":material/download:")
    if len(busy):
        st.caption("Most widespread dates: " + ", ".join(f"{day:%d %b} ({count} zones)" for day, count in busy.head(4).items()))

zones = sorted(scored.zone_id.unique(), key=zone_key)
with card("Inspect a zone's daily range"):
    zone = st.selectbox("Zone", zones, index=zones.index(top.zone_id))
    q = scored[scored.zone_id == zone].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=pd.concat([q.date, q.date[::-1]]), y=pd.concat([q.hi, q.lo[::-1]]),
                    fill="toself", fillcolor=BAND_FILL, line=dict(width=0), name="Expected range", hoverinfo="skip")
    fig.add_scatter(x=q.date, y=q.pred, name="Forecast", line=dict(color=MUTED, dash="dot"))
    fig.add_scatter(x=q.date, y=q.y, name="Actual", line=dict(color=EMPHASIS))
    flagged = q[q.flag]
    fig.add_scatter(x=flagged.date, y=flagged.y, name="Outside range", mode="markers",
                    marker=dict(size=9, color=FLAG_COLOR))
    fig.update_layout(yaxis_title="AC kWh per complete day", hovermode="x unified")
    show(style(fig, height=380))
