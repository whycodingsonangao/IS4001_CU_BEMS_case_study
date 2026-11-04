import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import MONTHS, SERIES, card, load_hourly, load_zone_day, page_header, show, style
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cubems import config

page_header("Daily profile clustering",
            "K-means on 24-hour profiles normalised by the day's total, so clusters capture when energy is "
            "used, not how much. Clustering is date-free; date, floor and weather are attached afterwards.",
            eyebrow="Question A · recurring patterns")

zd = load_zone_day()
h = load_hourly()

f1, f3, f4, f5 = st.columns([2, 3, 2, 1])
group = f1.segmented_control("Model", ["Model 1", "Model 2"], default="Model 1",
                             help="Model 1: AC + lighting + plug zones. Model 2: lighting + plug only.") or "Model 1"
zgroup = zd[zd["group"] == group]
all_f = sorted(zgroup["floor"].unique())
floors = st.pills("Floors", all_f, selection_mode="multi", default=all_f,
                 format_func=lambda f: f"Floor {f}") or []
window = f3.segmented_control("Window", ["Baseline Jan–Aug", "All 2019"], default="Baseline Jan–Aug") \
    or "Baseline Jan–Aug"
k = f4.slider("Clusters (k)", 2, 6, 3)
seed = f5.number_input("Seed", value=42, step=1)
start, end = ("2019-01-01", "2019-08-31") if window.startswith("Baseline") else ("2019-01-01", "2019-12-31")


@st.cache_data(show_spinner="Building 24-h profiles...")
def profiles(group: str, floors: tuple[int, ...], start: str, end: str) -> pd.DataFrame:
    days = zd[(zd["group"] == group) & zd["usable"] & zd["floor"].isin(floors)
              & zd["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    uses = [c for c in ("ac_kwh", "light_kwh", "plug_kwh") if group == "Model 1" or c != "ac_kwh"]
    x = h[h["floor"].isin(floors)].copy()
    x["date"] = x["timestamp"].dt.normalize()
    x = x.merge(days[["floor", "zone", "date"]], on=["floor", "zone", "date"])
    x = x[x["joint_minutes"] >= config.MIN_HOUR_MINUTES]
    x["kwh"] = x[uses].sum(axis=1, min_count=1)
    x["hour"] = x["timestamp"].dt.hour
    p = x.pivot_table(index=["floor", "zone", "date"], columns="hour", values="kwh").dropna()
    tot = p.sum(axis=1)
    return p[tot > 0].div(tot[tot > 0], axis=0) * 100


p = profiles(group, tuple(floors), start, end)
if len(p) < k * 5:
    st.warning("Not enough complete zone-days for this selection.")
    st.stop()

km = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit(p.values)
order = np.argsort(-np.bincount(km.labels_))  # Pattern 1 = largest cluster
rename = {old: new for new, old in enumerate(order)}
lab = np.array([rename[i] for i in km.labels_])
centres = km.cluster_centers_[order]
sil = silhouette_score(p.values, lab, sample_size=min(3000, len(p)), random_state=0)
names = [f"Pattern {i + 1}" for i in range(k)]

meta = p.index.to_frame(index=False).assign(pattern=[names[i] for i in lab])
meta = meta.merge(zd[["floor", "zone", "date", "day_type", "month", "season", "zone_id"]], on=["floor", "zone", "date"])


def describe(c: np.ndarray) -> str:
    work = c[8:18].sum()
    if work < 55:  # an even 24-h spread puts 42% in 08-18h
        return "Flat / standby (no working-hours peak)"
    if c[12] < 0.6 * c[[10, 14]].mean():
        return "Office hours with lunch-break dip"
    if c[18:].sum() > 15:
        return "Office hours, running late"
    return "Office hours, steady"


summary = pd.DataFrame({
    "Pattern": names,
    "Shape": [describe(c) for c in centres],
    "Zone-days": [(lab == i).sum() for i in range(k)],
    "Peak hour": [f"{int(np.argmax(c)):02d}:00" for c in centres],
    "Share 08–18h": [c[8:18].sum() for c in centres],
    "Workday share": [100 * (meta.loc[meta["pattern"] == n, "day_type"] == "Workday").mean() for n in names],
})

c = st.columns(3)
c[0].metric("Zone-days clustered", f"{len(p):,}", border=True)
c[1].metric("Silhouette", f"{sil:.2f}", border=True, help="0.25–0.5 reasonable, >0.5 strong separation")
c[2].metric("Largest pattern", f"{summary.iloc[0]['Zone-days'] / len(p):.0%}", border=True,
            help=summary.iloc[0]["Shape"])

with card("Typical daily shapes", f"Pattern 1 is <b>{summary.iloc[0]['Shape'].lower()}</b>, covering "
          f"{summary.iloc[0]['Zone-days']:,} zone-days."):
    fig = go.Figure()
    for i in range(k):
        fig.add_scatter(x=list(range(24)), y=centres[i], mode="lines+markers",
                        name=f"{names[i]} · {summary.iloc[i]['Shape']}",
                        line=dict(width=2, color=SERIES[i]), marker=dict(size=5),
                        hovertemplate=f"{names[i]}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Share of daily electricity (%)", hovermode="x unified")
    fig.update_xaxes(tickvals=list(range(0, 24, 3)), ticktext=[f"{i:02d}:00" for i in range(0, 24, 3)])
    show(style(fig, height=360))
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Share 08–18h": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Workday share": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    })

c1, c2 = st.columns(2)
with c1, card("When each pattern occurs", None, "Zone-days per month, stacked by pattern."):
    fig = px.histogram(meta.assign(m=meta["month"].map(lambda m: MONTHS[m - 1])), x="m", color="pattern",
                       barmode="stack", color_discrete_sequence=SERIES, labels={"m": "", "pattern": ""},
                       category_orders={"pattern": names, "m": MONTHS})
    fig.update_traces(marker_line=dict(width=1.5, color="rgba(255,255,255,0.9)"),
                      hovertemplate="%{x}: %{y} zone-days<extra></extra>")
    fig.update_layout(yaxis_title="Zone-days")
    show(style(fig, height=300))
with c2, card("Which zones follow which pattern", None, "Share of each zone's clustered days."):
    ct = pd.crosstab(meta["zone_id"], meta["pattern"], normalize="index") * 100
    st.dataframe(ct.reset_index(), hide_index=True, width="stretch", height=300, column_config={
        "zone_id": st.column_config.TextColumn("Zone", width="small"),
        **{n: st.column_config.ProgressColumn(n, format="%.0f%%", min_value=0, max_value=100) for n in names},
    })

st.caption("Choose k with the silhouette and how interpretable the shapes are, and check stability by changing "
           "the seed. The proposal expects about two to three patterns per model.")
