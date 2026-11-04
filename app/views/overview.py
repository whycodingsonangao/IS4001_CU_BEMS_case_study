import streamlit as st
from common import load_hourly, load_qa, load_qb, load_qb_decision, load_zone_day, page_header

from cubems.aggregate import valid_kwh


@st.cache_data(show_spinner="Computing headline findings...")
def headlines() -> dict:
    zd = load_zone_day()
    d = zd[(zd["year"] == 2019) & zd["usable"]]
    out = {"mwh": d["total_kwh"].sum() / 1e3, "zone_days": int(d["usable"].sum()),
           "usable_share": zd.loc[zd["year"] == 2019, "usable"].mean(),
           "ac_share": d["ac_kwh"].sum() / d["total_kwh"].sum()}
    z = d.groupby("zone_id")["total_kwh"].sum().sort_values(ascending=False)
    cum = z.cumsum() / z.sum()
    out["n80"], out["n_zones"] = int((cum < 0.8).sum() + 1), len(z)
    out["top_zone"], out["top_share"] = z.index[0], z.iloc[0] / z.sum()

    h = load_hourly()
    hz = h[h["timestamp"].dt.year == 2019].copy()
    hz["date"] = hz["timestamp"].dt.normalize()
    hz = hz.merge(d[["floor", "zone", "date", "day_type"]], on=["floor", "zone", "date"])
    kwh = sum(valid_kwh(hz, u).fillna(0) for u in ("ac", "light", "plug"))
    hr = hz["timestamp"].dt.hour
    after = (hz["day_type"] != "Workday") | (hr < 8) | (hr >= 18)
    out["after_share"] = float(kwh[after].sum() / kwh.sum())

    lab = load_qa("labels")
    if lab is not None:
        c1 = lab[lab["group"] == "C1"]
        out["c1_k"] = int(c1["cluster"].max())
        share = c1["cluster"].value_counts(normalize=True)
        out["c1_top_share"] = float(share.get(1, 0))
        solo = c1.groupby("cluster")["zone_id"].nunique()
        out["solo_zone"] = ", ".join(c1[c1["cluster"].isin(solo[solo == 1].index)]["zone_id"].unique())

    dec = load_qb_decision()
    if dec is not None:
        out.update({k: dec[k] for k in ("chosen", "final_mae", "final_skill_vs_benchmark", "episodes")})
        k = load_qb("kpis")
        out["worst_zone"], out["worst_burden"] = k.iloc[0]["zone_id"], float(k.iloc[0]["duration_burden"])
    return out


page_header("Where to Look First",
            "Show facilities managers where the building's electricity goes, how well each zone's AC use can be "
            "predicted a day ahead, and which zones then behave unexpectedly.",
            eyebrow="CU-BEMS case study · Group 7")

H = headlines()
c = st.columns(4)
c[0].metric("Recorded electricity, 2019", f"{H['mwh']:,.0f} MWh", border=True, help="Complete zone-days, all 33 zones")
c[1].metric("AC share of electricity", f"{H['ac_share']:.0%}", border=True)
c[2].metric("Complete zone-days, 2019", f"{H['zone_days']:,}", border=True,
            help=f"{H['usable_share']:.1%} of all 2019 zone-days have all 24 hours valid (57/60 rule)")
if "final_mae" in H:
    c[3].metric("Day-ahead AC skill vs benchmark", f"{H['final_skill_vs_benchmark']:+.1%}", border=True,
                help=f"Final test Sep–Dec, MAE {H['final_mae']:.2f} kWh per zone-hour ({H['chosen']})")

st.markdown("#### Headline findings")
a, b, w, x = st.columns(4)
with a.container(border=True, height="stretch"):
    st.badge("Question A", color="blue")
    st.markdown(f"**{H['n80']} of {H['n_zones']} zones** carry 80% of the building's electricity. "
                f"{H['top_zone']} alone is {H['top_share']:.0%}.")
    st.page_link("views/composition.py", label="Energy composition", icon=":material/arrow_forward:")
with b.container(border=True, height="stretch"):
    st.badge("Question A", color="blue")
    if "c1_k" in H:
        st.markdown(f"**{H['c1_k']} recurring patterns** in AC zones; the workday office pattern covers "
                    f"{H['c1_top_share']:.0%} of days." + (f" **{H['solo_zone']}** runs a 24/7 pattern of its own."
                                                         if H.get("solo_zone") else ""))
    st.page_link("views/patterns.py", label="Daily patterns", icon=":material/arrow_forward:")
with w.container(border=True, height="stretch"):
    st.badge("Question B", color="orange")
    if "final_mae" in H:
        st.markdown(f"The day-ahead forecast beats 'same hour last week' by **{H['final_skill_vs_benchmark']:.1%}** "
                    f"on Sep–Dec (MAE {H['final_mae']:.2f} kWh per zone-hour). The margin is modest.")
    st.page_link("views/forecast.py", label="Forecast accuracy", icon=":material/arrow_forward:")
with x.container(border=True, height="stretch"):
    st.badge("Question B", color="orange")
    if "worst_zone" in H:
        st.markdown(f"**{H['episodes']} exception episodes** where AC was unusual for the forecast. "
                    f"Check **{H['worst_zone']}** first ({H['worst_burden']:.1f}% of hours).")
    st.page_link("views/exceptions.py", label="Exceptions", icon=":material/arrow_forward:")

st.markdown("#### The two questions")
qa, qb = st.columns(2)
with qa.container(border=True, height="stretch"):
    st.markdown("**A · Which daily patterns recur, and how much kWh does each account for?** (descriptive)")
    st.markdown("- **Unit:** zone-day · **Method:** k-means on 24 hourly kWh per end use (C1: 72, C2: 48 features)\n"
                "- **Scaling:** each zone × end use ÷ its 95th-percentile hourly kWh (floor 0.1 kWh)\n"
                "- **k (3–8):** cluster size and spread → half-year ARI ≥ 0.8 → silhouette → interpretability\n"
                "- **Output:** kWh per cluster, zone, floor and end use; weather and calendar attached afterwards")
with qb.container(border=True, height="stretch"):
    st.markdown("**B · How well can each zone's hourly AC use be predicted a day ahead?** (predictive)")
    st.markdown("- **Unit:** zone-hour · **Issued:** 00:00 on day D for all 24 hours of D\n"
                "- **Models:** same-hour-last-week benchmark, Poisson gradient boosting (monotone), Ridge\n"
                "- **Evaluation:** chronological expanding window, monthly refit; dev May–Aug, final Sep–Dec\n"
                "- **Exception layer:** hours outside the forecast's locked residual range → event rate, duration burden")
    st.caption("Question A's cluster labels are never inputs to Question B.")

with st.expander("How the professor's feedback is built in"):
    st.markdown(
        "1. **External weather:** Open-Meteo (ERA5) day-level features are Question B inputs and are attached to "
        "Question A clusters as labels (C1 zones only). Correlation, not cause.\n"
        "2. **Forecasting for Question B:** Question B is now a day-ahead prediction task. Validation is chronological, "
        "because a forecaster can't train on later months; a seasonally stratified weekly-block CV runs as a diagnostic.\n"
        "3. **X factor:** the weather-aware model feeds the exception report, so an exception means 'unusual given the "
        "weather and recent history', not just 'high on a hot day'.")
