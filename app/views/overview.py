import numpy as np
import streamlit as st
from common import load_hourly, load_weather_daily, load_zone_day, page_header, run_anomaly


@st.cache_data(show_spinner="Computing headline findings...")
def headlines() -> dict:
    zd = load_zone_day()
    d = zd[(zd["year"] == 2019) & zd["usable"] & (zd["group"] != "Out of scope")]
    out = {"mwh": d["total_kwh"].sum() / 1e3, "usable_share": zd.loc[zd["year"] == 2019, "usable"].mean(),
           "zone_days": int(d["usable"].sum())}
    out["ac_share"] = d["ac_kwh"].sum() / d["total_kwh"].sum()

    z = d.groupby("zone_id")["total_kwh"].sum().sort_values(ascending=False)
    cum = z.cumsum() / z.sum()
    out["n80"], out["n_zones"] = int((cum < 0.8).sum() + 1), len(z)
    out["top_zone"], out["top_share"] = z.index[0], z.iloc[0] / z.sum()

    h = load_hourly()
    hz = h[["floor", "zone", "timestamp", "ac_kwh", "light_kwh", "plug_kwh"]].copy()
    hz["date"] = hz["timestamp"].dt.normalize()
    hz = hz.merge(d[["floor", "zone", "date", "day_type"]], on=["floor", "zone", "date"])
    hr = hz["timestamp"].dt.hour
    after = (hz["day_type"] != "Workday") | (hr < 8) | (hr >= 18)
    kwh = hz[["ac_kwh", "light_kwh", "plug_kwh"]].sum(axis=1)
    out["after_share"] = kwh[after].sum() / kwh.sum()

    m1 = d[(d["group"] == "Model 1") & (d["day_type"] == "Workday")].dropna(subset=["wx_work_temp_mean"])
    if len(m1):
        g = m1.groupby("date").agg(ac=("ac_kwh", "mean"), t=("wx_work_temp_mean", "first"))
        out["slope"], _ = np.polyfit(g["t"], g["ac"], 1)
        out["r"] = np.corrcoef(g["t"], g["ac"])[0, 1]

    scored, ep = run_anomaly()
    out["episodes"], out["ep_zones"] = len(ep), ep["zone_id"].nunique()
    burden = (ep.groupby("zone_id")["duration_h"].sum() / scored.groupby("zone_id").size()).dropna()
    out["worst_zone"], out["worst_burden"] = burden.idxmax(), 100 * burden.max()
    sat = ep[ep["start"].dt.dayofweek == 5]
    out["sat_episodes"] = len(sat)
    return out


page_header("Where to Look First",
            "Show facilities managers where the building's electricity goes and which zones are drifting "
            "from their normal behaviour, so they know where to look first.",
            eyebrow="CU-BEMS case study · Group 7")

H = headlines()
c = st.columns(4)
c[0].metric("Recorded electricity, 2019", f"{H['mwh']:,.0f} MWh", border=True,
            help="31 in-scope zones, usable zone-days only")
c[1].metric("AC share of electricity", f"{H['ac_share']:.0%}", border=True)
c[2].metric("Usable zone-days, 2019", f"{H['zone_days']:,}", border=True,
            help=f"{H['usable_share']:.1%} of all 2019 zone-days pass the 95% coverage rule")
c[3].metric("Candidate AC episodes, Sep–Dec", f"{H['episodes']:,}", border=True,
            help="Preliminary P5–P95 band; expect fewer once the weather-aware model is in")

st.markdown("#### Headline findings")
a, b, w, x = st.columns(4)
with a.container(border=True, height="stretch"):
    st.badge("Question A", color="blue")
    st.markdown(f"**{H['n80']} of {H['n_zones']} zones** carry 80% of the building's electricity. "
                f"{H['top_zone']} alone is {H['top_share']:.0%}.")
    st.page_link("views/composition.py", label="Priority ranking", icon=":material/arrow_forward:")
with b.container(border=True, height="stretch"):
    st.badge("Question A", color="blue")
    st.markdown(f"**{H['after_share']:.0%} of electricity** is used outside 08:00–18:00 on workdays "
                "or on weekends and holidays, the target for schedule flagging.")
    st.page_link("views/composition.py", label="After-hours load", icon=":material/arrow_forward:")
with w.container(border=True, height="stretch"):
    st.badge("Question A · weather", color="blue")
    if "slope" in H:
        st.markdown(f"**+{H['slope']:.1f} kWh per °C**: each extra degree of working-hours outdoor "
                    f"temperature goes with more AC per zone-day (r = {H['r']:.2f}, correlation only).")
    st.page_link("views/weather.py", label="Weather and AC", icon=":material/arrow_forward:")
with x.container(border=True, height="stretch"):
    st.badge("Question B", color="orange")
    st.markdown(f"**{H['worst_zone']}** has the highest duration burden ({H['worst_burden']:.1f}% of "
                f"monitored hours off-pattern). {H['sat_episodes']} episodes start on a Saturday.")
    st.page_link("views/anomaly.py", label="Exception list", icon=":material/arrow_forward:")

st.markdown("#### The two questions")
qa, qb = st.columns(2)
with qa.container(border=True, height="stretch"):
    st.markdown("**A · Where the electricity goes** (descriptive)")
    st.markdown("What drives the building's consumption, and how does its composition vary across floors, "
                "end uses and recurring operating patterns?")
    st.markdown("- **For:** building owner, energy and sustainability manager\n"
                "- **Output:** ranked list of where energy is concentrated, with evidence\n"
                "- **Decision:** where to focus energy-saving effort first\n"
                "- **KPI:** normalised kWh baseline and % energy change under a common coverage mask")
with qb.container(border=True, height="stretch"):
    st.markdown("**B · Which zones behave unusually** (anomaly detection)")
    st.markdown("Which zones show persistent, unexpected changes in AC operation or indoor conditions "
                "relative to their usual operating patterns?")
    st.markdown("- **For:** facilities manager and FM team\n"
                "- **Output:** short ranked list of zones drifting from normal\n"
                "- **Decision:** which zones to check first\n"
                "- **KPIs:** event rate and duration burden per valid monitored hour")

with st.expander("How the professor's feedback is built in"):
    st.markdown(
        "1. **External weather (Q-A):** Open-Meteo hourly reanalysis, summarised to daily mean and max "
        "temperature, humidity, dew point, radiation and cooling degree-hours, joined on date to the 23 "
        "AC-metered zones only. Regional, not measured at the building, so treat it as correlation.\n"
        "2. **Within-season folds (Q-B):** leave one month out within Hot (Mar–May), Rainy (Jun–Oct) and "
        "Cool (Nov–Feb). 2018 Jul–Dec months are available as a toggle pending team agreement.\n"
        "3. **X factor:** the episode detector accepts any expected band, so a weather-aware XGBoost "
        "expected-consumption model can replace the empirical band without other changes.")
wx = load_weather_daily()
if wx is None:
    st.warning("Weather data not built. Re-run `scripts/build_dataset.py` with internet access.")
