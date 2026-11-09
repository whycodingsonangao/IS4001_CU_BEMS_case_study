"""CU-BEMS dashboard entry point:  streamlit run app/Home.py"""

import streamlit as st
from common import apply_style

st.set_page_config(page_title="Where to Look First · CU-BEMS", page_icon=":material/bolt:", layout="wide")
apply_style()

pages = {
    "": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "A · Where energy goes": [
        st.Page("views/composition.py", title="Energy composition", icon=":material/stacked_bar_chart:"),
        st.Page("views/patterns.py", title="Daily patterns (k-means)", icon=":material/ssid_chart:"),
        st.Page("views/weather.py", title="Weather and AC", icon=":material/thermostat:"),
    ],
    "B · Next-day AC priorities": [
        st.Page("views/forecast.py", title="Next-day priorities", icon=":material/query_stats:"),
        st.Page("views/exceptions.py", title="Daily deviations", icon=":material/crisis_alert:"),
    ],
    "Data": [
        st.Page("views/quality.py", title="Data quality", icon=":material/fact_check:"),
        st.Page("views/sensitivity.py", title="Sensitivity (60/60)", icon=":material/compare_arrows:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("IS4001 Group 7 · CU-BEMS 2019  \nWeather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0)")
nav.run()
