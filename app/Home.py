"""CU-BEMS dashboard entry point:  streamlit run app/Home.py"""

import streamlit as st
from common import apply_style

st.set_page_config(page_title="Where to Look First · CU-BEMS", page_icon=":material/bolt:",
                   layout="wide")
apply_style()

pages = {
    "": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "A · Where energy goes": [
        st.Page("views/composition.py", title="Energy composition", icon=":material/stacked_bar_chart:"),
        st.Page("views/weather.py", title="Weather and AC", icon=":material/thermostat:"),
        st.Page("views/profiles.py", title="Daily profiles", icon=":material/ssid_chart:"),
    ],
    "B · Zone exceptions": [
        st.Page("views/anomaly.py", title="Anomaly explorer", icon=":material/crisis_alert:"),
        st.Page("views/folds.py", title="Seasonal folds", icon=":material/calendar_month:"),
    ],
    "Data": [
        st.Page("views/quality.py", title="Data quality", icon=":material/fact_check:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("IS4001 Group 7 · CU-BEMS 2018–19  \nWeather data by [Open-Meteo.com](https://open-meteo.com/) "
               "(CC BY 4.0)")
nav.run()
