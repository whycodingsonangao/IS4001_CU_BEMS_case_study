"""Registers the Operations Twin SVG component once (imported by app/views/twin.py).

Renderer choice: an inline `st.components.v2` component drawing plain SVG. It gives two-way zone
selection with Streamlit (click -> trigger value -> rerun; Python -> `data.selected` -> animated focus),
runs in the app page without an iframe, and adds no dependency or build step.
"""

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent / "components" / "twin"
BUILDING = "__building__"  # trigger value sent by the Back button

twin_view = st.components.v2.component(
    "cubems_operations_twin",
    html=(_DIR / "twin.html").read_text(encoding="utf-8"),
    css=(_DIR / "twin.css").read_text(encoding="utf-8"),
    js=(_DIR / "twin.js").read_text(encoding="utf-8"),
)
