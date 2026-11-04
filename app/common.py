"""Shared data loaders, palette, layout and chart helpers for the dashboard views."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cubems import config

# ---------------------------------------------------------------- palette
# Fixed categorical slots (validated; first three are CVD-safe all-pairs).
# Colour follows the entity: AC is always blue, Lighting orange, Plug aqua.
USE_COLORS = {"AC": "#2a78d6", "Light": "#eb6834", "Plug": "#1baf7a"}
USE_LABELS = {"AC": "AC", "Light": "Lighting", "Plug": "Plug load"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEASON_COLORS = {"Hot": "#eb6834", "Rainy": "#2a78d6", "Cool": "#1baf7a"}
# Sequential blue, light -> dark. The first stop is near-surface so "no problem / near zero" recedes.
SEQ_BLUE = [[0.0, "#f1f6fd"], [0.12, "#cde2fb"], [0.3, "#9ec5f4"], [0.5, "#5598e7"],
            [0.7, "#256abf"], [0.85, "#184f95"], [1.0, "#0d366b"]]
EMPHASIS = "#2a78d6"
DEEMPH = "#b7d3f6"
MUTED = "#898781"
BAND_FILL = "rgba(137,135,129,0.22)"
EPISODE_FILL = "rgba(237,161,0,0.16)"
FLAG_COLOR = "#d03b3b"  # status: critical (always paired with a label)
GRID = "rgba(137,135,129,0.22)"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
h1 {letter-spacing: -0.02em;}
[data-testid="stMetricValue"] {font-weight: 650;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
.eyebrow {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
          color: var(--primary-color, #0f9d6e); margin-bottom: -0.6rem;}
.takeaway {font-size: 1.02rem; margin: -0.35rem 0 0.4rem 0;}
.subtle {opacity: 0.72; font-size: 0.86rem;}
[data-testid="stSidebarNav"] {padding-top: 0.5rem;}
</style>
"""


def apply_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- loaders
def missing_data_stop() -> None:
    st.error("Processed data not found. From the repo root run:\n\n"
             "```\n.venv\\Scripts\\python scripts\\build_dataset.py\n```")
    st.stop()


@st.cache_data(show_spinner="Loading zone-day table...")
def load_zone_day() -> pd.DataFrame:
    if not config.ZONE_DAY_PATH.exists():
        missing_data_stop()
    return pd.read_parquet(config.ZONE_DAY_PATH)


@st.cache_data(show_spinner="Loading hourly table...")
def load_hourly() -> pd.DataFrame:
    if not config.HOURLY_PATH.exists():
        missing_data_stop()
    return pd.read_parquet(config.HOURLY_PATH)


@st.cache_data
def load_zones() -> pd.DataFrame:
    if not config.ZONES_PATH.exists():
        missing_data_stop()
    return pd.read_parquet(config.ZONES_PATH)


@st.cache_data
def load_weather_daily() -> pd.DataFrame | None:
    if not config.WEATHER_DAILY_PATH.exists():
        return None
    return pd.read_parquet(config.WEATHER_DAILY_PATH)


def _results(question: str, strict: bool) -> Path:
    return config.results_dir(question, config.STRICT_HOUR_MINUTES if strict else config.MIN_HOUR_MINUTES)


@st.cache_data(show_spinner="Loading Question A results...")
def load_qa(name: str, strict: bool = False) -> pd.DataFrame | None:
    p = _results("qa", strict) / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner="Loading Question B results...")
def load_qb(name: str, strict: bool = False) -> pd.DataFrame | None:
    p = _results("qb", strict) / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_qb_decision(strict: bool = False) -> dict | None:
    p = _results("qb", strict) / "decision.json"
    return json.loads(p.read_text()) if p.exists() else None


def need(obj, script: str):
    if obj is None:
        cmd = rf".venv\Scripts\python scripts\{script}"
        st.warning(f"Results not built yet. From the repo root run:\n\n```\n{cmd}\n```")
        st.stop()
    return obj


# ---------------------------------------------------------------- helpers
def zone_key(z: str) -> tuple[int, int]:
    return int(z[1]), int(z[3:])


def page_header(title: str, question: str | None = None, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if question:
        st.markdown(f'<p class="subtle">{question}</p>', unsafe_allow_html=True)


@contextmanager
def card(title: str, takeaway: str | None = None, note: str | None = None):
    """Bordered section: short title, a one-line takeaway (the finding), content, footnote."""
    with st.container(border=True):
        st.markdown(f"##### {title}")
        if takeaway:
            html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", takeaway)  # markdown bold inside an HTML block
            st.markdown(f'<div class="takeaway">{html}</div>', unsafe_allow_html=True)
        yield
        if note:
            st.caption(note)


def usable_note() -> str:
    return (f"Complete zone-days only: all 24 hours valid (≥{config.MIN_HOUR_MINUTES}/60 minutes) for every end use of "
            "the zone's group. Gaps are dropped, never filled with zeros. Hourly kWh = mean kW × 1 h.")


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Recessive grid, thin marks, legend on top-left. Text colours come from the Streamlit theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=8 if not legend else 36, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    traceorder="normal"),
        showlegend=legend,
        hovermode="closest",
        bargap=0.3,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, ticks="")
    return fig


def show(fig: go.Figure, **kw) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit",
                    config={"displayModeBar": False, "displaylogo": False}, **kw)


def heat_text(z: pd.DataFrame, hide: float | None = 0) -> np.ndarray:
    """Cell labels, blank where the value is uninformative (e.g. 0% missing)."""
    return np.where(z.isna() | (z.round(0) == hide), "", z.round(0).astype("Int64").astype(str))
