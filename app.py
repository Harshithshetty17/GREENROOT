"""GREENROOT — Streamlit presentation layer.

A thin, stateless view over the ``src`` package. Every computation — inference,
explanation, advisory synthesis, persistence — lives in a domain module; this
file only gathers input, dispatches, and renders.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
from urllib.parse import quote
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import (
    APP_VERSION,
    CONSENSUS_TOP_K,
    DEFAULT_INPUTS,
    FEATURE_BOUNDS,
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURE_UNITS,
    JACCARD_FIDELITY_THRESHOLD,
    OOD_ZSCORE_THRESHOLD,
    REPORTED_CV_ACCURACY,
)
from src.core import native
from src.services.soil_service import (
    district_profile,
    get_district_climate,
)
from src.utils import agronomy
from src.core.pwa import install as install_pwa
from src.core.theme import (
    BRAND,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    INK,
    MUTED,
    apply_matplotlib_theme,
    diverging_colours,
    series_palette,
    style_axes,
)
from src import auth
from src.database import db, db_manager
from src.models.batch import BatchProcessor, BatchResult, MAX_BATCH_ROWS, build_template
from src.models.inference import CropRecommender, ValidationError
from src.models.xai_engine import ExplainerConsensus
from src.services.soil_service import get_district_baseline, list_districts
from src.services.weather_service import get_weather
from src.utils.agronomy_advisory import CRITICAL, INFO, WARNING, generate_advisory
from src.utils.economics import estimate_cost, price_per_kg_from_bag
from src.utils.intervention import simulate_advisory
from src.utils import seasons as season_lib
from src.utils import i18n, reminders
from src.utils.plain_language import (
    category_name,
    confidence_band,
    describe_quantity,
    severity_name,
    text as _copy,
)
from src.utils.report_generator import (
    PDFUnavailableError,
    build_card,
    render_html,
    render_markdown,
    render_pdf,
    render_text,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="GREENROOT — Precision Agriculture DSS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Presentation constants
# --------------------------------------------------------------------------- #
apply_matplotlib_theme()

#: Chart colours come from the validated palette in src/core/theme.py, never
#: from literals here — that module documents why each data job gets which
#: colour family, and the ordering it uses was run through the CVD validator.
ACCENT = BRAND
NEUTRAL = MUTED

_CSS = """
<style>
  /* ---- Layout rhythm ------------------------------------------------- */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1480px; }
  section[data-testid="stSidebar"] { border-right: 1px solid #e4ebe6; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

  /* ---- Hero ----------------------------------------------------------- */
  /* The hero carries the current readings rather than sitting empty: an
     un-run app otherwise shows a large blank slab above the fold. */
  .gr-hero {
    background:
      radial-gradient(120% 140% at 88% -20%, rgba(255,255,255,.16) 0%,
                      rgba(255,255,255,0) 58%),
      linear-gradient(135deg, #115c3c 0%, #1b6e45 42%, #2f9e5f 100%);
    color: #fff; padding: 20px 26px 18px; border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(20,40,29,.06), 0 10px 28px rgba(20,40,29,.10);
  }
  .gr-hero h1 { margin: 0; font-size: 25px; font-weight: 700;
                letter-spacing: .4px; line-height: 1.15; }
  .gr-hero p  { margin: 3px 0 0; opacity: .90; font-size: 14px;
                line-height: 1.5; max-width: 72ch; }
  .gr-hero-top { display: flex; align-items: baseline; gap: 10px;
                 flex-wrap: wrap; }
  .gr-hero-place { margin-left: auto; font-size: 13px; font-weight: 600;
                   letter-spacing: .3px; padding: 4px 12px; border-radius: 999px;
                   background: rgba(255,255,255,.15); white-space: nowrap; }

  /* Reading strip: one tile per feature, label over value. */
  .gr-reads { display: grid; gap: 1px; margin-top: 16px;
              grid-template-columns: repeat(6, 1fr);
              background: rgba(255,255,255,.16); border-radius: 10px;
              overflow: hidden; }
  .gr-read { background: rgba(8,48,30,.22); padding: 9px 10px 10px; }
  .gr-read .k { font-size: 10.5px; letter-spacing: .7px; text-transform: uppercase;
                opacity: .78; font-weight: 600; }
  .gr-read .v { font-size: 19px; font-weight: 700; line-height: 1.2;
                margin-top: 2px; letter-spacing: -.3px; }
  .gr-read .v small { font-size: 11.5px; font-weight: 600; opacity: .72;
                      margin-left: 2px; letter-spacing: 0; }

  /* The three driver readouts sit side by side and are compared at a
     glance, so they need a shared baseline and a shared edge. */
  .gr-drivercard { border: 1px solid #dfe8e2; border-radius: 12px;
                   padding: 13px 15px; background: #fff; height: 100%; }
  .gr-drivercard.agree { background: #f3f8f5; border-color: #c3ddce; }
  .gr-drivercard .k { font-size: 11.5px; letter-spacing: .5px;
                      text-transform: uppercase; color: #5c6f63;
                      font-weight: 650; }
  .gr-drivercard .v { font-size: 16px; font-weight: 650; color: #14281d;
                      line-height: 1.35; margin-top: 5px; }
  .gr-drivercard.agree .v { color: #14603c; }

  /* After a run the banner carries the answer, not the inputs: it repeats
     on every tab, so it has to be short. */
  .gr-hero-compact { padding: 14px 26px 15px; margin-bottom: 14px; }
  .gr-hero-answer { display: flex; align-items: baseline; gap: 12px;
                    flex-wrap: wrap; margin-top: 8px; }
  .gr-hero-answer .crop { font-size: 30px; font-weight: 750; line-height: 1.05;
                          text-transform: uppercase; letter-spacing: .5px; }
  .gr-hero-answer .score { font-size: 14px; font-weight: 600; opacity: .92; }
  .gr-hero-reads { margin-top: 7px; font-size: 12.5px; opacity: .80;
                   letter-spacing: .2px; }

  /* A weak match must read as weak here too, or the banner contradicts the
     card directly beneath it. */
  .gr-hero-unsure {
    background:
      radial-gradient(120% 140% at 88% -20%, rgba(255,255,255,.14) 0%,
                      rgba(255,255,255,0) 58%),
      linear-gradient(135deg, #7a5c12 0%, #97731c 45%, #b8912e 100%);
  }

  /* Empty-state guidance: on-brand, and it says something worth reading
     instead of a stock blue notice restating the button label. */
  .gr-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
              margin: 4px 0 2px; }
  .gr-step { border: 1px solid #dfe8e2; border-radius: 14px; padding: 18px 20px;
             background: #fff; box-shadow: 0 1px 2px rgba(20,40,29,.04); }
  .gr-step .n { display: inline-flex; align-items: center; justify-content: center;
                width: 26px; height: 26px; border-radius: 999px; font-size: 13px;
                font-weight: 700; background: #e7f2eb; color: #14603c;
                margin-bottom: 9px; }
  .gr-step h4 { margin: 0 0 5px !important; padding: 0 !important;
                font-size: 15px; font-weight: 650; color: #14281d; }
  .gr-step p { margin: 0 !important; padding: 0 !important; font-size: 13.5px;
               line-height: 1.55; color: #5c6f63; }

  /* ---- Tabs ----------------------------------------------------------- */
  button[data-baseweb="tab"] { font-size: 14.5px; font-weight: 500;
                               padding: 10px 4px; }
  div[data-baseweb="tab-highlight"] { background-color: #1f7a4d; height: 3px; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #1f7a4d;
                                                     font-weight: 650; }

  /* ---- Surfaces ------------------------------------------------------- */
  .gr-card {
    border: 1px solid #dfe8e2; border-radius: 14px; padding: 20px 22px;
    background: #fff; box-shadow: 0 1px 2px rgba(20,40,29,.04);
  }
  .gr-primary {
    background: linear-gradient(160deg, #eef7f1 0%, #e3f0e9 100%);
    border: 1px solid #c3ddce; border-radius: 14px; padding: 22px 26px;
    box-shadow: 0 1px 2px rgba(20,40,29,.04);
  }
  .gr-primary .crop {
    font-size: 38px; font-weight: 750; color: #14603c; text-transform: uppercase;
    letter-spacing: .4px; line-height: 1.1; margin: 4px 0 2px;
  }
  .gr-primary .kn { font-size: 20px; font-weight: 600; color: #1f7a4d;
                    line-height: 1.3; margin: 0 0 6px; }
  .gr-primary .conf { font-size: 14.5px; color: #3d5548; font-weight: 500; }

  /* Below 60/100 the card drops its confident green: an uncertain answer
     should not look like a certain one. */
  .gr-primary.unsure {
    background: linear-gradient(160deg, #fdf9f0 0%, #fbf4e6 100%);
    border-color: #e8d6ac;
  }
  .gr-primary.unsure .crop { color: #8a6a1f; }
  .gr-primary.unsure .kn   { color: #8a6a1f; }
  .gr-primary .warnline {
    font-size: 12.5px; font-weight: 700; letter-spacing: .3px;
    text-transform: uppercase; color: #a6382a; margin-bottom: 8px;
  }

  /* Top control bar: the only things most people ever touch. */
  .gr-controls { margin-bottom: 2px; }
  .gr-wx { border: 1px solid #dfe8e2; border-radius: 10px; padding: 7px 12px;
           background: #f4f8f5; margin-bottom: 6px; }
  .gr-wx.live { background: #e7f2eb; border-color: #b9d8c6; }
  .gr-wx .k { font-size: 10px; letter-spacing: .7px; font-weight: 700;
              color: #5c6f63; }
  .gr-wx.live .k { color: #14603c; }
  .gr-wx .v { font-size: 16px; font-weight: 700; color: #14281d;
              line-height: 1.25; }
  .gr-wx .s { font-size: 11px; color: #5c6f63; }

  /* A recovery code is transcribed by hand onto paper. Big, monospaced and
     widely spaced so 8 and B cannot be confused at a glance. */
  .gr-code {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 27px; font-weight: 700; letter-spacing: 3px;
    text-align: center; color: #14281d;
    background: #fff; border: 2px dashed #c9a227; border-radius: 10px;
    padding: 14px 8px; margin: 6px 0 4px;
  }

  /* ---- Badges & metrics ----------------------------------------------- */
  .gr-badge {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-size: 12.5px; font-weight: 650; letter-spacing: .2px;
  }
  .gr-badge.ok   { background: #e7f2eb; color: #14603c; border: 1px solid #b9d8c6; }
  .gr-badge.warn { background: #fdf3f2; color: #a6382a; border: 1px solid #eec4bd; }
  .gr-metric { font-size: 42px; font-weight: 750; color: #1f7a4d;
               line-height: 1.05; margin: 2px 0 4px; letter-spacing: -1px; }
  .gr-sub { font-size: 13px; color: #5c6f63; line-height: 1.45; }
  .gr-driver { font-size: 19px; font-weight: 650; color: #14281d;
               line-height: 1.3; margin-top: 2px; }
  .gr-readout { font-size: 14px; color: #14281d; line-height: 1.5;
                background: #f4f8f5; border: 1px solid #e0e9e3;
                border-radius: 10px; padding: 10px 14px; }
  .gr-readout b { color: #14603c; }
  /* Sits beside the run button, so align to its optical centre. */
  .gr-readout-hint { font-size: 13px; color: #5c6f63; line-height: 1.5;
                     margin: 0; padding-top: 7px; }

  /* ---- Advisory items -------------------------------------------------- */
  .gr-advisory {
    border-left: 3px solid #dfe8e2; padding: 11px 16px; margin-bottom: 10px;
    border-radius: 0 10px 10px 0; font-size: 14px; line-height: 1.55;
  }
  .gr-advisory.critical { border-left-color: #c0392b; background: #fdf4f2; }
  .gr-advisory.warning  { border-left-color: #d98b0e; background: #fdf9f0; }
  .gr-advisory.info     { border-left-color: #1f7a4d; background: #f3f8f5; }
  .gr-advisory b { color: #14281d; font-weight: 650; }

  /* ---- Widgets --------------------------------------------------------- */
  div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700;
                                     color: #14281d; }
  div[data-testid="stMetricLabel"] { font-size: 13px; color: #5c6f63; }
  div[data-testid="stProgressBarTrack"] { background-color: #e3ebe6; height: 10px; }
  div[data-testid="stProgressBarTrack"] > div { background-color: #1f7a4d; }
  /* Larger, calmer tap targets: this is used on a phone in a field. */
  .stButton button { font-weight: 600; padding: .55rem 1rem; }
  .stDownloadButton button { font-weight: 600; }
  div[data-testid="stExpander"] details {
    border: 1px solid #e4ebe6; border-radius: 10px;
  }
  /* Tables: readable rather than dense. */
  div[data-testid="stDataFrame"] { border-radius: 10px; }

  /* ---- Phone ----------------------------------------------------------- */
  /* This is used one-handed, outdoors, on a mid-range Android. Columns must
     stack rather than squeeze, and every control must be thumb-sized. */
  @media (max-width: 820px) {
    /* The collapsed sidebar is 320px wide translated -300px, so its right
       20px sits over the content area. Inset the content past that rail or
       the first character of every line is painted over -- and drop the
       rail's fill, otherwise it reads as a stripe down the left edge and
       the page looks off-centre. Gutters are then equal on both sides. */
    .block-container { padding: 1rem 1.9rem 2.5rem 1.9rem; }
    section[data-testid="stSidebar"][aria-expanded="false"] {
      background: transparent; border-right: none;
    }

    /* Streamlit columns shrink rather than wrap by default; force a stack. */
    div[data-testid="stHorizontalBlock"] { flex-direction: column; gap: .85rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
      width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }

    /* A row of metrics would read better side by side than stacked, but
       there is no safe selector for it: :has(... stMetric) also matches the
       main two-column layout, because a metric sits somewhere inside its
       left column, and that layout must stack or the crop card wraps to
       three words a line. Stacked metrics cost space; an unstacked page is
       broken. */

    .gr-hero { padding: 16px 16px 14px; border-radius: 14px; }
    .gr-hero h1 { font-size: 21px; }
    .gr-hero p  { font-size: 12.5px; }
    .gr-hero-place { font-size: 12px; padding: 3px 10px; }
    .gr-hero-compact { padding: 13px 16px 14px; }
    .gr-hero-answer .crop { font-size: 25px; }
    .gr-hero-answer .score { font-size: 13px; }
    .gr-hero-reads { font-size: 11.5px; }
    /* Six tiles across 390px gives 55px each -- too narrow for "190 mm".
       Three across two rows keeps every value on one line. */
    .gr-reads { grid-template-columns: repeat(3, 1fr); margin-top: 13px; }
    .gr-read { padding: 8px 9px 9px; }
    .gr-read .v { font-size: 17px; }
    .gr-steps { grid-template-columns: 1fr; gap: 10px; }
    .gr-step { padding: 14px 16px; }
    /* Stacked above the button here, not beside it. */
    .gr-readout-hint { padding-top: 0; margin-bottom: 4px; }

    .gr-primary { padding: 18px 18px; }
    .gr-primary .crop { font-size: 30px; }
    .gr-metric { font-size: 34px; }
    .gr-driver { font-size: 17px; }
    .gr-advisory { font-size: 14.5px; padding: 12px 14px; }

    /* Thumb-sized targets; 44px is the accepted minimum. */
    .stButton button, .stDownloadButton button {
      min-height: 46px; font-size: 15px; width: 100%;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input { min-height: 42px; font-size: 16px; }
    /* 16px on inputs stops iOS Safari zooming the page on focus. */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] { min-height: 42px; }

    /* Tabs scroll horizontally; keep them tappable. */
    button[data-baseweb="tab"] { font-size: 13.5px; padding: 10px 2px; }

    /* The sidebar is a drawer here, so let it use the screen it needs. */
    section[data-testid="stSidebar"] { min-width: 82vw !important;
                                       max-width: 90vw !important; }
  }

  @media (max-width: 420px) {
    .gr-hero h1 { font-size: 19px; }
    .gr-read .v { font-size: 16px; }
    .gr-primary .crop { font-size: 26px; }
  }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)



# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the model…")
def load_recommender() -> Optional[CropRecommender]:
    """Load the deployed ensemble once per server process."""
    try:
        return CropRecommender().load()
    except FileNotFoundError as exc:
        st.error(f"**Model artefacts unavailable.**\n\n{exc}")
        return None


@st.cache_resource(show_spinner="Getting the checks ready…")
def load_explainer() -> Optional[ExplainerConsensus]:
    """Build the TreeSHAP/LIME consensus engine once per server process."""
    try:
        return ExplainerConsensus().fit()
    except Exception as exc:  # noqa: BLE001 - XAI is optional; never block the app.
        logging.warning("Explainability engine unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def load_batch_processor() -> Optional[BatchProcessor]:
    """Bulk advisory engine, sharing the already-loaded ensemble."""
    try:
        return BatchProcessor()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_districts() -> List[str]:
    """District list for the baseline selector."""
    return list_districts()


@st.cache_resource
def ensure_database() -> bool:
    """Initialise the audit schema once per server process."""
    try:
        db_manager.init_db()
        return True
    except Exception as exc:  # noqa: BLE001 - a read-only volume must not crash the UI.
        logging.warning("Audit database unavailable: %s", exc)
        return False


def _as_data_uri(document: str) -> str:
    """Wrap a complete HTML document as a base64 ``data:`` URI.

    The soil health card is a standalone document whose stylesheet targets
    generic selectors (``body``, ``table``, ``section``). Injecting it into the
    dashboard would leak those rules into Streamlit's own markup, so it is
    rendered inside an isolated iframe instead.
    """
    encoded = base64.b64encode(document.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded}"


def _style_axes(axes: plt.Axes, *, grid_axis: str = "x") -> None:
    """Apply the shared chart styling defined in the theme module."""
    style_axes(axes, grid_axis=grid_axis)


def current_language() -> str:
    """The interface language for this visitor.

    Held in session_state so it works for a guest -- the people who most need
    Kannada are the least likely to have made an account, so gating the
    language behind sign-in would put it out of reach of its own audience.
    A signed-in user's choice is persisted to their profile as well.
    """
    return i18n.normalise(st.session_state.get("ui_language", i18n.ENGLISH))


def tr(key: str, simple: bool = True) -> str:
    """Interface copy in the current register *and* the current language."""
    return _copy(key, simple, current_language())


def score_phrase(confidence: float) -> str:
    """``91 out of 100`` / ``೧೦೦ ರಲ್ಲಿ ೯೧`` -- the score, in words.

    Kannada puts the total first, so this is a reordering rather than a
    substitution; a template with the number interpolated in English order
    would read wrong.
    """
    value = f"{confidence:.0f}"
    if current_language() == i18n.KANNADA:
        digits = value.translate(str.maketrans("0123456789", "೦೧೨೩೪೫೬೭೮೯"))
        return f"೧೦೦ ರಲ್ಲಿ {digits}"
    return f"{value} out of 100"


def _bags_word(count: int, simple: bool = True) -> str:
    """``bag`` / ``bags`` / ``ಚೀಲ``.

    Kannada does not inflect this noun for number the way English does, so
    one word covers both and the English singular/plural split happens only
    on the English side.
    """
    if simple and current_language() == i18n.KANNADA:
        return i18n.KANNADA_EXTRA["bags"]
    return "bag" if count == 1 else "bags"


def _acres_words(acres: float, simple: bool = True) -> str:
    """``acre`` / ``acres`` / ``ಎಕರೆ``."""
    if simple and current_language() == i18n.KANNADA:
        return i18n.KANNADA_EXTRA["acre"]
    return "acre" if acres == 1 else "acres"


def _crop_label(crop: str, simple: bool = True) -> str:
    """A crop's name in the reader's language."""
    if simple and current_language() == i18n.KANNADA:
        return agronomy.kannada_name(crop)
    return crop.title()


def band_for(confidence: float):
    """Confidence band in the current language.

    "Fair match, 43 out of 100" is the sentence a farmer actually decides on,
    so it must not stay English on an otherwise Kannada screen.
    """
    band = confidence_band(confidence)
    label, detail = i18n.band_words(band.label, band.detail, current_language())
    return dataclasses.replace(band, label=label, detail=detail)


def t_extra(key: str, english: str) -> str:
    """One-off strings that are not in the COPY catalogue."""
    return i18n.translate(key, english, current_language(), simple=is_simple())


def is_simple() -> bool:
    """Whether to address the reader as a farmer rather than an examiner.

    One switch drives both personas: the farmer portal is plain-language mode,
    and Examiner / AI Mode is its inverse. Keeping it as a single derived
    predicate means every ``simple``-conditioned string in this file keeps
    working unchanged.
    """
    return not bool(st.session_state.get("examiner_mode", False))


def _feature_word(feature: str, simple: bool = True) -> str:
    """Return a short display name for a feature, plain or technical."""
    if not simple:
        return feature
    return tr(f"field_{feature}", True).split(" — ")[0].replace(" (N)", "").replace(
        " (P)", ""
    ).replace(" (K)", "")


# --------------------------------------------------------------------------- #
# Sidebar — data ingestion
# --------------------------------------------------------------------------- #
def _seed_defaults() -> None:
    """Populate widget state with defaults on first render.

    Widgets below bind by ``key`` alone. Passing a ``value=`` as well as a
    pre-populated session-state entry makes Streamlit warn that one of the two
    is ignored, so the default is seeded here instead — once, before any widget
    for that key is instantiated.
    """
    for name, default in DEFAULT_INPUTS.items():
        st.session_state.setdefault(f"in_{name}", float(default))


def _apply_district(district: str) -> None:
    """Load a district's full seven-feature profile into the input widgets.

    Wired to the selector's ``on_change`` so that choosing a district *is* the
    load -- the previous design needed a separate "load baseline" press, which
    meant the readings on screen could silently disagree with the district
    named beside them.
    """
    baseline = get_district_baseline(district)
    climate = get_district_climate(district)
    st.session_state["baseline"] = baseline
    st.session_state["climate"] = climate
    for key, value in district_profile(district).items():
        st.session_state[f"in_{key}"] = float(np.clip(value, *FEATURE_BOUNDS[key]))
    # A district load supersedes any earlier live reading.
    st.session_state.pop("weather", None)


def _on_district_change() -> None:
    _apply_district(st.session_state.get("district_pick", ""))


def _adopt_stored_language() -> None:
    """On first render after sign-in, switch to the language on the profile.

    Only once, and only when the visitor has not already chosen in this
    session -- otherwise it would keep overriding a deliberate switch.
    """
    if st.session_state.get("_language_adopted"):
        return
    user = current_user()
    if user is not None and i18n.is_supported(user.language):
        st.session_state["ui_language"] = user.language
        st.session_state["_language_adopted"] = True


def render_language_picker() -> None:
    """Language choice, available to guests and signed-in users alike."""
    codes = [code for code, _ in i18n.available_languages()]
    labels = dict(i18n.available_languages())
    current = current_language()

    chosen = st.sidebar.radio(
        # Bilingual on purpose: somebody who cannot read English has not
        # chosen Kannada yet, so this one label has to work in both.
        f"Language / {i18n.KANNADA_EXTRA['language']}",
        options=codes,
        index=codes.index(current),
        format_func=lambda code: labels[code],
        horizontal=True,
        key="ui_language",
    )

    if chosen == i18n.KANNADA:
        # Said in Kannada, because the person reading it chose Kannada.
        st.sidebar.caption(i18n.review_note(i18n.KANNADA))
        numbers = i18n.coverage()
        st.sidebar.caption(
            f"{numbers['copy_translated']} of {numbers['copy_keys']} screens "
            f"translated; the rest still show English."
        )

    # Remember it for a signed-in user, so a new phone comes up in their
    # language. The users table has carried a `language` column since v2.
    user = current_user()
    if user is not None and user.language != chosen:
        try:
            auth.update_profile(user.id, language=chosen)
        except Exception as exc:  # noqa: BLE001 - never block on a preference.
            logging.warning("Could not persist language choice: %s", exc)


def current_user():
    """The signed-in user, or None for a guest.

    Resolved from the session token on every run rather than cached: a
    revoked or expired token must stop working immediately, and an object
    left in session_state would outlive the session it represents.

    An unreachable database degrades to guest. Everything a guest can do
    works without persistence, so a storage failure must not lock people out
    of the recommendation itself.
    """
    try:
        return auth.user_for_token(st.session_state.get("auth_token"))
    except Exception as exc:  # noqa: BLE001 - never block the app on storage.
        logging.warning("Could not resolve session: %s", exc)
        return None


def _sign_out() -> None:
    auth.revoke_token(st.session_state.pop("auth_token", None))
    for key in ("account_panel", "signin_error"):
        st.session_state.pop(key, None)


def render_help_sidebar() -> None:
    """What this is, and the disclaimer. Shown to everyone.

    This used to sit inside the signed-in branch, which meant a guest -- a
    first-time visitor, the person most likely to want it -- could not reach
    it at all.
    """
    with st.sidebar.expander("Help & about", expanded=False):
        st.markdown(
            "**What is this?** GREENROOT suggests a crop for your land from "
            "seven soil and weather readings, and shows how it decided.\n\n"
            "**Is it a promise?** No. It is advice to help you decide. Check "
            "with your local agriculture officer before sowing.\n\n"
            "**Do I need an account?** No. Everything works without one."
        )
        st.caption(
            f"GREENROOT v{APP_VERSION} · 22 crops · "
            f"advisory output only"
        )


def render_account_sidebar(simple: bool) -> None:
    """Sign in, profile and settings -- all of it in the sidebar.

    Nothing here appears in the recommendation flow. A farmer who never signs
    in should see one small affordance and nothing else: the app works fully
    as a guest, and that is the default.
    """
    user = current_user()
    st.sidebar.markdown("---")

    # Shown once, immediately after sign-up or a reset. Only the hash is
    # stored, so there is no second chance to display it.
    fresh = st.session_state.get("fresh_recovery_code")
    if fresh:
        st.sidebar.warning("**Write this down now**")
        st.sidebar.markdown(
            f"<div class='gr-code'>{fresh}</div>", unsafe_allow_html=True
        )
        st.sidebar.caption(
            "This is the only way back into your account if you forget your "
            "PIN. We cannot show it again and we cannot look it up. There is "
            "no letter O and no number 1 in these codes."
        )
        if st.sidebar.button("I have written it down", width="stretch"):
            st.session_state.pop("fresh_recovery_code", None)
            st.rerun()

    if user is None:
        st.sidebar.markdown(f"#### {t_extra('account', 'Your account')}")
        st.sidebar.caption(
            "You do not need an account. Sign in only if you want your saved "
            "advice on more than one phone."
            if simple
            else "Optional. Signing in scopes the ledger to this account."
        )
        with st.sidebar.expander("Sign in / Create account", expanded=False):
            phone = st.text_input("Mobile number", key="signin_phone",
                                  placeholder="9876543210", max_chars=15)
            pin = st.text_input(f"{auth.accounts.PIN_LENGTH}-digit PIN",
                                key="signin_pin", type="password", max_chars=6)
            name = st.text_input("Your name (new accounts only)",
                                 key="signin_name")
            go, make = st.columns(2)
            if go.button(t_extra("sign_in", "Sign in"), width="stretch"):
                try:
                    signed = auth.sign_in(phone, pin)
                    st.session_state["auth_token"] = auth.issue_token(signed.id)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
            if make.button("Create", width="stretch"):
                try:
                    created, recovery = auth.register(
                        phone, pin, display_name=name or None,
                        district=st.session_state.get("district_pick"))
                    st.session_state["auth_token"] = auth.issue_token(created.id)
                    # Shown exactly once. Only its hash is kept, so it can
                    # never be displayed again.
                    st.session_state["fresh_recovery_code"] = recovery
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
            st.caption(
                "Your PIN is stored scrambled and cannot be read back, even "
                "by us. Do not use 1234 or your birth year."
            )

        with st.sidebar.expander("Forgot your PIN?", expanded=False):
            st.caption(
                "Use the recovery code you wrote down when you created the "
                "account."
            )
            r_phone = st.text_input("Mobile number", key="rec_phone",
                                    placeholder="9876543210", max_chars=15)
            r_code = st.text_input("Recovery code", key="rec_code",
                                   placeholder="ABCD-EFGH", max_chars=12)
            r_pin = st.text_input("New PIN", key="rec_pin", type="password",
                                  max_chars=6)
            if st.button("Reset my PIN", width="stretch", key="rec_go"):
                try:
                    restored, replacement = auth.reset_pin_with_code(
                        r_phone, r_code, r_pin)
                    st.session_state["auth_token"] = auth.issue_token(
                        restored.id)
                    st.session_state["fresh_recovery_code"] = replacement
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
            st.caption(
                "Using a code cancels it. You will get a new one to write "
                "down."
            )
            # Said to everybody rather than only to the accounts it applies
            # to: a message that appears for one number and not another
            # answers "is this number registered?" to anyone who asks.
            st.caption(
                "Accounts made before recovery codes existed do not have "
                "one. If you had an account back then, sign in with your PIN "
                "and create a code from your account panel."
            )
        return

    # ---- signed in ------------------------------------------------------
    st.sidebar.markdown(f"#### 👤 {user.greeting}")
    st.sidebar.caption(user.masked_phone)

    if st.sidebar.button(t_extra("log_out", "Log out"), width="stretch"):
        _sign_out()
        st.rerun()

    with st.sidebar.expander("Profile", expanded=False):
        districts = cached_districts()
        name = st.text_input("Name", value=user.display_name or "",
                             key="pf_name")
        village = st.text_input("Village", value=user.village or "",
                                key="pf_village")
        index = districts.index(user.district) if user.district in districts else 0
        district = st.selectbox("Usual district", districts, index=index,
                                key="pf_district")
        acres = st.number_input("Usual plot size (acres)", 0.1, 1000.0,
                                float(user.acres or 1.0), 0.5, key="pf_acres")
        if st.button("Save profile", width="stretch"):
            auth.update_profile(user.id, display_name=name or None,
                                village=village or None, district=district,
                                acres=float(acres))
            st.success("Saved.")
            st.rerun()

    with st.sidebar.expander("My fields", expanded=False):
        st.caption(
            "Save each field once, then load it instead of retyping the "
            "readings."
        )
        saved = auth.list_plots(user.id)
        if saved:
            picked = st.selectbox(
                "Your saved fields",
                options=saved,
                format_func=lambda p: p.label,
                key="plot_pick",
            )
            st.caption(picked.summary())
            load, drop = st.columns(2)
            if load.button("Load", width="stretch", key="plot_load"):
                # Writing straight into the widget keys is what makes this a
                # load rather than a suggestion.
                for feature, value in picked.readings.items():
                    st.session_state[f"in_{feature}"] = float(
                        np.clip(value, *FEATURE_BOUNDS[feature]))
                st.session_state["district_pick"] = picked.district
                st.session_state["acres"] = float(picked.acres)
                st.session_state.pop("prediction", None)
                st.rerun()
            if drop.button("Delete", width="stretch", key="plot_drop"):
                auth.delete_plot(user.id, picked.id)
                st.rerun()
        else:
            st.caption("No fields saved yet.")

        st.markdown("---")
        new_name = st.text_input("Save these readings as", key="plot_name",
                                 placeholder="North field", max_chars=40)
        if st.button("Save this field", width="stretch", key="plot_save"):
            try:
                auth.save_plot(
                    user.id,
                    new_name,
                    st.session_state.get("district_pick", ""),
                    float(st.session_state.get("acres", 1.0)),
                    {f: float(st.session_state[f"in_{f}"])
                     for f in FEATURE_NAMES},
                )
                st.success("Saved.")
                st.rerun()
            except auth.PlotError as exc:
                st.error(str(exc))

    with st.sidebar.expander("Settings", expanded=False):
        st.caption(
            "An English-only interface today. Kannada is planned; the crop "
            "names on your card are already bilingual."
        )
        st.markdown("**Change your PIN**")
        old = st.text_input("Current PIN", type="password", key="pin_old",
                            max_chars=6)
        new = st.text_input("New PIN", type="password", key="pin_new",
                            max_chars=6)
        if st.button("Change PIN", width="stretch"):
            try:
                auth.change_pin(user.id, old, new)
                st.success("PIN changed.")
            except auth.AuthError as exc:
                st.error(str(exc))

        st.markdown("---")
        st.markdown("**Recovery code**")
        if auth.has_recovery_code(user.id):
            st.caption(
                "You have one. Making a new code cancels the old one, so only "
                "do this if you have lost it."
            )
        else:
            st.caption(
                "This account has no recovery code — it was made before they "
                "existed. Make one now, or a forgotten PIN will lock you out "
                "for good."
            )
        rc_pin = st.text_input("Your PIN, to confirm", type="password",
                               key="rc_pin", max_chars=6)
        if st.button("Make a new recovery code", width="stretch", key="rc_go"):
            try:
                st.session_state["fresh_recovery_code"] = (
                    auth.regenerate_recovery_code(user.id, rc_pin))
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))

        st.markdown("---")
        st.markdown("**Delete my account**")
        st.caption(
            "This removes your account, your PIN and your saved plots from "
            "this device and the server. Advice you saved stays in the "
            "records, but is no longer linked to you."
        )
        confirm = st.text_input('Type DELETE to confirm', key="del_confirm")
        if st.button("Delete my account permanently", width="stretch"):
            if confirm.strip().upper() != "DELETE":
                st.error("Type DELETE in the box to confirm.")
            else:
                auth.delete_account(user.id)
                _sign_out()
                st.rerun()



def render_controls() -> Dict[str, object]:
    """The top control bar, in the main area rather than the sidebar.

    A farmer on a phone never opens the sidebar. Everything that has to be
    touched on a normal run -- where the land is, how big it is, what the
    weather is doing -- lives here; the seven raw readings stay one tap away
    in an expander, because most people will accept the district baseline.
    """
    _seed_defaults()
    simple = is_simple()
    districts = cached_districts()

    if "district_pick" not in st.session_state:
        st.session_state["district_pick"] = (
            "Udupi" if "Udupi" in districts else districts[0]
        )
        _apply_district(st.session_state["district_pick"])

    # On a phone these four controls stack, and a farmer scrolls past two
    # full screens of dropdowns before seeing a single word of advice. They
    # are set once and rarely changed, so they collapse behind a one-line
    # summary of what they currently say.
    current = st.session_state.get("district_pick", "")
    acres_now = float(st.session_state.get("acres", 1.0))

    # Streamlit restores a selectbox by looking up the label it stored, so
    # switching language leaves this key holding a stale label rather than a
    # season key. Repair it here, before the widget below is built -- after
    # that, assigning to a widget key is an error.
    season_key = season_lib.canonical(
        st.session_state.get("season", season_lib.KHARIF)
    )
    if st.session_state.get("season") != season_key:
        st.session_state["season"] = season_key
    season_now = i18n.season_name(
        season_key, season_lib.SEASONS[season_key][0],
        current_language() if simple else i18n.ENGLISH,
    )
    summary = (
        f"📍 {current}  ·  {acres_now:g} {_acres_words(acres_now, simple)}"
        f"  ·  {season_now}"
    )
    with st.expander(summary, expanded=False):
        bar = st.columns([1.3, 0.9, 1.1, 1.2], gap="medium")

        district = bar[0].selectbox(
            t_extra("your_district", "Your district") if simple else "District",
            options=districts,
            key="district_pick",
            on_change=_on_district_change,
            help=(
                "Picking your district fills in the typical soil and weather for "
                "that area."
                if simple
                else "Loads the NFSM survey median where one exists, otherwise "
                "the curated agro-climatic baseline, plus regional climate "
                "normals."
            ),
        )

        acres = bar[1].number_input(
            t_extra("how_many_acres", "How many acres?") if simple else "Area (acres)",
            min_value=0.1,
            max_value=1000.0,
            value=float(st.session_state.get("acres", 1.0)),
            step=0.5,
            key="acres",
            help="Fertiliser bags and the money estimate are worked out for this "
                 "area.",
        )

        season_keys = list(season_lib.SEASONS)
        if "season" not in st.session_state:
            st.session_state["season"] = season_lib.default_season(date.today().month)
        bar[2].selectbox(
            t_extra("when_sow", "When will you sow?") if simple else "Cropping season",
            options=season_keys,
            format_func=lambda key: i18n.season_name(
                key, season_lib.SEASONS[key][0],
                current_language() if simple else i18n.ENGLISH,
            ),
            key="season",
            help=(
                "A crop can suit your soil and still be wrong for the time of "
                "year. We check both."
                if simple
                else "Sowing window; used to flag calendar mismatches the edaphic "
                "model cannot see."
            ),
        )

        with bar[3]:
            weather = st.session_state.get("weather")
            climate = st.session_state.get("climate")
            if weather is not None and weather.is_live:
                st.markdown(
                    f"<div class='gr-wx live'><div class='k'>"
                    f"{t_extra('live_weather', 'LIVE WEATHER')}</div>"
                    f"<div class='v'>{weather.temperature:.0f}°C · "
                    f"{weather.humidity:.0f}% RH</div>"
                    f"<div class='s'>{weather.city}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                where = getattr(climate, "district", district)
                source = i18n.phrase(
                    "weather_normals", f"normals for {where}",
                    current_language() if simple else i18n.ENGLISH,
                    district=where,
                )
                st.markdown(
                    f"<div class='gr-wx'><div class='k'>"
                    f"{t_extra('typical_weather', 'TYPICAL WEATHER')}</div>"
                    f"<div class='v'>{st.session_state['in_temperature']:.0f}°C · "
                    f"{st.session_state['in_humidity']:.0f}% RH</div>"
                    f"<div class='s'>{source}</div></div>",
                    unsafe_allow_html=True,
                )
            if st.button(
                t_extra("use_live_weather", "Use live weather") if simple else "Fetch live telemetry",
                width="stretch",
                key="fetch_wx",
            ):
                reading = get_weather(district, st.session_state.get("wx_key") or None)
                st.session_state["weather"] = reading
                for key, value in reading.as_dict().items():
                    st.session_state[f"in_{key}"] = float(
                        np.clip(value, *FEATURE_BOUNDS[key])
                    )
                if not reading.is_live:
                    st.toast("No live reading — kept the typical weather.")
                st.rerun()

    # ---- Manual soil card adjustments, folded away --------------------- #
    baseline = st.session_state.get("baseline")
    if baseline is not None:
        provenance = (
            i18n.phrase(
                "soil_from_survey",
                f"Typical of {baseline.sample_count:,} soil tests from "
                f"{baseline.district}.",
                current_language(),
                district=baseline.district, count=f"{baseline.sample_count:,}",
            )
            if baseline.is_survey_backed and simple
            else f"NFSM median of {baseline.sample_count:,} laboratory samples "
            f"from {baseline.district}."
            if baseline.is_survey_backed
            else i18n.phrase(
                "soil_not_survey",
                f"Typical soil for {baseline.district} — not from a survey, "
                f"so correct it below if you have a soil card.",
                current_language(), district=baseline.district,
            )
            if simple
            else f"Curated agro-climatic baseline for {baseline.district} "
            f"(source: {baseline.source}); no survey samples for this unit."
        )

    values: Dict[str, float] = {}
    with st.expander(
        t_extra("change_readings", "Change my soil readings") if simple else "Manual feature override",
        expanded=False,
    ):
        if baseline is not None:
            st.caption(provenance)
        soil = st.columns(4)
        for column, name in zip(soil, ("N", "P", "K")):
            low, high = FEATURE_BOUNDS[name]
            index = FEATURE_NAMES.index(name)
            values[name] = column.number_input(
                tr(f"field_{name}", simple)
                if simple
                else f"{FEATURE_LABELS[index]} ({FEATURE_UNITS[index]})",
                min_value=float(low),
                max_value=float(high),
                step=1.0,
                key=f"in_{name}",
            )
        values["ph"] = soil[3].number_input(
            tr("field_ph", simple),
            min_value=float(FEATURE_BOUNDS["ph"][0]),
            max_value=float(FEATURE_BOUNDS["ph"][1]),
            step=0.1,
            key="in_ph",
        )

        climate_columns = st.columns(3)
        for column, name in zip(climate_columns,
                                ("temperature", "humidity", "rainfall")):
            low, high = FEATURE_BOUNDS[name]
            index = FEATURE_NAMES.index(name)
            values[name] = column.slider(
                f"{tr(f'field_{name}', simple)} ({FEATURE_UNITS[index]})",
                min_value=float(low),
                max_value=float(high),
                step=0.5,
                key=f"in_{name}",
            )

        st.text_input(
            "Weather key (optional)" if simple else "OpenWeatherMap API key",
            type="password",
            key="wx_key",
            help="Leave empty if you do not have one — the app still works.",
        )

    return {
        "district": district,
        "city": district,
        "acres": float(acres),
        "features": values,
        "season": season_lib.canonical(st.session_state.get("season")),
    }


# --------------------------------------------------------------------------- #
# Tab 1 — Precision Recommendation
# --------------------------------------------------------------------------- #
def render_empty_state(simple: bool) -> None:
    """What the user sees before the first run.

    A stock notice restating the button label wastes the only screen most
    users will judge the tool on, so this explains what the system actually
    does — including that it shows its reasoning, which is the point of it.
    """
    steps = (
        [
            # Name the control by the words printed on it. "The soil card
            # expander" is what it is called in the source, not on screen.
            ("Check your readings", "The numbers above describe your land. "
             "Change the district at the top, or tap \u201cChange my soil "
             "readings\u201d to correct them yourself."),
            ("Press the green button", "GREENROOT weighs your soil against "
             "22 crops and picks the one that fits best."),
            ("See why, not just what", "It shows which reading decided it, "
             "what to add to the soil, and whether the season suits."),
        ]
        if simple
        else [
            ("Set the feature vector", "Seven inputs: N, P, K, pH, temperature, "
             "humidity and rainfall, bounded and validated on entry."),
            ("Run the stacking ensemble", "Random Forest, AdaBoost and kNN feed "
             "a logistic meta-learner over 22 crop classes."),
            ("Audit the decision", "TreeSHAP and LIME are compared by Jaccard "
             "overlap at k=3, with Z-score covariate-shift flags."),
        ]
    )
    cards = "".join(
        f"<div class='gr-step'><div class='n'>{i}</div>"
        f"<h4>{title}</h4><p>{body}</p></div>"
        for i, (title, body) in enumerate(steps, start=1)
    )
    st.markdown(f"<div class='gr-steps'>{cards}</div>", unsafe_allow_html=True)


def _rupees(amount: float) -> str:
    """Indian digit grouping: 1,23,456 rather than 123,456.

    The Western three-digit grouping is genuinely harder for the intended
    reader to parse at a glance, and a money figure nobody can read quickly
    is not doing its job.
    """
    negative = amount < 0
    digits = f"{abs(round(amount)):.0f}"
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join(parts + [tail])
    return f"{'-' if negative else ''}₹{digits}"


def render_reminders(simple: bool) -> None:
    """What is due on advice already saved.

    The buildable half of a notification. Push needs a scheduler this
    deployment does not have, but the value -- "your crop is 34 days old,
    time for the first top dressing" -- is arithmetic over the saved date and
    the crop roadmap, so it is computed here and shown when the farmer next
    opens the app.
    """
    if not ensure_database():
        return
    viewer = current_user()
    try:
        frame = db_manager.fetch_audit_history(
            limit=40,
            **({"user_id": viewer.id} if viewer else {"guest_only": True}),
        )
    except Exception as exc:  # noqa: BLE001 - a reminder is never worth a crash.
        logging.warning("Could not read records for reminders: %s", exc)
        return
    if frame.empty:
        return

    due = reminders.upcoming(frame.to_dict("records"))
    if not due:
        return

    st.markdown(
        f"#### {t_extra('whats_due', 'What to do next') if simple else 'Field calendar'}"
    )
    language = current_language()
    kannada = simple and language == i18n.KANNADA
    for item in due:
        # The stage name and the action are the two halves a farmer acts on,
        # so both are localised; the day offset is what the harvest line
        # interpolates, and it is the gap between sowing and the due date.
        offset = (item.due_on - item.saved_on).days
        stage, action = i18n.stage_words(
            item.stage, item.action, language if simple else i18n.ENGLISH,
            day=offset,
        )
        crop = agronomy.kannada_name(item.crop) if kannada else item.crop.title()
        when = i18n.when_words(
            item.days_away, item.when_words(),
            language if simple else i18n.ENGLISH,
        )
        body = (
            f"**{crop} — {stage}** — {when}"
            f"{('  ·  ' + item.district) if item.district else ''}  \n"
            f"{action}"
        )
        {"overdue": st.error, "now": st.warning}.get(item.urgency, st.info)(body)

    st.caption(
        t_extra(
            "reminder_estimate",
            "These dates are worked out from the day you saved the advice, "
            "not the day you actually sowed — so treat them as close, not "
            "exact.",
        )
    )
    st.markdown("---")


def render_commercial_panel(state: Dict[str, object], simple: bool) -> None:
    """What the crop is worth, what to buy, and when to do it.

    The model answers "which crop"; this answers the three questions a farmer
    asks immediately afterwards -- what will it earn, what do I carry home
    from the dealer, and can I spray today.
    """
    prediction = state["prediction"]
    acres = float(state.get("acres") or 1.0)
    crop = prediction.crop

    base = agronomy.profile(crop)
    if base is None:
        # A class with no commercial profile: say so rather than showing
        # blank cards that look like a loading failure.
        st.info(
            f"No price or yield benchmark is on file for {crop.title()}, so "
            f"the money estimate is not shown. Everything else on this page "
            f"still applies."
        )
        return

    # The benchmark is a starting point, not a quotation. Let them correct it
    # and have every figure below follow -- a rate they recognise is the
    # difference between a number they act on and a number they ignore.
    entered = st.number_input(
        f"Your mandi rate for {base.english.lower()} (₹ per quintal)"
        if simple
        else f"APMC rate override — {base.english} (₹/quintal)",
        min_value=0.0,
        max_value=200000.0,
        value=float(base.price_per_quintal),
        step=50.0,
        key=f"price_{crop}",
        help=f"Benchmark is {_rupees(base.price_per_quintal)}. Change it to "
             f"your own mandi's rate and the figures below follow.",
    )
    override = float(entered) if entered > 0 else None
    crop_profile = agronomy.with_overrides(crop, price_per_quintal=override)
    money = crop_profile.economics(acres)
    if override is not None and abs(override - base.price_per_quintal) > 1:
        st.caption(
            f"Using your rate of {_rupees(override)}/quintal instead of the "
            f"{_rupees(base.price_per_quintal)} benchmark."
        )

    # ---- Financial ROI -------------------------------------------------- #
    earn_heading = (
        t_extra("what_earn", "What this could earn")
        if simple
        else "Benchmark economics"
    )
    st.markdown(f"#### {earn_heading}")
    roi = st.columns(3)
    roi[0].metric(
        t_extra("expected_harvest", "Expected harvest") if simple
        else "Gross yield",
        f"{money.yield_quintals:,.0f} quintal"
        + ("" if abs(money.yield_quintals - 1) < 0.5 else "s"),
        help=f"{crop_profile.yield_quintal_per_acre:g} quintals per acre "
             f"× {acres:g} acre(s).",
    )
    roi[1].metric(
        t_extra("mandi_rate", "Mandi rate") if simple else "APMC benchmark",
        f"{_rupees(crop_profile.price_per_quintal)}/qtl",
        help="Edit this above if you know your own mandi's rate.",
    )
    # A loss gets its own label and an unsigned figure. "Money left over:
    # -20,000" is a contradiction, and on a phone the minus sign is one thin
    # stroke in front of a large number -- the reading a farmer is most
    # likely to take from it is the exact opposite of what it means.
    roi[2].metric(
        (t_extra("money_short", "Money you would lose") if simple
         else "Net margin (loss)")
        if money.is_loss
        else t_extra("money_left", "Money left over") if simple
        else "Net margin",
        _rupees(abs(money.net)) if money.is_loss else _rupees(money.net),
    )
    roi[2].caption(
        (t_extra("after_costs_loss", "more than this crop brings in")
         if simple else f"against {_rupees(money.cost)} of costs")
        if money.is_loss
        else (t_extra("after_costs", "after costs") if simple
              else f"after {_rupees(money.cost)} of costs")
    )

    if money.is_loss:
        st.warning(
            "At this rate the crop does not cover its own cost of "
            "cultivation. Check the mandi rate against your own before you "
            "commit to it."
        )

    st.caption(
        f"Estimate only — {agronomy.BENCHMARK_BASIS}. Mandi rates move every "
        f"week, and cost of cultivation depends on whether the labour is "
        f"hired or your own. Put your real rate in the box above to correct "
        f"this."
    )

    # ---- Fertiliser bags ------------------------------------------------ #
    advisory = state.get("advisory")
    # nutrient_gaps is a signed "observed - required" in kg/ha, so a deficit
    # is negative. The converter wants a positive requirement; a surplus
    # means buy nothing, not buy a negative amount.
    gaps = dict(getattr(advisory, "nutrient_gaps", None) or {})
    need = {k: max(-float(gaps.get(k, 0.0)), 0.0) for k in ("N", "P", "K")}
    plan = agronomy.bag_plan(
        n_kg_per_hectare=need["N"],
        p_kg_per_hectare=need["P"],
        k_kg_per_hectare=need["K"],
        acres=acres,
    )

    st.markdown(
        f"#### {t_extra('what_to_buy', 'What to buy from the shop') if simple else 'Commercial fertiliser plan'}"
        f" — {acres:g} {_acres_words(acres, simple)}"
    )
    if plan.is_empty:
        st.success(
            "Your soil already has enough of all three. Do not buy fertiliser "
            "for this crop — it would be money wasted."
        )
    else:
        bags = st.columns(3)
        for column, line in zip(bags, plan.lines):
            column.metric(
                f"{line.product}",
                f"{line.whole_bags} {_bags_word(line.whole_bags, simple)}",
            )
            column.caption(f"{line.grade} · {line.kg:.0f} kg")
        if plan.nitrogen_from_dap_kg > 0:
            st.caption(
                f"The {plan.dap.whole_bags} bag(s) of DAP already carry "
                f"{plan.nitrogen_from_dap_kg:.0f} kg of nitrogen, which has "
                f"been taken off the urea above — buying urea for the full "
                f"nitrogen figure would over-fertilise the field."
            )

    # ---- Spray advisory ------------------------------------------------- #
    weather = state.get("weather")
    features = prediction.raw_features
    advice = agronomy.spray_advice(
        rainfall_mm=float(features.get("rainfall", 0.0)),
        humidity_pct=float(features.get("humidity", 0.0)),
        temperature_c=float(features.get("temperature", 0.0)),
        is_live=bool(getattr(weather, "is_live", False)),
    )
    spray_heading = (
        t_extra("can_i_spray", "Can I spray today?") if simple
        else "Today's spray window"
    )
    st.markdown(f"#### {spray_heading}")
    renderer = {
        agronomy.CLEAR: st.success,
        agronomy.CAUTION: st.warning,
        agronomy.HOLD: st.error,
    }[advice.status]
    renderer(f"{advice.icon} **{advice.headline}** — {advice.detail}")

    # ---- Roadmap -------------------------------------------------------- #
    plan_heading = (
        t_extra("season_plan", "Your plan for the season")
        if simple
        else "Crop calendar"
    )
    st.markdown(f"#### {plan_heading}")
    stages = agronomy.roadmap(crop)
    columns = st.columns(len(stages))
    for column, (index, stage) in zip(columns, enumerate(stages, start=1)):
        column.markdown(
            f"<div class='gr-step'><div class='n'>{index}</div>"
            f"<h4>{stage.name}</h4>"
            f"<p><b>{stage.when}</b><br>{stage.action}</p></div>",
            unsafe_allow_html=True,
        )

    _render_share(state, crop_profile, plan, money, simple)


def _render_share(state, crop_profile, plan, money, simple: bool) -> None:
    """Hand the advice to WhatsApp.

    WhatsApp is how this advice actually travels -- to a son in the city, to
    the dealer, to a neighbour. A wa.me link needs no API key and no app
    permission, and degrades to a copyable block of text if the link is not
    tappable on whatever the farmer is using.
    """
    prediction = state["prediction"]
    lines = [
        "GREENROOT",
        "",
        f"Crop: {crop_profile.english} ({agronomy.kannada_name(prediction.crop)})",
        f"Match: {prediction.confidence:.0f} / 100",
    ]
    if state.get("district"):
        lines.append(f"Place: {state['district']}")
    if not plan.is_empty:
        lines.append("")
        lines.append(f"Fertiliser for {plan.acres:g} acre(s):")
        lines += [f"  {line.say()}" for line in plan.lines if line.whole_bags]
    lines += [
        "",
        "Advice only, not a promise. Check with your agriculture officer "
        "before sowing.",
    ]
    message = "\n".join(lines)

    st.markdown(f"#### {t_extra('share_heading', 'Share this advice')}")
    st.link_button(
        t_extra("share_whatsapp", "📤 Send on WhatsApp"),
        f"https://wa.me/?text={quote(message)}",
        width="stretch",
    )
    with st.expander(t_extra("share_copy", "Or copy the text")):
        st.code(message, language=None)




def render_recommendation_tab(state: Dict[str, object]) -> None:
    """Primary crop card, ranked alternatives, advisory, and Z-score profile."""
    simple = is_simple()
    prediction = state.get("prediction")
    if prediction is None:
        render_empty_state(simple)
        return

    advisory = state["advisory"]
    district = state["district"]
    band = band_for(prediction.confidence)

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        headline = (
            f"{band.label} · {score_phrase(prediction.confidence)}"
            if simple
            else f"{prediction.confidence:.2f}% posterior probability"
        )
        # A weak match must not be dressed as a strong one. The card loses its
        # confident green below 60/100 and says so above the crop name, so a
        # farmer skimming on a phone cannot mistake a coin-flip for an answer.
        tone = "" if prediction.confidence >= 60 else " unsure"
        caveat = (
            ""
            if prediction.confidence >= 60
            else f"<div class='warnline'>"
            f"{t_extra('not_clear', 'Not a clear answer') if simple else 'Low posterior'}"
            f"{'' if simple else ' — treat as indicative'}</div>"
        )
        st.markdown(
            f'<div class="gr-primary{tone}">'
            f"{caveat}"
            f'<div class="gr-sub">{tr("primary_label", simple)}</div>'
            f'<div class="crop">{prediction.crop}</div>'
            f'<div class="kn">{agronomy.kannada_name(prediction.crop)}</div>'
            f'<div class="conf">{headline} · {district}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if simple:
            st.caption(band.detail)

        season = season_lib.canonical(state.get("season"))
        fit = season_lib.assess(prediction.crop, season)
        if not fit.suitable:
            st.error(f"📅 **{tr('season_clash', simple)}** — {fit.message(simple, current_language())}")
        elif fit.is_perennial:
            # Not silence: "the season does not apply here" is itself the
            # answer, and leaving it out looks like the check never ran.
            st.info(f"📅 {fit.message(simple, current_language())}")
        else:
            st.success(f"📅 {fit.message(simple, current_language())}")

        if prediction.is_low_confidence:
            st.warning(tr("low_confidence", simple))
        if prediction.is_out_of_distribution:
            names = ", ".join(prediction.ood_features)
            st.warning(
                f"{tr('unusual_input', simple)} ({names})"
                if simple
                else f"Out-of-distribution input: **{names}** exceed "
                f"{OOD_ZSCORE_THRESHOLD:.0f}σ of the training distribution. "
                f"This recommendation is an extrapolation."
            )

        st.markdown(f"#### {tr('ranked_heading', simple)}")
        for candidate in prediction.top_k:
            if simple:
                score = (
                    "less than 1 out of 100"
                    if candidate.confidence_pct < 1
                    else score_phrase(candidate.confidence_pct)
                )
                label = f"**{candidate.rank}. {candidate.crop.capitalize()}** — {score}"
            else:
                label = (
                    f"**{candidate.rank}. {candidate.crop.capitalize()}** — "
                    f"{candidate.confidence_pct:.2f}%"
                )
            candidate_fit = season_lib.assess(candidate.crop, season)
            if not candidate_fit.suitable:
                label += (
                    f"  ·  ⚠️ *{candidate_fit.sowable_labels().lower()} crop*"
                )
            st.markdown(label)
            st.progress(min(max(candidate.probability, 0.0), 1.0))

        st.markdown(f"#### {tr('save_heading', simple)}")
        if st.button(tr("save_button", simple), width="stretch"):
            _persist(state)

    with right:
        st.markdown(f"#### {tr('advisory_heading', simple)}")
        for item in advisory.items:
            st.markdown(
                f"<div class='gr-advisory {item.severity}'>"
                f"<b>{item.icon} {category_name(item.category, simple)}"
                f" · {severity_name(item.severity, simple)}</b><br>"
                f"{item.say(simple)}</div>",
                unsafe_allow_html=True,
            )

        if advisory.fertiliser_plan:
            st.markdown(f"#### {tr('fertiliser_heading', simple)}")
            if simple:
                # Farmers buy 50 kg sacks and work in acres, so lead with that.
                table = pd.DataFrame(
                    {
                        "Fertiliser": list(advisory.fertiliser_plan),
                        "How much to add": [
                            describe_quantity(v)
                            for v in advisory.fertiliser_plan.values()
                        ],
                    }
                )
            else:
                table = pd.DataFrame(
                    {
                        "Product": list(advisory.fertiliser_plan),
                        "Quantity (kg/ha)": list(advisory.fertiliser_plan.values()),
                    }
                )
            st.dataframe(table, hide_index=True, width="stretch")
            _render_worth_it(state, advisory, simple)

    st.markdown("---")
    st.markdown(f"#### {tr('compare_heading', simple)}")
    st.caption(tr("compare_caption", simple))

    frame = prediction.z_score_frame()
    labels = (
        [tr(f"field_{name}", simple) .split(" — ")[0] for name in FEATURE_NAMES]
        if simple
        else FEATURE_LABELS
    )
    figure, axes = plt.subplots(figsize=(7.6, 3.6))
    # Diverging, not status: a high nitrogen reading is neither good nor bad,
    # it is simply above the reference mean.
    axes.barh(
        labels, frame["z_score"],
        color=diverging_colours(frame["z_score"]), height=0.62,
    )
    axes.axvline(0, color=INK, linewidth=0.9)
    for bound in (-OOD_ZSCORE_THRESHOLD, OOD_ZSCORE_THRESHOLD):
        axes.axvline(bound, color=MUTED, linewidth=0.8)
    axes.set_xlabel(
        "More than usual  →" if simple
        else "Z-score (standard deviations from benchmark mean)",
        fontsize=10,
    )
    axes.invert_yaxis()
    _style_axes(axes)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _render_worth_it(
    state: Dict[str, object], advisory, simple: bool
) -> None:
    """Show what the prescription buys, and what it costs to follow.

    Two questions a fertiliser table alone cannot answer: does adding this
    actually improve the match, and how much extra yield must it earn back?
    """
    recommender = state.get("recommender")
    prediction = state.get("prediction")
    if recommender is None or prediction is None or not advisory.fertiliser_plan:
        return

    with st.expander(tr("worth_heading", simple), expanded=False):
        st.caption(tr("worth_intro", simple))

        simulation = simulate_advisory(
            recommender, prediction.raw_features, advisory, before=prediction
        )
        if simulation is None:
            st.info("Nothing to add, so nothing to weigh up.")
            return

        columns = st.columns(3)
        columns[0].metric(
            "Match now" if simple else "Confidence before",
            f"{simulation.before.confidence:.0f}/100"
            if simple
            else f"{simulation.before.confidence:.2f}%",
        )
        columns[1].metric(
            "Match after adding" if simple else "Confidence after",
            f"{simulation.after.confidence:.0f}/100"
            if simple
            else f"{simulation.after.confidence:.2f}%",
            delta=f"{simulation.original_crop_delta:+.0f}"
            if simple
            else f"{simulation.original_crop_delta:+.2f} pp",
        )
        columns[2].metric(
            "Best crop after" if simple else "Argmax after",
            simulation.after.crop.capitalize(),
        )

        st.markdown(f"**{simulation.verdict(simple)}**")

        rows = simulation.comparison_rows()
        if rows:
            st.dataframe(
                pd.DataFrame(rows).rename(
                    columns={
                        "nutrient": "Plant food" if simple else "Nutrient",
                        "before": "Now (kg/ha)",
                        "added": "You add (kg/ha)",
                        "after": "After (kg/ha)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        if simulation.unsimulated:
            st.caption(
                "The lime or gypsum advice above is **not** included in this "
                "comparison. How much a dose moves your pH depends on your "
                "soil's clay and organic matter, which this system does not "
                "measure — so we do not guess at it."
                if simple
                else "Excluded from the simulation: "
                + ", ".join(simulation.unsimulated)
                + ". pH response is buffered by soil properties this system "
                "does not measure."
            )

        st.markdown(f"##### {tr('price_heading', simple)}")
        st.caption(tr("price_hint", simple))

        prices: Dict[str, float] = {}
        price_columns = st.columns(min(len(advisory.fertiliser_plan), 3))
        for index, product in enumerate(advisory.fertiliser_plan):
            column = price_columns[index % len(price_columns)]
            bag_price = column.number_input(
                f"{product} — ₹ per 50 kg bag",
                min_value=0.0,
                step=10.0,
                value=0.0,
                key=f"bagprice_{product}",
                help="Leave at 0 if you do not know it.",
            )
            prices[product] = price_per_kg_from_bag(bag_price)

        estimate = estimate_cost(advisory.fertiliser_plan, prices)
        if not estimate.priced:
            st.info(estimate.summary(simple))
            return

        cost_columns = st.columns(2)
        cost_columns[0].metric(
            "Fertiliser cost" if simple else "Input cost",
            f"₹{estimate.total_per_acre:,.0f} / acre",
        )
        crop_price = cost_columns[1].number_input(
            f"₹ per quintal you expect for {prediction.crop}",
            min_value=0.0,
            step=100.0,
            value=0.0,
            key="crop_price",
            help="Your local mandi rate. Leave at 0 to skip this.",
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Fertiliser": line.product,
                        "Per acre": f"{line.kg_per_acre:.1f} kg "
                                    f"({line.bags_per_acre:.2f} bags)",
                        "Cost": f"₹{line.cost_per_acre:,.0f}",
                    }
                    for line in estimate.lines
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        break_even = estimate.break_even_message(
            prediction.crop, crop_price, simple
        )
        if break_even:
            st.success(break_even)
        else:
            st.caption(
                "Enter the price you expect per quintal to see how much extra "
                "yield this fertiliser has to earn back."
            )


def _persist(state: Dict[str, object]) -> None:
    """Write the current recommendation to the audit ledger."""
    if not ensure_database():
        st.error(
            t_extra("no_store", "Saving is not working on this server right "
                                "now. Your advice is still on screen — take a "
                                "photo of it, or use Share below.")
            if is_simple()
            else "Audit database is unavailable in this environment."
        )
        return

    prediction = state["prediction"]
    consensus = state.get("consensus")
    features = prediction.raw_features
    try:
        signed_in = current_user()
        record_id = db_manager.log_transaction(
            district=str(state["district"]),
            n=features["N"],
            p=features["P"],
            k=features["K"],
            ph=features["ph"],
            temperature=features["temperature"],
            humidity=features["humidity"],
            rainfall=features["rainfall"],
            recommended_crop=prediction.crop,
            confidence=prediction.confidence,
            primary_shap_driver=consensus.primary_driver if consensus else None,
            jaccard_index=consensus.jaccard if consensus else None,
            user_id=signed_in.id if signed_in else None,
        )
    except Exception as exc:  # noqa: BLE001 - surface, never crash the dashboard.
        logging.warning("Could not persist the recommendation: %s", exc)
        st.error(
            t_extra("save_failed", "Could not save it. Your advice is still "
                                   "on screen — take a photo of it, or use "
                                   "Share below.")
            if is_simple()
            else f"Could not persist the recommendation: {exc}"
        )
        return
    st.success(
        t_extra("advice_saved", "Saved. You can find it under Saved & Card.")
        if is_simple()
        else f"Committed to the audit ledger as record #{record_id}."
    )

    # Inside the Android shell, put a copy on the phone itself: the ledger
    # above lives on the server, and the farmer standing in the field is the
    # one who needs to read this back with no signal. A no-op in a browser.
    advisory = state.get("advisory")
    native.push_card(
        st,
        native.build_card(
            card_id=str(record_id),
            crop=prediction.crop,
            confidence=prediction.confidence,
            district=str(state["district"]),
            readings=native.readings_summary(features),
            advice=native.advice_lines(advisory, is_simple()) if advisory else [],
        ),
    )


# --------------------------------------------------------------------------- #
# Tab 2 — Explainable AI Consensus
# --------------------------------------------------------------------------- #
def render_xai_tab(state: Dict[str, object]) -> None:
    """Jaccard consensus card and side-by-side SHAP/LIME attributions."""
    simple = is_simple()
    prediction = state.get("prediction")
    if prediction is None:
        st.info(
            "Get a crop recommendation first, then come here to see why."
            if simple
            else "Generate a recommendation first to audit its explanation."
        )
        return

    st.markdown(f"#### {tr('why_heading', simple)}")
    st.caption(tr("why_intro", simple))

    consensus = state.get("consensus")
    if consensus is None:
        st.warning(
            "The 'why' checks are not available on this computer."
            if simple
            else "The explainability stack is unavailable. Install it with "
            "`pip install shap lime` and restart."
        )
        return

    badge = "ok" if consensus.is_high_fidelity else "warn"
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        verdict = tr("agree_yes" if consensus.is_high_fidelity else "agree_no", simple)
        if simple:
            score = f"{consensus.intersection} of {consensus.union_size}"
            sub_line = "reasons both checks agree on"
        else:
            score = f"{consensus.jaccard:.2f}"
            sub_line = (
                f"{consensus.intersection} of {consensus.union_size} drivers shared"
            )
        st.markdown(
            f"""<div class="gr-card">
                  <div class="gr-sub">{tr('agreement_label', simple)}</div>
                  <div class="gr-metric">{score}</div>
                  <div class="gr-sub">{sub_line}</div>
                  <div style="margin-top:12px">
                    <span class="gr-badge {badge}">{verdict}</span>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
        if simple:
            st.markdown(
                "**What this means.** "
                + (
                    "Both ways of checking picked the same main reasons, so "
                    "this advice rests on solid ground."
                    if consensus.is_high_fidelity
                    else "The two checks picked different reasons. Your land "
                    "sits close to the line between two crops. Treat this "
                    "advice as a starting point and get a soil test."
                )
            )
        else:
            st.latex(
                r"J(S, L) = \frac{|S \cap L|}{|S \cup L|} = "
                rf"\frac{{{consensus.intersection}}}{{{consensus.union_size}}} = "
                rf"{consensus.jaccard:.2f}"
            )
            st.caption(
                f"S and L are the top-{consensus.top_k} driver sets from "
                f"TreeSHAP and LIME. J ≥ {JACCARD_FIDELITY_THRESHOLD:.1f} is "
                f"reported as High Fidelity."
            )
            st.markdown(f"**Interpretation.** {consensus.interpretation()}")

        if not consensus.surrogate_agrees:
            st.warning(
                "The two checks were run on a slightly simpler model that "
                "picked a different crop here, so read the reasons below with "
                "care."
                if simple
                else "The interpretable surrogate assigns this instance a "
                "different class than the deployed ensemble, so the "
                "attribution below transfers only partially."
            )

    with right:
        def driver_names(names) -> str:
            return ", ".join(_feature_word(n, simple) for n in names) or "—"

        metrics = st.columns(3)
        for column, heading, names in (
            (metrics[0], f"{tr('check_one', simple)} says", consensus.shap_top_k),
            (metrics[1], f"{tr('check_two', simple)} says", consensus.lime_top_k),
            (
                metrics[2],
                "Both agree on" if simple else "Consensus drivers",
                consensus.consensus_drivers,
            ),
        ):
            agreed = names is consensus.consensus_drivers
            column.markdown(
                f"<div class='gr-drivercard{' agree' if agreed else ''}'>"
                f"<div class='k'>{heading}</div>"
                f"<div class='v'>{driver_names(names)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    with st.container():
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
        plain_names = {
            name: _feature_word(name, simple) for name in FEATURE_NAMES
        }
        for index, (title, values, subtitle) in enumerate(
            (
                (
                    tr("check_one", simple),
                    consensus.shap_values,
                    "How much each thing pushed the answer"
                    if simple
                    else "Exact Shapley attribution",
                ),
                (
                    tr("check_two", simple),
                    consensus.lime_values,
                    "A second opinion, worked out differently"
                    if simple
                    else "Local surrogate coefficients",
                ),
            )
        ):
            ordered = sorted(values.items(), key=lambda kv: abs(kv[1]))
            names = [plain_names.get(name, name) for name, _ in ordered]
            weights = [weight for _, weight in ordered]
            axes[index].barh(
                names, weights, color=diverging_colours(weights), height=0.6
            )
            axes[index].axvline(0, color=INK, linewidth=0.9)
            axes[index].set_title(f"{title}\n{subtitle}", fontsize=10.5)
            axes[index].set_xlabel(
                "How strongly it pushed  →" if simple else "Attribution",
                fontsize=10.5,
            )
            _style_axes(axes[index])
        figure.suptitle(
            f"Why {consensus.predicted_crop.title()} was chosen"
            if simple
            else "Local attributions for the "
            f"{consensus.predicted_crop.title()} decision",
            fontsize=11.5,
            x=0.01,
            ha="left",
        )
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        st.caption(
            "The two checks use different scales, so compare the *order* of "
            "the bars, not their length."
            if simple
            else "Attribution scales differ between the two methods — only "
            "the *rankings* are compared, which is precisely what the Jaccard "
            "index measures."
        )

    with st.expander("See the numbers" if simple else "Attribution detail"):
        st.dataframe(
            consensus.to_frame().round(5), hide_index=True, width="stretch"
        )


# --------------------------------------------------------------------------- #
# Tab 3 — What-If Sensitivity Engine
# --------------------------------------------------------------------------- #
def render_sensitivity_tab(state: Dict[str, object]) -> None:
    """Trace the decision surface along a single perturbed feature axis."""
    simple = is_simple()
    prediction = state.get("prediction")
    recommender = state.get("recommender")
    if prediction is None or recommender is None:
        st.info(
            "Get a crop recommendation first, then try changing things here."
            if simple
            else "Generate a recommendation first to run a sensitivity sweep."
        )
        return

    st.markdown(f"#### {tr('whatif_heading', simple)}")
    st.caption(tr("whatif_intro", simple))

    controls = st.columns([1.4, 1, 1])
    feature = controls[0].selectbox(
        tr("whatif_feature", simple),
        options=FEATURE_NAMES,
        format_func=lambda name: (
            tr(f"field_{name}", simple).split(" — ")[0]
            if simple
            else FEATURE_LABELS[FEATURE_NAMES.index(name)]
        ),
    )
    span = controls[1].slider(
        "How far to change it (±%)" if simple else "Sweep range (±%)",
        10, 100, 100, step=10,
    )
    # Capped at the validated categorical slot count; past this the guidance
    # is to fold or facet, never to generate a ninth hue.
    n_curves = controls[2].slider(
        "How many crops to show" if simple else "Crops to plot", 2, 6, 4
    )

    anchor = [prediction.raw_features[name] for name in FEATURE_NAMES]
    current = float(prediction.raw_features[feature])
    low, high = FEATURE_BOUNDS[feature]

    # A ±100% band around zero collapses to a point, so fall back to the full
    # admissible range for features whose current reading is effectively zero.
    if abs(current) < 1e-6:
        sweep_low, sweep_high = low, high
    else:
        sweep_low = current * (1 - span / 100.0)
        sweep_high = current * (1 + span / 100.0)
    grid = np.linspace(max(sweep_low, low), min(sweep_high, high), 80)

    try:
        grid, probabilities = recommender.sweep(anchor, feature, grid)
    except ValidationError as exc:
        st.error(f"Sweep failed: {exc}")
        return

    classes = recommender.class_names
    # Rank by peak posterior across the sweep so crops that only become viable
    # at one end of the range still appear.
    ranked = np.argsort(probabilities.max(axis=0))[::-1][:n_curves]

    figure, axes = plt.subplots(figsize=(8.8, 3.8))
    # Crop identity is categorical: fixed slot order, never a generated or
    # cycled hue, and never a value ramp (viridis would double-encode rank).
    palette = series_palette(len(ranked))
    for colour, index in zip(palette, ranked):
        crop = classes[int(index)]
        # Emphasis: the current recommendation is the story, so it carries
        # weight while the alternatives stay thin.
        recommended = crop == prediction.crop
        axes.plot(
            grid,
            probabilities[:, index],
            linewidth=2.8 if recommended else 1.8,
            color=colour,
            label=f"{crop} (now)" if recommended else crop,
            zorder=3 if recommended else 2,
        )
    axes.axvline(
        current,
        color=MUTED,
        linewidth=1.4,
        label=f"your reading: {current:.0f}",
        zorder=1,
    )
    index = FEATURE_NAMES.index(feature)
    axes.set_xlabel(f"{FEATURE_LABELS[index]} ({FEATURE_UNITS[index]})", fontsize=11)
    axes.set_ylabel(
        "Chance this crop suits" if simple else "Posterior probability", fontsize=10
    )
    axes.set_ylim(-0.02, 1.02)
    # A legend is always present for >= 2 series, so identity is never
    # carried by colour alone. It goes ABOVE the axes: matplotlib's default
    # loc="best" scores candidate corners for emptiness, and on a sweep where
    # the curves span the full 0-1 range every corner is occupied, so it lands
    # on top of the data.
    axes.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(ranked) + 1, 4),
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.4,
    )
    _style_axes(axes, grid_axis="y")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    # Where does the argmax flip? Those crossings are the actionable thresholds.
    argmax = probabilities.argmax(axis=1)
    switches = [
        (float(grid[i]), classes[int(argmax[i - 1])], classes[int(argmax[i])])
        for i in range(1, len(argmax))
        if argmax[i] != argmax[i - 1]
    ]
    summary = st.columns(3)
    summary[0].metric(
        f"Your {feature} now" if simple else f"Current {feature}", f"{current:.1f}"
    )
    summary[1].metric(
        "Range tried" if simple else "Sweep range",
        f"{grid.min():.1f} – {grid.max():.1f}",
    )
    summary[2].metric(
        "Times the best crop changes" if simple else "Decision boundaries crossed",
        str(len(switches)),
    )

    if switches:
        st.markdown(f"**{tr('switch_heading', simple)}**")
        columns = (
            [f"If {feature} reaches", "Best crop changes from", "to"]
            if simple
            else [f"{feature} threshold", "From crop", "To crop"]
        )
        st.dataframe(
            pd.DataFrame(switches, columns=columns).round(2),
            hide_index=True,
            width="stretch",
        )
    else:
        st.success(
            f"Changing {feature} does not change the answer — {prediction.crop} "
            f"stays the best crop across the whole range. Something else is "
            f"deciding it."
            if simple
            else f"The {prediction.crop} recommendation is stable across the "
            f"entire ±{span}% sweep of {feature} — this factor is not the "
            f"binding constraint for this plot."
        )



# --------------------------------------------------------------------------- #
# Tab 4 — Bulk Advisory
# --------------------------------------------------------------------------- #
def render_bulk_tab() -> None:
    """Recommendations for a whole survey in one pass.

    An extension officer serves a village, not one cultivator. This tab takes a
    laboratory CSV export and returns a recommendation per sample, with the
    rows needing human review surfaced first.
    """
    simple = is_simple()
    st.markdown(f"#### {tr('bulk_heading', simple)}")
    st.caption(tr("bulk_intro", simple))

    processor = load_batch_processor()
    if processor is None:
        st.error("Model artefacts unavailable; bulk advisory is disabled.")
        return

    template = build_template()
    left, right = st.columns([2, 1], gap="large")
    with left:
        upload = st.file_uploader(
            "Soil survey file",
            type=["csv", "xlsx"],
            help=(
                f"CSV or Excel, up to {MAX_BATCH_ROWS:,} rows. Required "
                f"columns: {', '.join(FEATURE_NAMES)}."
            ),
        )
    with right:
        st.markdown("**No file to hand?**")
        st.download_button(
            "⬇️ Download template",
            data=template.to_csv(index=False),
            file_name="greenroot_bulk_template.csv",
            mime="text/csv",
            width="stretch",
        )
        if st.button("▶️ Run the template", width="stretch"):
            st.session_state["bulk_source"] = template

    if upload is not None:
        try:
            if upload.name.lower().endswith(".xlsx"):
                try:
                    st.session_state["bulk_source"] = pd.read_excel(upload)
                except ImportError:
                    # Reading .xlsx needs an optional engine; say so plainly
                    # rather than surfacing pandas' internal message.
                    st.error(
                        "Reading Excel files needs the `openpyxl` package. "
                        "Install it with `pip install openpyxl`, or save the "
                        "sheet as CSV and upload that instead."
                    )
                    return
            else:
                st.session_state["bulk_source"] = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001 - surface any parse failure.
            st.error(f"Could not read the file: {exc}")
            return

    source = st.session_state.get("bulk_source")
    if source is None:
        with st.expander("Expected file format"):
            st.dataframe(template, hide_index=True, width="stretch")
            st.caption(
                "Optional columns `sample_id`, `district`, `village` and "
                "`farmer` are carried through to the output for traceability."
            )
        return

    st.markdown(f"**Loaded {len(source):,} row(s)** · {len(source.columns)} columns")
    with st.expander("Preview the uploaded data"):
        st.dataframe(source.head(20), hide_index=True, width="stretch")

    try:
        with st.spinner(f"Scoring {len(source):,} samples…"):
            result = processor.process(source)
    except ValueError as exc:
        st.error(f"**Cannot process this file.** {exc}")
        return

    _render_bulk_result(result)


def _render_bulk_result(result: BatchResult) -> None:
    """Render the summary, review queue, and exports for a bulk run."""
    summary = result.summary()

    metrics = st.columns(5)
    metrics[0].metric("Processed", f"{summary['processed']:,}")
    metrics[1].metric("Rejected", f"{summary['rejected']:,}")
    metrics[2].metric("Distinct crops", summary["distinct_crops"])
    metrics[3].metric("Mean confidence", f"{summary['mean_confidence']:.1f}%")
    metrics[4].metric(
        tr("needs_review", is_simple()),
        f"{summary['low_confidence'] + summary['out_of_distribution']:,}",
        help="Samples we are unsure about, or with unusual readings.",
    )

    with st.expander("How your column headers were interpreted"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Model feature": list(result.resolved_columns),
                    "Your column": list(result.resolved_columns.values()),
                }
            ),
            hide_index=True,
            width="stretch",
        )

    if result.is_empty:
        st.warning("No row passed validation. See the rejected rows below.")
    else:
        flagged = result.flagged()
        if not flagged.empty:
            st.warning(
                f"**{len(flagged)} sample(s) need human review** — low "
                f"confidence or inputs beyond the training distribution. A "
                f"bulk run is exactly where a quietly extrapolated "
                f"recommendation would otherwise pass unnoticed."
            )
            with st.expander("Review queue", expanded=True):
                st.dataframe(flagged, hide_index=True, width="stretch")

        st.markdown("#### Recommendations")
        st.dataframe(result.recommendations, hide_index=True, width="stretch", height=360)

        distribution = result.crop_distribution()
        chart, table = st.columns([1.6, 1], gap="large")
        with chart:
            figure, axes = plt.subplots(
                figsize=(8, max(2.6, 0.36 * len(distribution)))
            )
            axes.barh(
                distribution["crop"][::-1],
                distribution["count"][::-1],
                color=ACCENT,
                height=0.62,
            )
            axes.set_xlabel("Samples recommended", fontsize=11)
            axes.set_title("Recommended crop distribution across the survey", fontsize=11)
            _style_axes(axes)
            figure.tight_layout()
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        with table:
            display = distribution.copy()
            display["share"] = (display["share"] * 100).round(1).astype(str) + "%"
            display["mean_confidence"] = display["mean_confidence"].round(1)
            st.dataframe(display, hide_index=True, width="stretch")

    if not result.rejected.empty:
        st.markdown("#### Rejected rows")
        st.caption(
            "Each row is reported with its 1-based position in your file and "
            "the reason it was inadmissible, so the source data can be "
            "corrected."
        )
        st.dataframe(result.rejected, hide_index=True, width="stretch")

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    downloads = st.columns(2)
    if not result.is_empty:
        downloads[0].download_button(
            "⬇️ Recommendations (CSV)",
            data=result.recommendations.to_csv(index=False),
            file_name=f"greenroot_bulk_recommendations_{stamp}.csv",
            mime="text/csv",
            width="stretch",
        )
    if not result.rejected.empty:
        downloads[1].download_button(
            "⬇️ Rejected rows (CSV)",
            data=result.rejected.to_csv(index=False),
            file_name=f"greenroot_bulk_rejected_{stamp}.csv",
            mime="text/csv",
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Tab 5 — Audit Trail & Governance
# --------------------------------------------------------------------------- #
def render_audit_tab() -> None:
    """Filterable view over the persisted recommendation ledger."""
    simple = is_simple()
    st.markdown(f"#### {tr('records_heading', simple)}")
    st.caption(
        t_extra("records_blurb",
                "Every piece of advice you saved is kept here, with the "
                "readings it was based on.")
        if simple
        else "Every committed recommendation is recorded with its inputs, its "
        "confidence, its dominant SHAP driver, and its explainer agreement "
        "index — the evidence trail behind advice acted on in the field."
    )

    if not ensure_database():
        st.error(
            t_extra("no_store", "Saving is not working on this server right "
                                "now. Your advice is still on screen — take a "
                                "photo of it, or use Share below.")
            if is_simple()
            else "Audit database is unavailable in this environment."
        )
        return

    # Folded away by default: stacked on a phone these three controls fill
    # the screen before a single saved record is visible, and the default
    # window -- the last 30 days -- is the one most people want.
    today = date.today()
    with st.expander(
        t_extra("change_dates", "Change the dates") if simple
        else "Filter the ledger",
        expanded=False,
    ):
        controls = st.columns(3)
        start = controls[0].date_input("From", value=today - timedelta(days=30))
        end = controls[1].date_input("To", value=today)
        limit = controls[2].number_input(
            "How many to show" if simple else "Max records", 10, 5000, 200, step=10
        )

    # A farmer sees their own records; a guest sees the guest ledger on this
    # server. Examiner mode is deliberately unscoped -- it is the audit view.
    viewer = current_user()
    scope = (
        {"user_id": viewer.id}
        if viewer is not None and simple
        else {"guest_only": True}
        if simple
        else {}
    )
    frame = db_manager.fetch_audit_history(
        **scope,
        limit=int(limit),
        start_date=str(start),
        end_date=str(end),
    )

    if frame.empty:
        st.info(
            t_extra("no_records",
                    "Nothing saved in these dates. Press the green button at "
                    "the top to get advice, then save it — it will show up "
                    "here.")
            if simple
            else "No records in this window. Commit a recommendation from the "
            "**Precision Recommendation** tab to populate the ledger."
        )
        return

    total = db_manager.count_records()
    if simple:
        summary = st.columns(3)
        summary[0].metric(
            t_extra("advice_saved_count", "Advice saved"),
            f"{len(frame):,}",
            # Only worth saying when the date window is hiding something,
            # and as a caption: metric deltas always draw a direction arrow.
            help=f"of {total:,} saved in all" if total != len(frame) else None,
        )
        summary[1].metric(
            t_extra("average_match", "Average match"),
            f"{frame['confidence'].mean():.0f} / 100",
        )
        summary[2].metric(
            t_extra("different_crops", "Different crops"),
            f"{frame['recommended_crop'].nunique()}",
        )
    else:
        summary = st.columns(4)
        summary[0].metric(
            "Records shown",
            f"{len(frame):,}",
            help=f"of {total:,} total" if total != len(frame) else None,
        )
        summary[1].metric("Mean confidence", f"{frame['confidence'].mean():.1f}%")
        summary[2].metric("Distinct crops", f"{frame['recommended_crop'].nunique()}")
        jaccard = pd.to_numeric(frame["jaccard_index"], errors="coerce").dropna()
        summary[3].metric(
            "Mean Jaccard", f"{jaccard.mean():.2f}" if not jaccard.empty else "—"
        )

    st.dataframe(
        db.farmer_view(frame, current_language()) if simple
        else db.examiner_view(frame),
        hide_index=True,
        width="stretch",
        height=db.table_height(len(frame)),
    )

    # Reload a past reading into the form. An officer revisiting a plot should
    # not have to retype seven numbers off a printout.
    reload_columns = st.columns([2, 1])
    options = {
        (
            f"#{int(row['id'])} · {row['district']} · "
            f"{_crop_label(str(row['recommended_crop']), simple)} · "
            f"{db.record_date(row['timestamp'], simple, current_language())}"
        ): row
        for _, row in frame.iterrows()
    }
    chosen = reload_columns[0].selectbox(
        t_extra("open_saved", "Open a saved reading") if simple
        else "Reload a record into the form",
        options=list(options),
        index=0,
        key="reload_pick",
    )
    if reload_columns[1].button(
        t_extra("load_readings", "↩️ Load these readings"),
        width="stretch", key="reload_go",
    ):
        row = options[chosen]
        for feature, column in (
            ("N", "N"), ("P", "P"), ("K", "K"), ("ph", "pH"),
            ("temperature", "temp"), ("humidity", "humidity"),
            ("rainfall", "rainfall"),
        ):
            low, high = FEATURE_BOUNDS[feature]
            st.session_state[f"in_{feature}"] = float(
                np.clip(float(row[column]), low, high)
            )
        st.success(
            f"Loaded the readings from record #{int(row['id'])}. "
            f"Press the green button at the top to run them again."
        )

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    st.download_button(
        t_extra("download_all", "⬇️ Download all of this (CSV)") if simple
        else "⬇️ Download audit trail (CSV)",
        data=buffer.getvalue(),
        file_name=f"greenroot_audit_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander(
        t_extra("which_crops", "Which crops came up most") if simple
        else "Distribution by recommended crop"
    ):
        counts = frame["recommended_crop"].value_counts()
        figure, axes = plt.subplots(figsize=(9, max(2.4, 0.34 * len(counts))))
        axes.barh(counts.index[::-1], counts.to_numpy()[::-1], color=ACCENT, height=0.6)
        axes.set_xlabel("Recommendations logged", fontsize=11)
        _style_axes(axes)
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)


# --------------------------------------------------------------------------- #
# Tab 6 — Farmer Soil Health Card
# --------------------------------------------------------------------------- #
def render_card_tab(state: Dict[str, object]) -> None:
    """Printable soil health card in three export formats."""
    prediction = state.get("prediction")
    if prediction is None:
        st.info("Generate a recommendation first to issue a soil health card.")
        return

    simple = is_simple()
    reads_kannada = simple and current_language() == i18n.KANNADA
    st.markdown(f"#### {tr('card_heading', simple)}")
    st.caption(
        t_extra("card_print", "Print this and keep it, or show it to your "
                              "agriculture officer.")
        if simple
        else "The card an extension officer hands to the cultivator. Every "
        "export renders from one payload, so the figures cannot diverge "
        "between formats."
    )

    controls = st.columns([1.4, 2])
    bilingual = controls[0].toggle(
        "ಕನ್ನಡ · Bilingual card",
        # Somebody who has already told us they read Kannada should not have
        # to find a toggle to get a card they can read.
        value=reads_kannada,
        help="Show Kannada alongside English on the farmer-facing card.",
    )
    if bilingual:
        controls[1].caption(
            t_extra(
                "card_bilingual_note",
                "English is retained beside every Kannada term, so a "
                "translation error cannot silently change the advice. "
                "Translations are a prototype mapping and need native-speaker "
                "review before field use.",
            )
        )

    card = dataclasses.replace(
        build_card(
            district=str(state["district"]),
            prediction=prediction,
            advisory=state["advisory"],
            consensus=state.get("consensus"),
        ),
        bilingual=bilingual,
    )

    st.iframe(_as_data_uri(render_html(card)), height=900)

    stamp = f"{datetime.now():%Y%m%d_%H%M}"

    try:
        pdf_bytes = render_pdf(card)
    except PDFUnavailableError:
        pdf_bytes = None

    # Said before the choice, not after it. The PDF is the obvious button to
    # press and it is the one export that cannot carry Kannada, so somebody
    # reading a bilingual card needs to know that while they are choosing.
    if bilingual and pdf_bytes is not None:
        st.caption(
            t_extra(
                "card_pdf_english_only",
                "The PDF is English-only: its built-in fonts cannot render "
                "Kannada. Download the HTML for a printout in both languages.",
            )
        )

    downloads = st.columns(4)

    if pdf_bytes is not None:
        downloads[0].download_button(
            "⬇️ PDF (print)",
            data=pdf_bytes,
            file_name=f"soil_health_card_{stamp}.pdf",
            mime="application/pdf",
            # Whichever export the reader can actually read leads.
            type="secondary" if bilingual else "primary",
            width="stretch",
        )
    else:
        downloads[0].button(
            "PDF unavailable", disabled=True, width="stretch",
            help="Install the optional dependency: pip install fpdf2",
        )

    downloads[1].download_button(
        "⬇️ HTML",
        data=render_html(card),
        file_name=f"soil_health_card_{stamp}.html",
        mime="text/html",
        type="primary" if bilingual else "secondary",
        width="stretch",
    )
    downloads[2].download_button(
        "⬇️ Markdown",
        data=render_markdown(card),
        file_name=f"soil_health_card_{stamp}.md",
        mime="text/markdown",
        width="stretch",
    )
    downloads[3].download_button(
        "⬇️ Plain text",
        data=render_text(card),
        file_name=f"soil_health_card_{stamp}.txt",
        mime="text/plain",
        width="stretch",
    )

    with st.expander("Plain-text preview (SMS / thermal printer)"):
        st.code(render_text(card), language=None)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    """Assemble the dashboard."""
    # Makes "Add to Home screen" produce a full-screen app with its own icon.
    # Purely cosmetic — if it fails the dashboard is unaffected.
    install_pwa(st)

    recommender = load_recommender()
    if recommender is None:
        st.stop()

    st.sidebar.markdown("### GREENROOT")

    # Both of these render nothing; they run here for their order alone.
    # ensure_database() must precede current_user(), which reads the sessions
    # table, and _adopt_stored_language() writes the ui_language key -- which
    # Streamlit forbids once the radio bound to that key exists. Called after
    # the picker, as it used to be, every sign-in and every account creation
    # ended in StreamlitWidgetAlreadyInstantiatedError and a page of red.
    ensure_database()
    _adopt_stored_language()

    # Above the account panel and outside it: a guest must be able to change
    # language without signing up.
    render_language_picker()

    st.sidebar.toggle(
        "👨‍🏫 Examiner / AI Mode",
        value=False,
        key="examiner_mode",
        help="Off: the farmer portal. On: the model-audit deck — probability "
             "distribution, Z-scores, SHAP/LIME consensus, sensitivity "
             "curves and the raw ledger.",
    )
    st.sidebar.caption(
        "Examiner mode shows the evidence behind the recommendation: "
        "explainer agreement, calibration and the audit trail."
        if not is_simple()
        else "Turn this on to see how the model reached its answer."
    )

    render_account_sidebar(is_simple())
    render_help_sidebar()
    inputs = render_controls()
    simple = is_simple()

    tagline = (
        "Tells you which crop suits your land, and why"
        if simple
        else f"Stacking ensemble meta-learning · multi-explainer consensus "
        f"auditing · {REPORTED_CV_ACCURACY * 100:.2f}% stratified 5-fold "
        f"cross-validated accuracy across 22 crop classes"
    )
    features: Dict[str, float] = inputs["features"]  # type: ignore[assignment]
    # The readings live in the hero. Before a run there is nothing else to
    # show, and a banner holding only a title leaves a blank slab above the
    # fold; this also gives the numbers room to be legible at arm's length.
    reads = [
        ("Nitrogen", f"{features['N']:.0f}", "kg/ha"),
        ("Phosphorus", f"{features['P']:.0f}", "kg/ha"),
        ("Potassium", f"{features['K']:.0f}", "kg/ha"),
        ("Soil pH", f"{features['ph']:.1f}", ""),
        ("Temperature", f"{features['temperature']:.0f}", "°C"),
        ("Rainfall", f"{features['rainfall']:.0f}", "mm"),
    ]
    tiles = "".join(
        f"<div class='gr-read'><div class='k'>{label}</div>"
        f"<div class='v'>{value}<small>{unit}</small></div></div>"
        for label, value, unit in reads
    )

    # The hero and the hint below it both depend on whether there is an
    # answer yet -- but the answer is computed further down, and st.tabs
    # renders every tab in a single pass and switches between them on the
    # client, so no rerun happens when the farmer changes tab. Reading
    # session_state here would leave the banner a full run behind. Reserve
    # the slots now and fill them once the prediction is known.
    hero_slot = st.empty()
    bar_left, bar_right = st.columns([2.2, 1], gap="medium")
    hint_slot = bar_left.empty()
    with bar_right:
        main_run = st.button(
            tr("run", simple), type="primary", width="stretch", key="run_main"
        )

    ensure_database()

    if main_run:
        vector = [features[name] for name in FEATURE_NAMES]
        try:
            prediction = recommender.predict(vector, top_k=3)
        except ValidationError as exc:
            st.error(f"**Invalid input.** {exc}")
            st.stop()

        advisory = generate_advisory(
            prediction.crop,
            prediction.raw_features,
            confidence=prediction.confidence,
            ood_features=prediction.ood_features,
        )

        consensus = None
        explainer = load_explainer()
        if explainer is not None:
            spinner = (
                t_extra("checking_answer", "Checking the answer…")
                if is_simple()
                else "Auditing the decision with TreeSHAP and LIME…"
            )
            with st.spinner(spinner):
                try:
                    consensus = explainer.explain(vector, prediction.crop)
                except Exception as exc:  # noqa: BLE001 - XAI must never block.
                    logging.warning("Explanation failed: %s", exc)

        st.session_state["prediction"] = prediction
        st.session_state["advisory"] = advisory
        st.session_state["consensus"] = consensus
        st.session_state["district"] = inputs["district"]

    state: Dict[str, object] = {
        "prediction": st.session_state.get("prediction"),
        "advisory": st.session_state.get("advisory"),
        "consensus": st.session_state.get("consensus"),
        "district": st.session_state.get("district", inputs["district"]),
        "season": inputs["season"],
        "recommender": recommender,
        "acres": inputs["acres"],
        "weather": st.session_state.get("weather"),
        "price_override": st.session_state.get("price_override"),
    }

    # Both slots are filled here, after the run handler: this is the first
    # point in the script where "is there an answer" is finally true or false.
    answered = st.session_state.get("prediction")
    if answered is None:
        hero_slot.markdown(
            f"""<div class="gr-hero">
                  <div class="gr-hero-top">
                    <h1>🌱 GREENROOT</h1>
                    <span class="gr-hero-place">{inputs['district']}</span>
                  </div>
                  <p>{tagline}</p>
                  <div class="gr-reads">{tiles}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        hint_slot.markdown(
            "<div class='gr-readout-hint'>"
            + (
                t_extra(
                    "these_are_readings",
                    "These are your land's readings. Change your district "
                    "above if they are wrong, then press the green button.",
                )
                if simple
                else "Set the district above, or override features in the "
                "manual expander."
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        band = band_for(answered.confidence)
        score = (
            f"{band.label} \u00b7 {score_phrase(answered.confidence)}"
            if simple
            else f"{answered.confidence:.2f}% posterior"
        )
        # Collapsed: at full height this banner repeats on all six tabs and,
        # on a phone, pushes every tab's content below the fold.
        hero_tone = "" if answered.confidence >= 60 else " gr-hero-unsure"
        hero_slot.markdown(
            f"""<div class="gr-hero gr-hero-compact{hero_tone}">
                  <div class="gr-hero-top">
                    <h1>🌱 GREENROOT</h1>
                    <span class="gr-hero-place">{inputs['district']}</span>
                  </div>
                  <div class="gr-hero-answer">
                    <span class="crop">{answered.crop}</span>
                    <span class="score">{score}</span>
                  </div>
                  <div class="gr-hero-reads">{
                      native.readings_summary(features)}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        hint_slot.markdown(
            "<div class='gr-readout-hint'>"
            + (
                t_extra(
                    "changed_something",
                    "Changed something? Edit your readings at the top, then "
                    "press the button again.",
                )
                if simple
                else "Change the inputs above and re-run to refresh every "
                "tab."
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # Two audiences, two decks. The farmer portal leads with what to do;
    # the examiner deck leads with why to believe it. Everything is still
    # reachable in both -- the toggle changes the order and the wording, not
    # what the system is willing to show.
    if simple:
        # Two tabs, not five. A farmer wants one answer -- which crop, what
        # to buy, when to do it -- and splitting that across three screens
        # made them hunt for the half they needed. The explainer-consensus
        # screen is gone from this view entirely: Jaccard agreement between
        # TreeSHAP and LIME is an examiner's question, not a farmer's. It is
        # still computed, and still one toggle away in Examiner mode.
        tabs = st.tabs([t_extra("your_advice", "🌱 Your Advice"),
                        t_extra("saved_and_card", "🗂️ Saved & Card")])
        with tabs[0]:
            render_recommendation_tab(state)
            if state.get("prediction") is not None:
                st.markdown("---")
                render_commercial_panel(state, simple)
        with tabs[1]:
            render_reminders(simple)
            render_card_tab(state)
            st.markdown("---")
            render_audit_tab()
    else:
        tabs = st.tabs(
            [
                "🎯 Prediction & Z-scores",
                "🔍 XAI Consensus",
                "🧭 What-If Sensitivity",
                "💰 Commercial Model",
                "📦 Bulk Advisory",
                "📋 Audit Trail",
                "🧾 Soil Health Card",
            ]
        )
        with tabs[0]:
            render_recommendation_tab(state)
        with tabs[1]:
            render_xai_tab(state)
        with tabs[2]:
            render_sensitivity_tab(state)
        with tabs[3]:
            if state.get("prediction") is None:
                st.info("Generate a recommendation to price it.")
            else:
                render_commercial_panel(state, simple)
        with tabs[4]:
            render_bulk_tab()
        with tabs[5]:
            render_audit_tab()
        with tabs[6]:
            render_card_tab(state)

    st.markdown("---")
    st.caption(
        tr("disclaimer", simple)
        if simple
        else f"GREENROOT · stacking ensemble (Random Forest + AdaBoost + k-NN "
        f"→ logistic-regression meta-learner) · "
        f"{len(recommender.class_names)} crop classes · Jaccard consensus at "
        f"k={CONSENSUS_TOP_K}. Advisory output only — corroborate with a "
        f"certified laboratory soil test before committing a season."
    )


if __name__ == "__main__":
    main()
