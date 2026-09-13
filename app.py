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
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import (
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
from src.database import db_manager
from src.models.batch import BatchProcessor, BatchResult, MAX_BATCH_ROWS, build_template
from src.models.inference import CropRecommender, ValidationError
from src.models.xai_engine import ExplainerConsensus
from src.services.soil_service import get_district_baseline, list_districts
from src.services.weather_service import get_weather
from src.utils.agronomy_advisory import CRITICAL, INFO, WARNING, generate_advisory
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
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Presentation constants
# --------------------------------------------------------------------------- #
ACCENT = "#1f7a4d"
ACCENT_SOFT = "#e7f2eb"
POSITIVE = "#2f9e5f"
NEGATIVE = "#c0563f"
NEUTRAL = "#8a9a90"

_CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  .gr-hero { background: linear-gradient(135deg, #1f7a4d 0%, #2f9e5f 100%);
             color: #fff; padding: 20px 26px; border-radius: 12px;
             margin-bottom: 18px; }
  .gr-hero h1 { margin: 0 0 4px; font-size: 25px; letter-spacing: .2px; }
  .gr-hero p  { margin: 0; opacity: .92; font-size: 13.5px; }
  .gr-card { border: 1px solid #d9e3dc; border-radius: 12px; padding: 18px 20px;
             background: #fff; }
  .gr-primary { background: #e7f2eb; border: 1px solid #b9d8c6;
                border-radius: 12px; padding: 20px 24px; }
  .gr-primary .crop { font-size: 34px; font-weight: 700; color: #1f7a4d;
                      text-transform: uppercase; letter-spacing: .6px;
                      line-height: 1.15; }
  .gr-primary .conf { font-size: 14px; color: #5c6f63; margin-top: 2px; }
  .gr-badge { display: inline-block; padding: 4px 12px; border-radius: 999px;
              font-size: 12px; font-weight: 600; letter-spacing: .3px; }
  .gr-badge.ok   { background: #e7f2eb; color: #1f7a4d; border: 1px solid #b9d8c6; }
  .gr-badge.warn { background: #fdf3f2; color: #a6382a; border: 1px solid #eec4bd; }
  .gr-metric { font-size: 40px; font-weight: 700; color: #1f7a4d;
               line-height: 1.1; }
  .gr-sub { font-size: 12.5px; color: #5c6f63; }
  .gr-advisory { border-left: 3px solid #d9e3dc; padding: 9px 14px;
                 margin-bottom: 9px; border-radius: 0 8px 8px 0; font-size: 13.5px; }
  .gr-advisory.critical { border-left-color: #c0392b; background: #fdf3f2; }
  .gr-advisory.warning  { border-left-color: #d98b0e; background: #fdf8ee; }
  .gr-advisory.info     { border-left-color: #1f7a4d; background: #f4f8f5; }
  .gr-advisory b { color: #14281d; }
  div[data-testid="stMetricValue"] { font-size: 23px; }
  div[data-testid="stProgress"] > div > div > div > div { background-color: #1f7a4d; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

_SEVERITY_LABEL = {CRITICAL: "Critical", WARNING: "Advisory", INFO: "Nominal"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading stacking ensemble…")
def load_recommender() -> Optional[CropRecommender]:
    """Load the deployed ensemble once per server process."""
    try:
        return CropRecommender().load()
    except FileNotFoundError as exc:
        st.error(f"**Model artefacts unavailable.**\n\n{exc}")
        return None


@st.cache_resource(show_spinner="Fitting explainability surrogate…")
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


def _style_axes(axes: plt.Axes) -> None:
    """Apply the shared minimal chart styling."""
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color("#c9d6cd")
    axes.tick_params(colors="#5c6f63", labelsize=9)
    axes.grid(axis="x", color="#eef3f0", linewidth=0.8)
    axes.set_axisbelow(True)


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


def render_sidebar() -> Dict[str, object]:
    """Collect every model input. Returns the raw feature dict plus context."""
    _seed_defaults()
    st.sidebar.markdown("### 📍 Location & Climate")

    districts = cached_districts()
    district = st.sidebar.selectbox(
        "District / Taluk",
        options=districts,
        index=districts.index("Udupi") if "Udupi" in districts else 0,
        help="Selects the NFSM laboratory baseline used to pre-fill soil chemistry.",
    )

    city = st.sidebar.text_input("Weather station / City", value=district)
    api_key = st.sidebar.text_input(
        "OpenWeatherMap API key",
        type="password",
        help="Optional. Without a key the system uses calibrated offline defaults.",
    )

    if st.sidebar.button("🌦️ Sync live weather", width="stretch"):
        reading = get_weather(city, api_key or None)
        st.session_state["weather"] = reading
        for key, value in reading.as_dict().items():
            st.session_state[f"in_{key}"] = float(
                np.clip(value, *FEATURE_BOUNDS[key])
            )

    weather = st.session_state.get("weather")
    if weather is not None:
        if weather.is_live:
            st.sidebar.success(
                f"Live · {weather.city} · {weather.temperature:.1f} °C · "
                f"{weather.humidity:.0f} % RH"
            )
        else:
            st.sidebar.info(f"Offline defaults — {weather.message}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Soil Chemistry")

    if st.sidebar.button("📥 Load district baseline", width="stretch"):
        baseline = get_district_baseline(district)
        st.session_state["baseline"] = baseline
        for key, value in baseline.as_dict().items():
            st.session_state[f"in_{key}"] = float(np.clip(value, *FEATURE_BOUNDS[key]))

    baseline = st.session_state.get("baseline")
    if baseline is not None:
        if baseline.is_survey_backed:
            st.sidebar.caption(
                f"NFSM median of {baseline.sample_count:,} laboratory samples "
                f"from {baseline.district}."
            )
        else:
            st.sidebar.caption(
                f"Curated agro-climatic baseline for {baseline.district} "
                f"(source: {baseline.source})."
            )

    values: Dict[str, float] = {}
    for name in ("N", "P", "K"):
        low, high = FEATURE_BOUNDS[name]
        index = FEATURE_NAMES.index(name)
        values[name] = st.sidebar.number_input(
            f"{FEATURE_LABELS[index]} ({FEATURE_UNITS[index]})",
            min_value=float(low),
            max_value=float(high),
            step=1.0,
            key=f"in_{name}",
        )

    values["ph"] = st.sidebar.number_input(
        "Soil pH",
        min_value=float(FEATURE_BOUNDS["ph"][0]),
        max_value=float(FEATURE_BOUNDS["ph"][1]),
        step=0.1,
        key="in_ph",
    )

    st.sidebar.markdown("### 🌡️ Microclimate")
    for name in ("temperature", "humidity", "rainfall"):
        low, high = FEATURE_BOUNDS[name]
        index = FEATURE_NAMES.index(name)
        values[name] = st.sidebar.slider(
            f"{FEATURE_LABELS[index]} ({FEATURE_UNITS[index]})",
            min_value=float(low),
            max_value=float(high),
            step=0.5,
            key=f"in_{name}",
        )

    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "🚀 Generate recommendation", type="primary", width="stretch"
    )

    return {"district": district, "city": city, "features": values, "run": run}


# --------------------------------------------------------------------------- #
# Tab 1 — Precision Recommendation
# --------------------------------------------------------------------------- #
def render_recommendation_tab(state: Dict[str, object]) -> None:
    """Primary crop card, ranked alternatives, advisory, and Z-score profile."""
    prediction = state.get("prediction")
    if prediction is None:
        st.info(
            "Configure soil chemistry and microclimate in the sidebar, then "
            "select **Generate recommendation**."
        )
        return

    advisory = state["advisory"]
    district = state["district"]

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown(
            f"""<div class="gr-primary">
                  <div class="gr-sub">Recommended primary crop</div>
                  <div class="crop">{prediction.crop}</div>
                  <div class="conf">{prediction.confidence:.2f}% posterior
                    probability · {district}</div>
                </div>""",
            unsafe_allow_html=True,
        )

        if prediction.is_low_confidence:
            st.warning(
                "Posterior below 50% — the reading falls between crop "
                "envelopes. Weigh the runner-up options carefully."
            )
        if prediction.is_out_of_distribution:
            st.warning(
                f"Out-of-distribution input: "
                f"**{', '.join(prediction.ood_features)}** exceed "
                f"{OOD_ZSCORE_THRESHOLD:.0f}σ of the training distribution. "
                f"This recommendation is an extrapolation."
            )

        st.markdown("#### Ranked suitability")
        for candidate in prediction.top_k:
            st.markdown(
                f"**{candidate.rank}. {candidate.crop.capitalize()}** — "
                f"{candidate.confidence_pct:.2f}%"
            )
            st.progress(min(max(candidate.probability, 0.0), 1.0))

        st.markdown("#### Persist to audit ledger")
        if st.button("💾 Commit recommendation", width="stretch"):
            _persist(state)

    with right:
        st.markdown("#### Agronomic advisory")
        for item in advisory.items:
            st.markdown(
                f"<div class='gr-advisory {item.severity}'>"
                f"<b>{item.icon} {item.category}"
                f" · {_SEVERITY_LABEL.get(item.severity, '')}</b><br>{item.message}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if advisory.fertiliser_plan:
            st.markdown("#### Fertiliser prescription (per hectare)")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Product": list(advisory.fertiliser_plan),
                        "Quantity (kg/ha)": list(advisory.fertiliser_plan.values()),
                    }
                ),
                hide_index=True,
                width="stretch",
            )

    st.markdown("---")
    st.markdown("#### Input deviation from the benchmark training distribution")
    st.caption(
        "Each bar is the standardised deviation "
        "z = (x − μ) ⁄ σ of an input from the benchmark mean encoded in "
        f"`scaler.pkl`. Bars beyond ±{OOD_ZSCORE_THRESHOLD:.0f}σ mark "
        "covariate shift."
    )

    frame = prediction.z_score_frame()
    figure, axes = plt.subplots(figsize=(10, 3.4))
    colours = [POSITIVE if z >= 0 else NEGATIVE for z in frame["z_score"]]
    axes.barh(FEATURE_LABELS, frame["z_score"], color=colours, height=0.62)
    axes.axvline(0, color="#14281d", linewidth=0.9)
    for bound in (-OOD_ZSCORE_THRESHOLD, OOD_ZSCORE_THRESHOLD):
        axes.axvline(bound, color=NEUTRAL, linewidth=0.8, linestyle="--")
    axes.set_xlabel("Z-score (standard deviations from benchmark mean)", fontsize=10)
    axes.invert_yaxis()
    _style_axes(axes)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _persist(state: Dict[str, object]) -> None:
    """Write the current recommendation to the audit ledger."""
    if not ensure_database():
        st.error("Audit database is unavailable in this environment.")
        return

    prediction = state["prediction"]
    consensus = state.get("consensus")
    features = prediction.raw_features
    try:
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
        )
    except Exception as exc:  # noqa: BLE001 - surface, never crash the dashboard.
        st.error(f"Could not persist the recommendation: {exc}")
        return
    st.success(f"Committed to the audit ledger as record #{record_id}.")


# --------------------------------------------------------------------------- #
# Tab 2 — Explainable AI Consensus
# --------------------------------------------------------------------------- #
def render_xai_tab(state: Dict[str, object]) -> None:
    """Jaccard consensus card and side-by-side SHAP/LIME attributions."""
    prediction = state.get("prediction")
    if prediction is None:
        st.info("Generate a recommendation first to audit its explanation.")
        return

    consensus = state.get("consensus")
    if consensus is None:
        st.warning(
            "The explainability stack is unavailable. Install it with "
            "`pip install shap lime` and restart."
        )
        return

    badge = "ok" if consensus.is_high_fidelity else "warn"
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown(
            f"""<div class="gr-card">
                  <div class="gr-sub">Jaccard Agreement Index (k={consensus.top_k})</div>
                  <div class="gr-metric">{consensus.jaccard:.2f}</div>
                  <div class="gr-sub">{consensus.intersection} of
                    {consensus.union_size} drivers shared</div>
                  <div style="margin-top:12px">
                    <span class="gr-badge {badge}">{consensus.verdict}</span>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.latex(
            r"J(S, L) = \frac{|S \cap L|}{|S \cup L|} = "
            rf"\frac{{{consensus.intersection}}}{{{consensus.union_size}}} = "
            rf"{consensus.jaccard:.2f}"
        )
        st.caption(
            f"S and L are the top-{consensus.top_k} driver sets from TreeSHAP "
            f"and LIME. J ≥ {JACCARD_FIDELITY_THRESHOLD:.1f} is reported as "
            f"High Fidelity."
        )
        st.markdown(f"**Interpretation.** {consensus.interpretation()}")

        if not consensus.surrogate_agrees:
            st.warning(
                "The interpretable surrogate assigns this instance a different "
                "class than the deployed ensemble, so the attribution below "
                "transfers only partially."
            )

    with right:
        metrics = st.columns(3)
        metrics[0].metric("TreeSHAP top-k", ", ".join(consensus.shap_top_k))
        metrics[1].metric("LIME top-k", ", ".join(consensus.lime_top_k))
        metrics[2].metric(
            "Consensus drivers", ", ".join(consensus.consensus_drivers) or "—"
        )

        figure, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
        for index, (title, values, subtitle) in enumerate(
            (
                (
                    "TreeSHAP",
                    consensus.shap_values,
                    "Exact Shapley attribution",
                ),
                (
                    "LIME",
                    consensus.lime_values,
                    "Local surrogate coefficients",
                ),
            )
        ):
            ordered = sorted(values.items(), key=lambda kv: abs(kv[1]))
            names = [name for name, _ in ordered]
            weights = [weight for _, weight in ordered]
            colours = [POSITIVE if w >= 0 else NEGATIVE for w in weights]
            axes[index].barh(names, weights, color=colours, height=0.6)
            axes[index].axvline(0, color="#14281d", linewidth=0.9)
            axes[index].set_title(f"{title}\n{subtitle}", fontsize=10.5)
            axes[index].set_xlabel("Attribution", fontsize=9.5)
            _style_axes(axes[index])
        figure.suptitle(
            f"Local attributions for the {consensus.predicted_crop} decision",
            fontsize=11.5,
        )
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        st.caption(
            "Attribution scales differ between the two methods — only the "
            "*rankings* are compared, which is precisely what the Jaccard "
            "index measures."
        )

    with st.expander("Attribution detail"):
        st.dataframe(
            consensus.to_frame().round(5), hide_index=True, width="stretch"
        )


# --------------------------------------------------------------------------- #
# Tab 3 — What-If Sensitivity Engine
# --------------------------------------------------------------------------- #
def render_sensitivity_tab(state: Dict[str, object]) -> None:
    """Trace the decision surface along a single perturbed feature axis."""
    prediction = state.get("prediction")
    recommender = state.get("recommender")
    if prediction is None or recommender is None:
        st.info("Generate a recommendation first to run a sensitivity sweep.")
        return

    st.markdown("#### Single-factor perturbation analysis")
    st.caption(
        "One feature is swept across a ±100% band around its current value "
        "while the other six are held constant. The curves trace the "
        "ensemble's posterior along that axis, exposing the decision "
        "boundaries it has learned."
    )

    controls = st.columns([1.2, 1, 1])
    feature = controls[0].selectbox(
        "Feature to perturb",
        options=FEATURE_NAMES,
        format_func=lambda name: FEATURE_LABELS[FEATURE_NAMES.index(name)],
    )
    span = controls[1].slider("Sweep range (±%)", 10, 100, 100, step=10)
    n_curves = controls[2].slider("Crops to plot", 2, 8, 4)

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

    figure, axes = plt.subplots(figsize=(11, 4.4))
    palette = plt.cm.viridis(np.linspace(0.08, 0.86, len(ranked)))
    for colour, index in zip(palette, ranked):
        axes.plot(
            grid,
            probabilities[:, index],
            linewidth=2.0,
            color=colour,
            label=classes[int(index)],
        )
    axes.axvline(
        current,
        color=NEGATIVE,
        linestyle="--",
        linewidth=1.4,
        label=f"current = {current:.1f}",
    )
    index = FEATURE_NAMES.index(feature)
    axes.set_xlabel(f"{FEATURE_LABELS[index]} ({FEATURE_UNITS[index]})", fontsize=10)
    axes.set_ylabel("Posterior probability", fontsize=10)
    axes.set_ylim(-0.02, 1.02)
    axes.legend(frameon=False, fontsize=9, ncol=min(len(ranked) + 1, 5))
    _style_axes(axes)
    axes.grid(axis="y", color="#eef3f0", linewidth=0.8)
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
    summary[0].metric(f"Current {feature}", f"{current:.1f}")
    summary[1].metric("Sweep range", f"{grid.min():.1f} – {grid.max():.1f}")
    summary[2].metric("Decision boundaries crossed", str(len(switches)))

    if switches:
        st.markdown("**Recommendation switch points**")
        st.dataframe(
            pd.DataFrame(
                switches, columns=[f"{feature} threshold", "From crop", "To crop"]
            ).round(2),
            hide_index=True,
            width="stretch",
        )
    else:
        st.success(
            f"The {prediction.crop} recommendation is stable across the entire "
            f"±{span}% sweep of {feature} — this factor is not the binding "
            f"constraint for this plot."
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
    st.markdown("#### Bulk advisory from a soil survey")
    st.caption(
        "Upload a laboratory export or survey sheet and receive a "
        "recommendation for every sample. Column names are matched leniently — "
        "`N`, `Nitrogen` and `avl_n` are all understood — and each row is "
        "validated independently, so one bad reading never aborts the run."
    )

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
        "Needs review",
        f"{summary['low_confidence'] + summary['out_of_distribution']:,}",
        help="Samples with confidence below 50% or inputs outside the "
             "training distribution.",
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
            axes.set_xlabel("Samples recommended", fontsize=10)
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
    st.markdown("#### Recommendation audit ledger")
    st.caption(
        "Every committed recommendation is recorded with its inputs, its "
        "confidence, its dominant SHAP driver, and its explainer agreement "
        "index — the evidence trail behind advice acted on in the field."
    )

    if not ensure_database():
        st.error("Audit database is unavailable in this environment.")
        return

    controls = st.columns([1, 1, 1, 1])
    today = date.today()
    start = controls[0].date_input("From", value=today - timedelta(days=30))
    end = controls[1].date_input("To", value=today)
    limit = controls[2].number_input("Max records", 10, 5000, 200, step=10)

    frame = db_manager.fetch_audit_history(
        limit=int(limit),
        start_date=str(start),
        end_date=str(end),
    )
    controls[3].metric("Total records", f"{db_manager.count_records():,}")

    if frame.empty:
        st.info(
            "No records in this window. Commit a recommendation from the "
            "**Precision Recommendation** tab to populate the ledger."
        )
        return

    summary = st.columns(4)
    summary[0].metric("Records shown", f"{len(frame):,}")
    summary[1].metric("Mean confidence", f"{frame['confidence'].mean():.1f}%")
    summary[2].metric("Distinct crops", f"{frame['recommended_crop'].nunique()}")
    jaccard = pd.to_numeric(frame["jaccard_index"], errors="coerce").dropna()
    summary[3].metric(
        "Mean Jaccard", f"{jaccard.mean():.2f}" if not jaccard.empty else "—"
    )

    st.dataframe(frame, hide_index=True, width="stretch", height=380)

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    st.download_button(
        "⬇️ Download audit trail (CSV)",
        data=buffer.getvalue(),
        file_name=f"greenroot_audit_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Distribution by recommended crop"):
        counts = frame["recommended_crop"].value_counts()
        figure, axes = plt.subplots(figsize=(9, max(2.4, 0.34 * len(counts))))
        axes.barh(counts.index[::-1], counts.to_numpy()[::-1], color=ACCENT, height=0.6)
        axes.set_xlabel("Recommendations logged", fontsize=10)
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

    st.markdown("#### Printable soil health card")
    st.caption(
        "The card an extension officer hands to the cultivator. Every export "
        "renders from one payload, so the figures cannot diverge between "
        "formats."
    )

    controls = st.columns([1.4, 2])
    bilingual = controls[0].toggle(
        "ಕನ್ನಡ · Bilingual card",
        value=False,
        help="Show Kannada alongside English on the farmer-facing card.",
    )
    if bilingual:
        controls[1].caption(
            "English is retained beside every Kannada term, so a translation "
            "error cannot silently change the advice. Translations are a "
            "prototype mapping and need native-speaker review before field use."
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
    downloads = st.columns(4)

    try:
        pdf_bytes = render_pdf(card)
    except PDFUnavailableError:
        pdf_bytes = None

    if pdf_bytes is not None:
        downloads[0].download_button(
            "⬇️ PDF (print)",
            data=pdf_bytes,
            file_name=f"soil_health_card_{stamp}.pdf",
            mime="application/pdf",
            type="primary",
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

    if bilingual and pdf_bytes is not None:
        st.caption(
            "The PDF is English-only: its core fonts are Latin-1 and this "
            "project does not bundle a Kannada typeface. Use the HTML export "
            "for a bilingual printout."
        )

    with st.expander("Plain-text preview (SMS / thermal printer)"):
        st.code(render_text(card), language=None)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    """Assemble the dashboard."""
    st.markdown(
        f"""<div class="gr-hero">
              <h1>🌱 GREENROOT — Intelligent Precision Agriculture DSS</h1>
              <p>Stacking ensemble meta-learning · multi-explainer consensus
                 auditing · {REPORTED_CV_ACCURACY * 100:.2f}% stratified
                 5-fold cross-validated accuracy across 22 crop classes</p>
            </div>""",
        unsafe_allow_html=True,
    )

    recommender = load_recommender()
    if recommender is None:
        st.stop()

    inputs = render_sidebar()
    ensure_database()

    if inputs["run"]:
        features: Dict[str, float] = inputs["features"]  # type: ignore[assignment]
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
            with st.spinner("Auditing the decision with TreeSHAP and LIME…"):
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
        "recommender": recommender,
    }

    tabs = st.tabs(
        [
            "🎯 Precision Recommendation",
            "🔍 Explainable AI Consensus",
            "🧭 What-If Sensitivity",
            "📦 Bulk Advisory",
            "📋 Audit Trail & Governance",
            "🧾 Farmer Soil Health Card",
        ]
    )
    with tabs[0]:
        render_recommendation_tab(state)
    with tabs[1]:
        render_xai_tab(state)
    with tabs[2]:
        render_sensitivity_tab(state)
    with tabs[3]:
        render_bulk_tab()
    with tabs[4]:
        render_audit_tab()
    with tabs[5]:
        render_card_tab(state)

    st.markdown("---")
    st.caption(
        f"GREENROOT · stacking ensemble (Random Forest + AdaBoost + k-NN → "
        f"logistic-regression meta-learner) · {len(recommender.class_names)} "
        f"crop classes · Jaccard consensus at k={CONSENSUS_TOP_K}. "
        f"Advisory output only — corroborate with a certified laboratory soil "
        f"test before committing a season."
    )


if __name__ == "__main__":
    main()
