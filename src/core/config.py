"""Centralised configuration for the GREENROOT decision support system.

This module is the single source of truth for filesystem paths, the canonical
feature contract of the pre-trained stacking ensemble, physiological validity
bounds used for input sanitisation, and the thresholds that govern the
multi-explainer consensus audit.

No other module in ``src`` should hard-code a path, a feature name, or a
numeric threshold; importing from here keeps the inference contract aligned
with the artefacts persisted in ``models/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Final, List, Tuple

# --------------------------------------------------------------------------- #
# 1. Filesystem topology
# --------------------------------------------------------------------------- #
#: Repository root (``src/core/config.py`` -> ``src/core`` -> ``src`` -> root).
BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"

# Pre-trained artefacts. These are read-only: the system never retrains or
# overwrites them at runtime.
MODEL_PATH: Final[Path] = MODELS_DIR / "stacking_model.pkl"
SCALER_PATH: Final[Path] = MODELS_DIR / "scaler.pkl"
CLASS_NAMES_PATH: Final[Path] = MODELS_DIR / "class_names.pkl"

#: Derived (regenerable) cache for the interpretable surrogate used by TreeSHAP.
#: Distinct from the three protected artefacts above.
SURROGATE_PATH: Final[Path] = MODELS_DIR / "xai_surrogate.pkl"

# Datasets
BENCHMARK_DATASET: Final[Path] = DATA_DIR / "Crop_recommendation.csv"
NFSM_DATASET: Final[Path] = DATA_DIR / "Cleaned_NFSM_Dataset.csv"
MASTER_DATASET: Final[Path] = DATA_DIR / "GreenRoot_Consolidated_Master_Dataset.csv"

#: Audit / governance store.
DATABASE_PATH: Final[Path] = BASE_DIR / "crop_recommendations.db"

# Evaluation artefacts emitted by ``evaluate_system.py``.
CONFUSION_MATRIX_PATH: Final[Path] = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH: Final[Path] = REPORTS_DIR / "classification_report.csv"
LATENCY_REPORT_PATH: Final[Path] = REPORTS_DIR / "latency_benchmark.csv"
OOD_REPORT_PATH: Final[Path] = REPORTS_DIR / "ood_robustness.csv"
EVALUATION_SUMMARY_PATH: Final[Path] = REPORTS_DIR / "evaluation_summary.md"

# --------------------------------------------------------------------------- #
# 2. Feature contract
# --------------------------------------------------------------------------- #
#: Ordered feature vector expected by ``scaler.pkl`` and ``stacking_model.pkl``.
#: The ordering is load-bearing — never reorder without refitting the scaler.
FEATURE_NAMES: Final[List[str]] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

#: Human-readable labels for dashboards and reports, index-aligned with
#: :data:`FEATURE_NAMES`.
FEATURE_LABELS: Final[List[str]] = [
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)",
    "Temperature",
    "Humidity",
    "Soil pH",
    "Rainfall",
]

#: Measurement units, index-aligned with :data:`FEATURE_NAMES`.
FEATURE_UNITS: Final[List[str]] = [
    "kg/ha",
    "kg/ha",
    "kg/ha",
    "°C",
    "%",
    "pH",
    "mm",
]

N_FEATURES: Final[int] = len(FEATURE_NAMES)

#: Number of crop classes in the target space of the stacking ensemble.
N_CLASSES: Final[int] = 22

# --------------------------------------------------------------------------- #
# 3. Physiological safety bounds
# --------------------------------------------------------------------------- #
#: Hard admissibility envelope, ``feature -> (minimum, maximum)``.
#:
#: Values outside these bounds are physically implausible for an agronomic soil
#: test and are rejected before they ever reach the scaler. The pH window in
#: particular guards against the decimal-shift artefacts present in raw NFSM
#: laboratory exports (observed values up to 752).
FEATURE_BOUNDS: Final[Dict[str, Tuple[float, float]]] = {
    "N": (0.0, 500.0),
    "P": (0.0, 500.0),
    "K": (0.0, 800.0),
    "temperature": (0.0, 55.0),
    "humidity": (0.0, 100.0),
    "ph": (3.0, 10.0),
    "rainfall": (0.0, 1200.0),
}

#: Ergonomic slider defaults for the dashboard; a coastal-Karnataka kharif
#: profile that sits inside the benchmark training manifold.
DEFAULT_INPUTS: Final[Dict[str, float]] = {
    "N": 60.0,
    "P": 40.0,
    "K": 45.0,
    "temperature": 26.0,
    "humidity": 78.0,
    "ph": 6.2,
    "rainfall": 190.0,
}

#: |Z| beyond this is flagged as a covariate-shift / out-of-distribution signal
#: relative to the benchmark training distribution encoded in ``scaler.pkl``.
OOD_ZSCORE_THRESHOLD: Final[float] = 3.0

#: Predictions below this softmax confidence are surfaced as low-certainty.
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.50

# --------------------------------------------------------------------------- #
# 4. Explainability consensus
# --------------------------------------------------------------------------- #
#: Cardinality ``k`` of the top-driver sets compared by the Jaccard index.
CONSENSUS_TOP_K: Final[int] = 3

#: J >= this value is reported as "High Fidelity"; below it, "Local Divergence".
JACCARD_FIDELITY_THRESHOLD: Final[float] = 0.50

#: Trees in the interpretable surrogate that TreeSHAP is computed against.
SURROGATE_N_ESTIMATORS: Final[int] = 120
SURROGATE_MAX_DEPTH: Final[int] = 12

#: Perturbation samples drawn by the LIME tabular explainer per instance.
LIME_NUM_SAMPLES: Final[int] = 1000

#: Global determinism seed for every stochastic component.
RANDOM_STATE: Final[int] = 42

# --------------------------------------------------------------------------- #
# 5. External services
# --------------------------------------------------------------------------- #
OPENWEATHER_ENDPOINT: Final[str] = "https://api.openweathermap.org/data/2.5/weather"

#: Hard ceiling on the outbound weather call, in seconds. The dashboard must
#: never block a farmer-facing interaction for longer than this.
WEATHER_TIMEOUT_SECONDS: Final[float] = 5.0

#: Time-to-live of the in-process weather response cache, in seconds.
WEATHER_CACHE_TTL_SECONDS: Final[float] = 600.0

#: Read from the environment so the key is never committed to source control.
OPENWEATHER_API_KEY: Final[str] = os.getenv("OPENWEATHER_API_KEY", "")

#: Deterministic offline fallback used when no key is configured or the network
#: is unreachable. Values approximate a coastal-Karnataka monsoon microclimate.
WEATHER_MOCK_DEFAULTS: Final[Dict[str, float]] = {
    "temperature": 26.0,
    "humidity": 78.0,
    "rainfall": 190.0,
}

# --------------------------------------------------------------------------- #
# 6. Evaluation protocol
# --------------------------------------------------------------------------- #
TEST_SPLIT_SIZE: Final[float] = 0.20
LATENCY_TRIALS: Final[int] = 100
LATENCY_WARMUP_TRIALS: Final[int] = 5
FIGURE_DPI: Final[int] = 300

#: Headline cross-validated accuracy of the persisted stacking ensemble,
#: reported by ``train.py`` under stratified 5-fold CV.
#: Shown in About. Bump on a release the user would notice.
APP_VERSION: Final[str] = "1.1.0"

REPORTED_CV_ACCURACY: Final[float] = 0.9941


def ensure_directories() -> None:
    """Create the writable output directories if they do not already exist.

    Read-only asset directories (``data/``, ``models/``) are deliberately not
    created here — their absence is a configuration error that should surface
    loudly at load time rather than be silently papered over.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [name for name in dir() if not name.startswith("_")]
