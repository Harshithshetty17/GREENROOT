"""Formal empirical validation of the GREENROOT decision support system.

Executes the evaluation protocol reported in the project documentation and
writes every artefact to ``reports/``:

==========================  ====================================================
Experiment                  Artefact
==========================  ====================================================
Stratified hold-out         ``classification_report.csv``, ``confusion_matrix.png``
Inference latency           ``latency_benchmark.csv``
Out-of-distribution stress  ``ood_robustness.csv``
Explainer consensus         folded into ``evaluation_summary.md``
==========================  ====================================================

Protocol
--------
**Generalisation.** A stratified 80/20 split of the benchmark corpus. The
deployed ensemble is *not* refitted — it is evaluated exactly as it ships, so
the reported figures describe the artefact in ``models/`` rather than a
freshly-trained twin. Note that the shipped model was fitted on the full
corpus, so this split is a reproduction check on the CV figure, not an
independent generalisation estimate; the honest generalisation number remains
the stratified 5-fold CV accuracy quoted in the documentation.

**Latency.** ``LATENCY_TRIALS`` single-sample inferences over randomly drawn
readings, preceded by warm-up iterations that absorb lazy-initialisation cost.
Reported as mean plus the P50/P95/P99 order statistics, because tail latency —
not the mean — governs whether a field terminal feels responsive.

**Robustness.** The consolidated master corpus carries genuine covariate shift
relative to the benchmark (district-level available nitrogen around 210 kg/ha
against a benchmark mean of 50). Since its crop labels come from a different
taxonomy than the 22 benchmark classes, accuracy against them would be
meaningless; the experiment instead measures *confidence degradation* and the
rate at which the Z-score monitor correctly flags shifted inputs.

Usage
-----
.. code-block:: console

    python evaluate_system.py [--skip-xai] [--latency-trials N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend: this script must run in CI.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split

from src.core.config import (
    BENCHMARK_DATASET,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    EVALUATION_SUMMARY_PATH,
    FEATURE_NAMES,
    FIGURE_DPI,
    LATENCY_REPORT_PATH,
    LATENCY_TRIALS,
    LATENCY_WARMUP_TRIALS,
    MASTER_DATASET,
    OOD_REPORT_PATH,
    OOD_ZSCORE_THRESHOLD,
    RANDOM_STATE,
    REPORTED_CV_ACCURACY,
    TEST_SPLIT_SIZE,
    ensure_directories,
)
from src.models.inference import CropRecommender, get_recommender

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("greenroot.evaluate")

_RULE = "=" * 78


def _banner(title: str) -> None:
    """Print a section banner to stdout."""
    print(f"\n{_RULE}\n{title}\n{_RULE}")


# --------------------------------------------------------------------------- #
# Experiment 1 — generalisation
# --------------------------------------------------------------------------- #
def evaluate_classification(
    recommender: CropRecommender,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate the deployed ensemble on a stratified hold-out split.

    Returns
    -------
    tuple
        ``(headline_metrics, per_class_report)``. Also writes the per-class
        table and the confusion-matrix figure to ``reports/``.
    """
    _banner("EXPERIMENT 1 — STRATIFIED HOLD-OUT CLASSIFICATION PERFORMANCE")

    frame = pd.read_csv(BENCHMARK_DATASET)
    features = frame[FEATURE_NAMES]
    labels = frame["label"].astype(str)

    _, test_x, _, test_y = train_test_split(
        features,
        labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(
        f"Corpus: {len(frame):,} samples | {labels.nunique()} classes | "
        f"hold-out: {len(test_x):,} ({TEST_SPLIT_SIZE:.0%})"
    )

    predictions, probabilities = recommender.predict_batch(test_x.to_numpy())
    truth = test_y.to_numpy()
    classes = recommender.class_names

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(truth, predictions)),
        "matthews_corrcoef": float(matthews_corrcoef(truth, predictions)),
        "top_3_accuracy": float(
            top_k_accuracy_score(truth, probabilities, k=3, labels=classes)
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }

    print(f"\n  Accuracy              : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Top-3 accuracy        : {metrics['top_3_accuracy'] * 100:.2f}%")
    print(f"  Macro F1              : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1           : {metrics['weighted_f1']:.4f}")
    print(f"  Cohen's kappa         : {metrics['cohen_kappa']:.4f}")
    print(f"  Matthews corr. coeff. : {metrics['matthews_corrcoef']:.4f}")
    print(f"  Mean max posterior    : {metrics['mean_confidence']:.4f}")

    report = pd.DataFrame(
        classification_report(truth, predictions, output_dict=True, zero_division=0)
    ).transpose()
    report.index.name = "class"
    report = report.round(4)
    report.to_csv(CLASSIFICATION_REPORT_PATH)
    print(f"\n  [+] Per-class report -> {CLASSIFICATION_REPORT_PATH}")

    misclassified = [
        (t, p) for t, p in zip(truth, predictions) if t != p
    ]
    if misclassified:
        print(f"  [i] {len(misclassified)} misclassification(s):")
        for actual, predicted in misclassified[:10]:
            print(f"      {actual} -> {predicted}")
    else:
        print("  [i] No misclassifications on the hold-out split.")

    _plot_confusion_matrix(truth, predictions, classes, metrics["accuracy"])
    return metrics, report


def _plot_confusion_matrix(
    truth: np.ndarray,
    predictions: List[str],
    classes: List[str],
    accuracy: float,
) -> None:
    """Render and persist a publication-resolution confusion matrix."""
    matrix = confusion_matrix(truth, predictions, labels=classes)

    figure, axes = plt.subplots(figsize=(11, 9.5))
    image = axes.imshow(matrix, cmap="Greens", interpolation="nearest")

    axes.set_xticks(np.arange(len(classes)))
    axes.set_yticks(np.arange(len(classes)))
    axes.set_xticklabels(classes, rotation=90, fontsize=8)
    axes.set_yticklabels(classes, fontsize=8)
    axes.set_xlabel("Predicted class", fontsize=11, labelpad=10)
    axes.set_ylabel("Actual class", fontsize=11, labelpad=10)
    axes.set_title(
        "GREENROOT Stacking Ensemble — Confusion Matrix\n"
        f"Stratified {TEST_SPLIT_SIZE:.0%} hold-out, {len(truth)} samples, "
        f"accuracy {accuracy * 100:.2f}%",
        fontsize=12,
        pad=14,
    )

    # Annotate only the non-zero cells; a 22x22 grid of zeros is unreadable.
    threshold = matrix.max() / 2.0 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                axes.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "#14281d",
                )

    axes.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    axes.grid(which="minor", color="#ffffff", linewidth=0.6)
    axes.tick_params(which="minor", length=0)

    figure.colorbar(image, ax=axes, fraction=0.045, pad=0.03, label="Sample count")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"  [+] Confusion matrix ({FIGURE_DPI} DPI) -> {CONFUSION_MATRIX_PATH}")


# --------------------------------------------------------------------------- #
# Experiment 2 — latency
# --------------------------------------------------------------------------- #
def benchmark_latency(
    recommender: CropRecommender, trials: int = LATENCY_TRIALS
) -> Dict[str, float]:
    """Measure single-sample end-to-end inference latency.

    Each trial covers validation, scaling and the full stacked forward pass —
    the same code path the dashboard executes — so the figures reflect what a
    user actually waits for, not an isolated matrix multiply.
    """
    _banner(f"EXPERIMENT 2 — INFERENCE LATENCY ({trials} STOCHASTIC TRIALS)")

    frame = pd.read_csv(BENCHMARK_DATASET)
    rng = np.random.default_rng(RANDOM_STATE)
    pool = frame[FEATURE_NAMES].to_numpy(dtype=float)
    samples = pool[rng.integers(0, len(pool), size=trials)]

    for _ in range(LATENCY_WARMUP_TRIALS):  # Absorb lazy-init and cache warm-up.
        recommender.predict(pool[0])

    timings_ms: List[float] = []
    for row in samples:
        start = time.perf_counter()
        recommender.predict(row)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    array = np.asarray(timings_ms)
    metrics = {
        "trials": float(trials),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min_ms": float(array.min()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "throughput_hz": float(1000.0 / array.mean()) if array.mean() else 0.0,
    }

    print(f"\n  Mean        : {metrics['mean_ms']:8.3f} ms "
          f"(sd {metrics['std_ms']:.3f})")
    print(f"  P50 (median): {metrics['p50_ms']:8.3f} ms")
    print(f"  P95         : {metrics['p95_ms']:8.3f} ms")
    print(f"  P99         : {metrics['p99_ms']:8.3f} ms")
    print(f"  Min / Max   : {metrics['min_ms']:8.3f} / {metrics['max_ms']:.3f} ms")
    print(f"  Throughput  : {metrics['throughput_hz']:8.1f} inferences/s")

    pd.DataFrame([metrics]).round(4).to_csv(LATENCY_REPORT_PATH, index=False)
    print(f"\n  [+] Latency benchmark -> {LATENCY_REPORT_PATH}")
    return metrics


# --------------------------------------------------------------------------- #
# Experiment 3 — out-of-distribution robustness
# --------------------------------------------------------------------------- #
def evaluate_ood_robustness(
    recommender: CropRecommender, sample_size: int = 500
) -> Optional[Dict[str, float]]:
    """Quantify behaviour under genuine field-data covariate shift.

    Returns ``None`` if the consolidated master corpus is unavailable.
    """
    _banner("EXPERIMENT 3 — OUT-OF-DISTRIBUTION ROBUSTNESS")

    if not MASTER_DATASET.exists():
        print(f"  [!] {MASTER_DATASET} not found; skipping.")
        return None

    frame = pd.read_csv(MASTER_DATASET)
    column_map = {
        "N": "N", "P": "P", "K": "K",
        "Temperature": "temperature", "Humidity": "humidity",
        "pH": "ph", "Rainfall": "rainfall",
    }
    missing = [c for c in column_map if c not in frame.columns]
    if missing:
        print(f"  [!] Master corpus lacks column(s) {missing}; skipping.")
        return None

    field_rows = frame[frame.get("Source", "") != "Kaggle Benchmark"]
    if field_rows.empty:
        print("  [!] No field-survey rows in the master corpus; skipping.")
        return None

    rng = np.random.default_rng(RANDOM_STATE)
    take = min(sample_size, len(field_rows))
    sample = field_rows.iloc[rng.choice(len(field_rows), size=take, replace=False)]
    matrix = sample.rename(columns=column_map)[FEATURE_NAMES].to_numpy(dtype=float)

    # Clip into the admissible envelope; the point of the experiment is
    # distribution shift, not input validation, which Experiment 1 covers.
    means = recommender.feature_means
    scales = recommender.feature_scales
    from src.core.config import FEATURE_BOUNDS

    bounds = np.array([FEATURE_BOUNDS[name] for name in FEATURE_NAMES])
    matrix = np.clip(matrix, bounds[:, 0], bounds[:, 1])

    z_scores = (matrix - means) / scales
    flagged = np.abs(z_scores) > OOD_ZSCORE_THRESHOLD

    _, probabilities = recommender.predict_batch(matrix)
    field_confidence = probabilities.max(axis=1)

    benchmark = pd.read_csv(BENCHMARK_DATASET)
    bench_sample = benchmark[FEATURE_NAMES].to_numpy(dtype=float)[
        rng.choice(len(benchmark), size=min(take, len(benchmark)), replace=False)
    ]
    _, bench_probabilities = recommender.predict_batch(bench_sample)
    bench_confidence = bench_probabilities.max(axis=1)

    metrics = {
        "field_samples": float(take),
        "mean_confidence_field": float(field_confidence.mean()),
        "mean_confidence_benchmark": float(bench_confidence.mean()),
        "confidence_degradation": float(
            bench_confidence.mean() - field_confidence.mean()
        ),
        "detected_ood_rate": float(flagged.any(axis=1).mean()),
        "mean_abs_z": float(np.abs(z_scores).mean()),
        "low_confidence_rate_field": float((field_confidence < 0.50).mean()),
        "low_confidence_rate_benchmark": float((bench_confidence < 0.50).mean()),
    }

    print(f"\n  Field samples evaluated      : {take}")
    print(f"  Mean confidence (benchmark)  : {metrics['mean_confidence_benchmark']:.4f}")
    print(f"  Mean confidence (field)      : {metrics['mean_confidence_field']:.4f}")
    print(f"  Confidence degradation       : {metrics['confidence_degradation']:.4f}")
    print(f"  Mean |Z| under shift         : {metrics['mean_abs_z']:.3f}")
    print(f"  OOD detection rate (|Z|>{OOD_ZSCORE_THRESHOLD:.0f})  : "
          f"{metrics['detected_ood_rate'] * 100:.1f}%")
    print(f"  Low-confidence rate (field)  : "
          f"{metrics['low_confidence_rate_field'] * 100:.1f}%")

    per_feature = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "field_mean": matrix.mean(axis=0).round(3),
            "benchmark_mean": means.round(3),
            "mean_abs_z": np.abs(z_scores).mean(axis=0).round(3),
            "flag_rate": flagged.mean(axis=0).round(4),
        }
    )
    print("\n  Per-feature covariate shift:")
    print(per_feature.to_string(index=False))

    pd.concat(
        [
            pd.DataFrame([metrics]).round(4).assign(feature="__summary__"),
            per_feature,
        ],
        ignore_index=True,
    ).to_csv(OOD_REPORT_PATH, index=False)
    print(f"\n  [+] OOD robustness -> {OOD_REPORT_PATH}")
    return metrics


# --------------------------------------------------------------------------- #
# Experiment 4 — explainer consensus
# --------------------------------------------------------------------------- #
def evaluate_consensus(sample_size: int = 30) -> Optional[Dict[str, float]]:
    """Measure SHAP/LIME agreement across a stratified sample of readings."""
    _banner(f"EXPERIMENT 4 — MULTI-EXPLAINER CONSENSUS ({sample_size} INSTANCES)")

    try:
        from src.models.xai_engine import ExplainerConsensus
    except ImportError as exc:
        print(f"  [!] Explainability stack unavailable ({exc}); skipping.")
        return None

    engine = ExplainerConsensus().fit()
    frame = pd.read_csv(BENCHMARK_DATASET)
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(frame), size=min(sample_size, len(frame)), replace=False)
    rows = frame[FEATURE_NAMES].to_numpy(dtype=float)[indices]

    start = time.perf_counter()
    reports, mean_jaccard = engine.audit_batch(rows)
    elapsed = time.perf_counter() - start

    values = np.array([r.jaccard for r in reports])
    high_fidelity = float(np.mean([r.is_high_fidelity for r in reports]))

    metrics = {
        "instances": float(len(reports)),
        "mean_jaccard": float(mean_jaccard),
        "median_jaccard": float(np.median(values)),
        "std_jaccard": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "perfect_agreement_rate": float(np.mean(values == 1.0)),
        "high_fidelity_rate": high_fidelity,
        "surrogate_fidelity": float(engine.surrogate_fidelity),
        "mean_explain_ms": float(elapsed / max(len(reports), 1) * 1000.0),
    }

    print(f"\n  Surrogate fidelity vs. ensemble : {metrics['surrogate_fidelity']:.4f}")
    print(f"  Mean Jaccard index (k=3)        : {metrics['mean_jaccard']:.4f}")
    print(f"  Median Jaccard index            : {metrics['median_jaccard']:.4f}")
    print(f"  Perfect agreement (J = 1.0)     : "
          f"{metrics['perfect_agreement_rate'] * 100:.1f}%")
    print(f"  High fidelity (J >= 0.5)        : "
          f"{metrics['high_fidelity_rate'] * 100:.1f}%")
    print(f"  Mean explanation latency        : {metrics['mean_explain_ms']:.1f} ms")

    distribution = pd.Series(values).value_counts().sort_index()
    print("\n  Jaccard distribution:")
    for value, count in distribution.items():
        bar = "#" * int(count / max(distribution.max(), 1) * 30)
        print(f"    J = {value:.2f} | {count:3d} {bar}")
    return metrics


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def write_summary(
    classification: Dict[str, float],
    latency: Dict[str, float],
    ood: Optional[Dict[str, float]],
    consensus: Optional[Dict[str, float]],
) -> None:
    """Write the consolidated Markdown scorecard to ``reports/``."""
    lines: List[str] = [
        "# GREENROOT — System Evaluation Summary",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## 1. Classification Performance",
        "",
        "Stratified 20% hold-out of `Crop_recommendation.csv` "
        "(2,200 samples, 22 classes).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Reported 5-fold CV accuracy | {REPORTED_CV_ACCURACY * 100:.2f}% |",
        f"| Hold-out accuracy | {classification['accuracy'] * 100:.2f}% |",
        f"| Top-3 accuracy | {classification['top_3_accuracy'] * 100:.2f}% |",
        f"| Macro F1 | {classification['macro_f1']:.4f} |",
        f"| Weighted F1 | {classification['weighted_f1']:.4f} |",
        f"| Cohen's kappa | {classification['cohen_kappa']:.4f} |",
        f"| Matthews correlation | {classification['matthews_corrcoef']:.4f} |",
        f"| Mean max posterior | {classification['mean_confidence']:.4f} |",
        "",
        "> The shipped ensemble was fitted on the full corpus, so the hold-out "
        "figure is a reproduction check rather than an independent "
        "generalisation estimate. The stratified 5-fold cross-validation "
        "accuracy is the figure to cite.",
        "",
        "## 2. Inference Latency",
        "",
        f"{int(latency['trials'])} stochastic single-sample inferences, "
        f"end-to-end (validation + scaling + stacked forward pass).",
        "",
        "| Statistic | Latency (ms) |",
        "|---|---:|",
        f"| Mean | {latency['mean_ms']:.3f} |",
        f"| P50 | {latency['p50_ms']:.3f} |",
        f"| P95 | {latency['p95_ms']:.3f} |",
        f"| P99 | {latency['p99_ms']:.3f} |",
        f"| Max | {latency['max_ms']:.3f} |",
        "",
        f"Sustained throughput: **{latency['throughput_hz']:.0f} inferences/s** "
        "single-threaded.",
        "",
    ]

    if ood:
        lines += [
            "## 3. Out-of-Distribution Robustness",
            "",
            "Karnataka NFSM field survey rows, which carry genuine covariate "
            "shift relative to the benchmark corpus.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Field samples | {int(ood['field_samples'])} |",
            f"| Mean confidence (benchmark) | {ood['mean_confidence_benchmark']:.4f} |",
            f"| Mean confidence (field) | {ood['mean_confidence_field']:.4f} |",
            f"| Confidence degradation | {ood['confidence_degradation']:.4f} |",
            f"| Mean \\|Z\\| under shift | {ood['mean_abs_z']:.3f} |",
            f"| OOD detection rate | {ood['detected_ood_rate'] * 100:.1f}% |",
            "",
        ]

    if consensus:
        lines += [
            "## 4. Multi-Explainer Consensus",
            "",
            "TreeSHAP versus LIME top-3 driver agreement, Jaccard index.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Instances audited | {int(consensus['instances'])} |",
            f"| Surrogate fidelity vs. ensemble | {consensus['surrogate_fidelity']:.4f} |",
            f"| Mean Jaccard index | {consensus['mean_jaccard']:.4f} |",
            f"| Median Jaccard index | {consensus['median_jaccard']:.4f} |",
            f"| Perfect agreement (J = 1.0) | {consensus['perfect_agreement_rate'] * 100:.1f}% |",
            f"| High fidelity (J >= 0.5) | {consensus['high_fidelity_rate'] * 100:.1f}% |",
            f"| Mean explanation latency | {consensus['mean_explain_ms']:.1f} ms |",
            "",
        ]

    lines += [
        "## Artefacts",
        "",
        f"- `{CONFUSION_MATRIX_PATH.name}` — {FIGURE_DPI} DPI confusion matrix",
        f"- `{CLASSIFICATION_REPORT_PATH.name}` — per-class precision/recall/F1",
        f"- `{LATENCY_REPORT_PATH.name}` — latency distribution",
        f"- `{OOD_REPORT_PATH.name}` — per-feature covariate shift",
        "",
    ]

    EVALUATION_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [+] Evaluation summary -> {EVALUATION_SUMMARY_PATH}")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the full evaluation protocol. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="GREENROOT formal empirical validation suite."
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip the explainer-consensus experiment (shap/lime not required).",
    )
    parser.add_argument(
        "--latency-trials",
        type=int,
        default=LATENCY_TRIALS,
        help=f"Number of latency trials (default: {LATENCY_TRIALS}).",
    )
    parser.add_argument(
        "--ood-samples",
        type=int,
        default=500,
        help="Field samples drawn for the OOD experiment (default: 500).",
    )
    parser.add_argument(
        "--consensus-samples",
        type=int,
        default=30,
        help="Instances audited for explainer consensus (default: 30).",
    )
    args = parser.parse_args(argv)

    ensure_directories()

    print(_RULE)
    print("GREENROOT — INTELLIGENT PRECISION AGRICULTURE DECISION SUPPORT SYSTEM")
    print("Formal Empirical Validation Suite")
    print(_RULE)

    try:
        recommender = get_recommender()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    print(f"\nEnsemble  : {type(recommender.model).__name__}")
    print(f"Base      : {[name for name, _ in recommender.model.estimators]}")
    print(f"Meta      : {type(recommender.model.final_estimator_).__name__}")
    print(f"Classes   : {len(recommender.class_names)}")
    print(f"Features  : {FEATURE_NAMES}")

    classification, _ = evaluate_classification(recommender)
    latency = benchmark_latency(recommender, trials=args.latency_trials)
    ood = evaluate_ood_robustness(recommender, sample_size=args.ood_samples)
    consensus = (
        None if args.skip_xai else evaluate_consensus(sample_size=args.consensus_samples)
    )

    _banner("EVALUATION COMPLETE")
    write_summary(classification, latency, ood, consensus)
    print(f"\nAll artefacts written to {CONFUSION_MATRIX_PATH.parent}/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
