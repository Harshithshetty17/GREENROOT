#!/usr/bin/env python3
"""Independent cross-validation of the GREENROOT stacking architecture.

Refits the architecture from scratch, in memory, and reports what it scores.
Nothing here touches ``models/``: the shipped ``stacking_model.pkl``,
``scaler.pkl`` and ``class_names.pkl`` are never loaded, written, or deleted,
and the script asserts as much before it exits.

Why this exists
---------------
``train.py`` reports a cross-validated accuracy, but it scales the whole
dataset *before* the split::

    X_scaled = scaler.fit_transform(X)          # sees every row
    cross_val_score(clf, X_scaled, y, cv=cv)    # then splits

Every validation fold is therefore standardised using statistics that include
its own rows. That is preprocessing leakage, and a reported figure obtained
that way is not a clean out-of-sample estimate. This script runs the
methodologically correct version -- the scaler refitted inside each fold via a
``Pipeline`` -- and also reproduces the leaky variant, so the size of the
difference is measured rather than assumed.

On the confidence interval
--------------------------
The interval printed below is the ordinary t-interval over the five fold
scores. It is reported because it is what the brief asks for, but it is
**optimistic**, and an examiner should know why: the five training sets
overlap heavily (each shares 3/4 of its rows with every other), so the fold
scores are positively correlated and their variance understates the true
sampling variance. Nadeau and Bengio's corrected resampled t-test is the
right instrument for comparing two architectures on the same folds; see
``validate_architecture.py``, which uses it.

Usage
-----
    python verify_cv.py [--folds 5] [--seed 42] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "Crop_recommendation.csv"
MODEL_DIR = ROOT / "models"
PROTECTED = ("stacking_model.pkl", "scaler.pkl", "class_names.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

#: 97.5th percentile of Student's t, by degrees of freedom. Avoids a scipy
#: dependency for the handful of fold counts anyone actually uses.
_T_CRITICAL: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    14: 2.145, 19: 2.093, 29: 2.045,
}


def _t_critical(df: int) -> float:
    """Two-sided 95% critical value, falling back to the normal limit."""
    if df in _T_CRITICAL:
        return _T_CRITICAL[df]
    nearest = min(_T_CRITICAL, key=lambda k: abs(k - df))
    return _T_CRITICAL[nearest] if df < 30 else 1.960


def build_architecture(seed: int = 42) -> StackingClassifier:
    """The exact architecture ``train.py`` fits.

    Random Forest, AdaBoost and k-NN feeding a logistic meta-learner over
    out-of-fold predicted probabilities.
    """
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=seed)),
            ("adaboost", AdaBoostClassifier(n_estimators=50, random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def _snapshot(paths) -> Dict[str, tuple]:
    """Size and mtime of each protected artifact, for a before/after check."""
    out = {}
    for name in paths:
        f = MODEL_DIR / name
        out[name] = (f.stat().st_size, f.stat().st_mtime_ns) if f.exists() else None
    return out


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int,
    seed: int,
    leaky: bool,
) -> Dict[str, object]:
    """One cross-validation run.

    Parameters
    ----------
    leaky:
        ``True`` reproduces ``train.py``: standardise everything first, then
        split. ``False`` puts the scaler in a ``Pipeline`` so it is refitted
        on the training part of each fold only.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    if leaky:
        estimator = build_architecture(seed)
        data = StandardScaler().fit_transform(X)
    else:
        estimator = Pipeline(
            [("scaler", StandardScaler()), ("stack", build_architecture(seed))]
        )
        data = X

    started = time.perf_counter()
    scores = cross_validate(
        estimator, data, y, cv=splitter,
        scoring=("accuracy", "f1_macro"), n_jobs=-1,
    )
    elapsed = time.perf_counter() - started

    accuracy = np.asarray(scores["test_accuracy"], dtype=float)
    macro_f1 = np.asarray(scores["test_f1_macro"], dtype=float)

    # Sample standard deviation: ddof=1. With five folds the population
    # form would understate the spread by about 11%.
    std = float(accuracy.std(ddof=1))
    half_width = _t_critical(folds - 1) * std / np.sqrt(folds)

    return {
        "leaky": leaky,
        "folds": folds,
        "seed": seed,
        "fold_accuracies": [float(v) for v in accuracy],
        "mean_accuracy": float(accuracy.mean()),
        "std_accuracy": std,
        "ci95_low": float(accuracy.mean() - half_width),
        "ci95_high": float(accuracy.mean() + half_width),
        "mean_macro_f1": float(macro_f1.mean()),
        "std_macro_f1": float(macro_f1.std(ddof=1)),
        "seconds": elapsed,
    }


def _report(result: Dict[str, object], title: str) -> None:
    pct = 100.0
    print(f"\n{title}")
    print("-" * len(title))
    folds = " ".join(f"{v * pct:6.2f}" for v in result["fold_accuracies"])
    print(f"  Fold accuracies    : {folds}")
    print(
        f"  Mean accuracy      : {result['mean_accuracy'] * pct:.2f}% "
        f"± {result['std_accuracy'] * pct:.2f} (SD)"
    )
    print(
        f"  95% CI             : "
        f"[{result['ci95_low'] * pct:.2f}%, {result['ci95_high'] * pct:.2f}%]"
    )
    print(
        f"  Macro F1           : {result['mean_macro_f1']:.4f} "
        f"± {result['std_macro_f1']:.4f}"
    )
    print(f"  Wall clock         : {result['seconds']:.1f}s")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the metrics to this file")
    parser.add_argument("--skip-leaky", action="store_true",
                        help="skip the train.py-style comparison run")
    args = parser.parse_args(argv)

    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}", file=sys.stderr)
        return 1
    if args.folds < 2:
        print("--folds must be at least 2", file=sys.stderr)
        return 1

    before = _snapshot(PROTECTED)

    frame = pd.read_csv(DATASET)
    missing = [c for c in (*FEATURES, TARGET) if c not in frame.columns]
    if missing:
        print(f"Dataset is missing columns: {missing}", file=sys.stderr)
        return 1

    X, y = frame[FEATURES], frame[TARGET]

    print("=" * 66)
    print("GREENROOT — independent stratified cross-validation")
    print("=" * 66)
    print(f"  Dataset            : {DATASET.relative_to(ROOT)}")
    print(f"  Samples × features : {X.shape[0]} × {X.shape[1]}")
    print(f"  Classes            : {y.nunique()}")
    print(f"  Smallest class     : {y.value_counts().min()} samples")
    print(f"  Splitter           : StratifiedKFold("
          f"n_splits={args.folds}, shuffle=True, random_state={args.seed})")
    print("  Architecture       : Stacking(RF + AdaBoost + kNN) → "
          "LogisticRegression")

    clean = run_cv(X, y, folds=args.folds, seed=args.seed, leaky=False)
    _report(clean, "Leak-free (scaler refitted inside each fold)")

    payload: Dict[str, object] = {"clean": clean}

    if not args.skip_leaky:
        leaky = run_cv(X, y, folds=args.folds, seed=args.seed, leaky=True)
        _report(leaky, "train.py methodology (scaled before splitting)")
        payload["leaky"] = leaky

        optimism = (leaky["mean_accuracy"] - clean["mean_accuracy"]) * 100
        print(f"\n  Leakage optimism   : {optimism:+.3f} percentage points")
        print(
            "  "
            + (
                "Immaterial on this benchmark — the features are already well "
                "separated,\n  so the fold statistics barely move. The "
                "reported figure stands."
                if abs(optimism) < 0.05
                else "Material. The headline figure should be restated from "
                "the leak-free run."
            )
        )
        payload["optimism_pp"] = optimism

    print(
        "\n  Note: the CI above is an ordinary t-interval over correlated fold\n"
        "  scores and is therefore optimistic. Use the corrected resampled\n"
        "  t-test in validate_architecture.py to compare architectures."
    )

    after = _snapshot(PROTECTED)
    if before != after:
        changed = [k for k in before if before[k] != after[k]]
        print(f"\n  ARTIFACT MUTATED: {changed}", file=sys.stderr)
        return 2
    print(f"\n  Artifacts untouched: {', '.join(PROTECTED)} ✓")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"  Metrics written to : {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
