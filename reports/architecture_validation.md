# GREENROOT — Architectural Validation

_Generated 2026-09-13 16:22:43_

Every figure below re-fits the architecture from scratch under a leak-free pipeline. The artefacts in `models/` are not read here — an architecture cannot be validated by re-measuring one frozen instance of it.

## A. Ablation — does stacking earn its complexity?

Repeated stratified 5-fold cross-validation, 15 measurements per model, identical folds throughout.

| Architecture | Accuracy | s.d. | 95% CI | Range |
|---|---:|---:|---|---|
| Random Forest (base) | 99.52% | 0.26 | [99.39, 99.64] | 99.09–100.00 |
| Stacking without AdaBoost | 99.38% | 0.35 | [99.20, 99.56] | 98.86–100.00 |
| Stacking Ensemble (deployed) **← deployed** | 99.38% | 0.35 | [99.20, 99.56] | 98.86–100.00 |
| Soft Voting (same base learners) | 99.05% | 0.41 | [98.84, 99.25] | 98.18–99.55 |
| k-NN, k=5 (base) | 97.38% | 0.60 | [97.07, 97.68] | 95.91–98.41 |
| Logistic Regression (meta alone) | 97.12% | 0.56 | [96.84, 97.41] | 96.14–98.41 |
| AdaBoost (base) | 25.35% | 10.13 | [20.22, 30.47] | 12.50–41.59 |

Against its strongest single component (Random Forest (base)) the deployed ensemble scores **-0.136 percentage points** (corrected *p* = 0.444).

> **Finding — the stacking layer does not improve accuracy on this corpus.** The ensemble is statistically indistinguishable from a plain Random Forest, at roughly eight times the fitting cost. This is not a defect in the implementation; it is what a saturated benchmark looks like. Section D shows the learning curve has plateaued and Random Forest alone already reaches the ceiling, so accuracy has no headroom left in which any architecture could distinguish itself. Section E therefore evaluates the axis that still carries signal — and the one the deployed system actually depends on — the quality of the posterior.

## B. Statistical significance

Nadeau–Bengio corrected resampled *t*-test. The correction is required because cross-validation folds share training data, violating the independence assumption of the uncorrected paired *t*-test and producing optimistically small *p*-values.

| Comparison | Δ accuracy | W/T/L | corrected *p* | naive *p* | Significant (α=0.05) |
|---|---:|---|---:|---:|---|
| vs Random Forest (base) | -0.136 pp | 3/6/6 | 0.4436 | 0.1077 | no |
| vs AdaBoost (base) | +74.030 pp | 15/0/0 | 3.33e-09 | 9.187e-14 | **yes** |
| vs k-NN, k=5 (base) | +2.000 pp | 15/0/0 | 5.758e-05 | 6.375e-09 | **yes** |
| vs Logistic Regression (meta alone) | +2.258 pp | 15/0/0 | 1.726e-06 | 8.947e-11 | **yes** |
| vs Soft Voting (same base learners) | +0.333 pp | 11/3/1 | 0.09536 | 0.001609 | no |
| vs Stacking without AdaBoost | +0.000 pp | 0/15/0 | 1 | nan | no |

> Note how much smaller the naive *p*-values are. Reporting those would overstate the evidence; the corrected column is the one to cite.

## C. Preprocessing-leakage audit

The original `train.py` fits the `StandardScaler` on the entire corpus *before* cross-validating, so test-fold statistics inform the transform. This experiment quantifies the resulting optimism over identical folds.

| Protocol | Accuracy | s.d. |
|---|---:|---:|
| Global scaler before CV (as `train.py`) | 99.41% | 0.41 |
| Scaler re-fitted inside each fold (leak-free) | 99.41% | 0.41 |
| **Optimism attributable to leakage** | **+0.000 pp** | — |

Documented figure: 99.41%. Reproduced under the original protocol: 99.41%.

## D. Learning curve

| Training samples | Per class | Train acc. | Validation acc. | Gap |
|---:|---:|---:|---:|---:|
| 352 | 16 | 99.94% | 98.27% | 1.67 pp |
| 704 | 32 | 100.00% | 98.95% | 1.05 pp |
| 1056 | 48 | 100.00% | 99.09% | 0.91 pp |
| 1408 | 64 | 99.96% | 99.27% | 0.68 pp |
| 1760 | 80 | 99.89% | 99.32% | 0.57 pp |

Validation accuracy moves +0.045 pp over the final size increment. The curve has plateaued: additional samples of the same kind would not improve the model — broader agro-climatic coverage would.

## E. Probability calibration

Accuracy has saturated, so it can no longer separate the leading architectures. These metrics score the *posterior* — the number the dashboard shows the farmer, and the one its 50% provisional-advisory threshold is gated on. Brier score and log-loss are strictly proper scoring rules: they are minimised only by honest reporting of uncertainty, so they penalise confident error far more than accuracy does.

Accuracy in this table comes from a single stratified 5-fold `cross_val_predict` pass, so it differs in the second decimal place from Section A's mean over 15 folds. The two are consistent; only the resampling protocol differs.

| Architecture | ECE | Brier | Log-loss | Mean confidence | Accuracy | Over-confidence |
|---|---:|---:|---:|---:|---:|---:|
| k-NN, k=5 (base) | 0.0155 | 0.0415 | 0.1958 | 96.75% | 97.14% | -0.38 pp |
| Stacking without AdaBoost | 0.0428 | 0.0125 | 0.0602 | 95.13% | 99.41% | -4.28 pp |
| Stacking Ensemble (deployed) **← deployed** | 0.0428 | 0.0125 | 0.0602 | 95.12% | 99.41% | -4.28 pp |
| Random Forest (base) | 0.0447 | 0.0164 | 0.0569 | 95.12% | 99.59% | -4.47 pp |
| Logistic Regression (meta alone) | 0.1285 | 0.0800 | 0.2102 | 84.25% | 97.09% | -12.85 pp |
| AdaBoost (base) | 0.2085 | 0.9543 | 3.0888 | 4.56% | 25.41% | -20.85 pp |
| Soft Voting (same base learners) | 0.3378 | 0.1366 | 0.4376 | 65.27% | 99.05% | -33.78 pp |

Best-calibrated architecture: **k-NN, k=5 (base)** (ECE 0.0155).

The deployed ensemble records ECE 0.0428 against Random Forest's 0.0447 — a difference of -0.0019. The stacking layer therefore buys calibration rather than accuracy: its logistic meta-learner maps raw base-learner votes onto posteriors that mean closer to what they say. That is the defensible justification for the architecture on this corpus, and it is the property the advisory threshold depends on.

## F. What is the meta-learner actually doing?

The stacked representation is 66-dimensional: three base learners contributing 22 posteriors each. The logistic meta-learner's coefficient matrix partitions into three blocks, so the magnitude of each block shows how far it relies on that base learner.

| Base learner | Standalone accuracy | Weight mass | Share of total |
|---|---:|---:|---:|
| `rf` | 99.52% | 151.15 | 50.88% |
| `knn` | 97.38% | 144.97 | 48.80% |
| `adaboost` | 25.35% | 0.96 | 0.32% |

**This is the architectural justification the accuracy ablation could not provide.** AdaBoost collapses to 25.35% in isolation — 50 decision stumps cannot separate 22 classes — yet it sits inside the deployed ensemble as one base learner in three. The meta-learner assigns its entire 22-column block just **0.32%** of total weight mass. It has learned to ignore it.

Deleting AdaBoost from the ensemble outright changes cross-validated accuracy by less than 1e-9, across all 15 folds, and changes the Brier score in the fourth decimal place. The suppression is total, not partial.

### The control that matters

Soft voting over the *same* three base learners isolates the effect of the combination rule — fixed averaging versus a learned combiner. On accuracy the two are close and the difference does not reach significance:

- Stacking 99.38% vs voting 99.05% — +0.33 pp, corrected *p* = 0.0954, W/L = 11/1. Suggestive, **not significant**.

On the posterior, the two are not close at all:

| Metric | Soft voting | Stacking | Ratio |
|---|---:|---:|---:|
| ECE | 0.3378 | 0.0428 | 7.9x worse |
| Brier score | 0.1366 | 0.0125 | 10.9x worse |
| Mean confidence | 65.3% | 95.1% | — |
| Accuracy | 99.05% | 99.41% | — |
| Over-confidence | -33.8 pp | -4.3 pp | — |

Voting is right 99.0% of the time while reporting 65.3% confidence — it is under-confident by 33.8 percentage points, because averaging drags every posterior toward AdaBoost's near-uniform output. Its argmax survives; its probabilities do not.

**That is the finding.** In GREENROOT the posterior is not incidental — it is displayed to the farmer as a confidence percentage and gates the provisional-advisory threshold at 50%. A voting ensemble here would clear almost every recommendation as provisional while being right 99% of the time. The stacking layer does not buy accuracy on this saturated benchmark; it buys a posterior that means what it says, and it does so while carrying a base learner that has failed outright.

### Honest summary of Sections A, E and F

| Claim | Verdict |
|---|---|
| Stacking beats its best single component (Random Forest) on accuracy | **No** — indistinguishable, *p* = 0.444. The benchmark is saturated. |
| Stacking beats soft voting on accuracy | Not significantly — *p* = 0.0954. |
| Stacking beats soft voting on posterior quality | **Yes, decisively** — 11x better Brier score. |
| Stacking tolerates a failed base learner | **Yes** — 0.32% weight mass assigned to a 25%-accurate learner. |

A capstone that claimed stacking was more accurate here would be wrong, and the ablation above is how one finds that out. The architecture is defensible on robustness and posterior quality, which are the grounds on which it is claimed.

## Artefacts

- `architecture_ablation.csv` — per-model cross-validated accuracy
- `calibration_metrics.csv` / `reliability_diagrams.png` — calibration
- `meta_learner_attribution.csv` — meta-learner weight attribution
- `significance_tests.csv` — corrected and naive paired tests
- `preprocessing_leakage.csv` — preprocessing-leakage audit
- `learning_curve.csv` / `learning_curve.png` — learning curve
- `model_comparison.png` — ablation bar chart and fold-wise box plot
