# GREENROOT — System Evaluation Summary

_Generated 2026-09-13 15:24:49_

## 1. Classification Performance

Stratified 20% hold-out of `Crop_recommendation.csv` (2,200 samples, 22 classes).

| Metric | Value |
|---|---:|
| Reported 5-fold CV accuracy | 99.41% |
| Hold-out accuracy | 100.00% |
| Top-3 accuracy | 100.00% |
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |
| Cohen's kappa | 1.0000 |
| Matthews correlation | 1.0000 |
| Mean max posterior | 0.9675 |

> The shipped ensemble was fitted on the full corpus, so the hold-out figure is a reproduction check rather than an independent generalisation estimate. The stratified 5-fold cross-validation accuracy is the figure to cite.

## 2. Inference Latency

100 stochastic single-sample inferences, end-to-end (validation + scaling + stacked forward pass).

| Statistic | Latency (ms) |
|---|---:|
| Mean | 16.406 |
| P50 | 16.194 |
| P95 | 17.029 |
| P99 | 21.156 |
| Max | 27.546 |

Sustained throughput: **61 inferences/s** single-threaded.

## 3. Out-of-Distribution Robustness

Karnataka NFSM field survey rows, which carry genuine covariate shift relative to the benchmark corpus.

| Metric | Value |
|---|---:|
| Field samples | 500 |
| Mean confidence (benchmark) | 0.9645 |
| Mean confidence (field) | 0.4143 |
| Confidence degradation | 0.5502 |
| Mean \|Z\| under shift | 2.517 |
| OOD detection rate | 100.0% |

## 4. Multi-Explainer Consensus

TreeSHAP versus LIME top-3 driver agreement, Jaccard index.

| Metric | Value |
|---|---:|
| Instances audited | 30 |
| Surrogate fidelity vs. ensemble | 0.9955 |
| Mean Jaccard index | 0.6933 |
| Median Jaccard index | 0.5000 |
| Perfect agreement (J = 1.0) | 46.7% |
| High fidelity (J >= 0.5) | 86.7% |
| Mean explanation latency | 119.3 ms |

## Artefacts

- `confusion_matrix.png` — 300 DPI confusion matrix
- `classification_report.csv` — per-class precision/recall/F1
- `latency_benchmark.csv` — latency distribution
- `ood_robustness.csv` — per-feature covariate shift
