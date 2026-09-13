# GREENROOT — System Evaluation Summary

_Generated 2026-09-13 17:14:42_

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

20 stochastic single-sample inferences, end-to-end (validation + scaling + stacked forward pass).

| Statistic | Latency (ms) |
|---|---:|
| Mean | 15.298 |
| P50 | 15.267 |
| P95 | 15.930 |
| P99 | 16.163 |
| Max | 16.222 |

Sustained throughput: **65 inferences/s** single-threaded.

## 3. Out-of-Distribution Robustness

Karnataka NFSM field survey rows, which carry genuine covariate shift relative to the benchmark corpus.

| Metric | Value |
|---|---:|
| Field samples | 100 |
| Mean confidence (benchmark) | 0.9677 |
| Mean confidence (field) | 0.4159 |
| Confidence degradation | 0.5517 |
| Mean \|Z\| under shift | 2.513 |
| OOD detection rate | 100.0% |

## Artefacts

- `confusion_matrix.png` — 300 DPI confusion matrix
- `classification_report.csv` — per-class precision/recall/F1
- `latency_benchmark.csv` — latency distribution
- `ood_robustness.csv` — per-feature covariate shift
