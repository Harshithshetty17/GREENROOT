# GREENROOT — System Evaluation Summary

_Generated 2026-09-14 04:18:05_

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

15 stochastic single-sample inferences, end-to-end (validation + scaling + stacked forward pass).

| Statistic | Latency (ms) |
|---|---:|
| Mean | 15.506 |
| P50 | 15.369 |
| P95 | 16.678 |
| P99 | 16.889 |
| Max | 16.942 |

Sustained throughput: **64 inferences/s** single-threaded.

## 3. Out-of-Distribution Robustness

Karnataka NFSM field survey rows, which carry genuine covariate shift relative to the benchmark corpus.

| Metric | Value |
|---|---:|
| Field samples | 60 |
| Mean confidence (benchmark) | 0.9712 |
| Mean confidence (field) | 0.4152 |
| Confidence degradation | 0.5560 |
| Mean \|Z\| under shift | 2.472 |
| OOD detection rate | 100.0% |

## Artefacts

- `confusion_matrix.png` — 300 DPI confusion matrix
- `classification_report.csv` — per-class precision/recall/F1
- `latency_benchmark.csv` — latency distribution
- `ood_robustness.csv` — per-feature covariate shift
