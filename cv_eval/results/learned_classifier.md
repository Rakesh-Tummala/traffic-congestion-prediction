# Learned congestion classifier vs fixed thresholds

124 usable frames; classes Low 92, Moderate 22, High 8, Severe 2
Repeated 5-fold cross-validation x10 (out-of-sample).

## Detections: full frame, conf 0.08

| model | accuracy | balanced_acc | macro_f1 | std (bal. acc) |
|---|---|---|---|---|
| fixed density thresholds (shipped) | 0.75 | 0.62 | 0.64 | 0.00 |
| logistic regression | 0.74 | 0.54 | 0.50 | 0.07 |
| random forest | 0.75 | 0.50 | 0.46 | 0.02 |
| gradient boosting | 0.75 | 0.41 | 0.41 | 0.03 |

## Detections: sliced (tiled), conf 0.08

| model | accuracy | balanced_acc | macro_f1 | std (bal. acc) |
|---|---|---|---|---|
| fixed density thresholds (shipped) | 0.74 | 0.62 | 0.59 | 0.00 |
| logistic regression | 0.74 | 0.50 | 0.47 | 0.08 |
| random forest | 0.75 | 0.49 | 0.46 | 0.05 |
| gradient boosting | 0.74 | 0.39 | 0.39 | 0.04 |
