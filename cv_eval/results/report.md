# CV evaluation results

Usable labelled frames: 124 (class counts: Low 92, Moderate 22, High 8, Severe 2)

## 1. Vehicle counting vs hand counts (sorted by MAE)

`within` = share of frames where the count is within max(2, 25%) of the hand count.

| model | imgsz | conf | MAE | RMSE | bias | within | r |
|---|---|---|---|---|---|---|---|
| yolov8m | 640 | 0.08 | 4.15 | 7.18 | -2.57 | 0.64 | 0.88 |
| yolov8m | 640 | 0.05 | 4.23 | 6.58 | 0.02 | 0.60 | 0.88 |
| yolov8s+tiled | 640 | 0.15 | 4.32 | 7.67 | -2.48 | 0.61 | 0.85 |
| yolov8m | 1280 | 0.08 | 4.34 | 7.34 | -2.11 | 0.57 | 0.86 |
| yolov8m | 640 | 0.10 | 4.52 | 8.14 | -3.64 | 0.63 | 0.87 |
| yolov8s | 640 | 0.08 | 4.55 | 7.88 | -2.52 | 0.59 | 0.84 |
| yolov8s+tiled | 640 | 0.10 | 4.58 | 7.07 | -0.06 | 0.60 | 0.87 |
| yolov8m | 1280 | 0.05 | 4.65 | 7.28 | 0.99 | 0.65 | 0.87 |
| yolov8m | 1280 | 0.10 | 4.66 | 7.91 | -3.19 | 0.59 | 0.85 |
| yolov8s | 640 | 0.10 | 4.85 | 8.63 | -3.66 | 0.60 | 0.83 |
| yolov8s+tiled | 640 | 0.08 | 4.98 | 7.54 | 1.17 | 0.51 | 0.87 |
| yolov8n | 640 | 0.08 | 5.04 | 8.72 | -2.62 | 0.52 | 0.81 |
| yolov8s | 1280 | 0.10 | 5.13 | 8.63 | -2.76 | 0.60 | 0.81 |
| yolov8s | 640 | 0.05 | 5.15 | 7.89 | 0.28 | 0.55 | 0.84 |
| yolov8s | 1280 | 0.08 | 5.25 | 8.52 | -1.38 | 0.58 | 0.81 |
| yolov8n | 640 | 0.10 | 5.32 | 9.07 | -3.95 | 0.49 | 0.81 |
| yolov8s+tiled | 640 | 0.25 | 5.65 | 9.68 | -4.98 | 0.50 | 0.81 |
| yolov8m | 1280 | 0.15 | 5.66 | 9.35 | -5.06 | 0.52 | 0.85 |
| yolov8m | 640 | 0.15 | 5.69 | 10.04 | -5.36 | 0.52 | 0.84 |
| yolov8s | 640 | 0.15 | 5.77 | 9.98 | -5.23 | 0.51 | 0.82 |
| yolov8s | 1280 | 0.15 | 5.84 | 9.64 | -4.79 | 0.49 | 0.80 |
| yolov8n | 640 | 0.05 | 5.94 | 10.35 | 0.35 | 0.53 | 0.81 |
| yolov8n | 1280 | 0.10 | 6.12 | 10.43 | -4.02 | 0.47 | 0.75 |
| yolov8s+tiled | 640 | 0.05 | 6.21 | 9.61 | 3.92 | 0.48 | 0.88 |
| yolov8n | 640 | 0.15 | 6.23 | 10.19 | -5.90 | 0.44 | 0.81 |
| yolov8n | 1280 | 0.08 | 6.48 | 10.55 | -2.76 | 0.44 | 0.75 |
| yolov8n | 1280 | 0.15 | 6.85 | 11.21 | -6.11 | 0.46 | 0.73 |
| yolov8s | 1280 | 0.05 | 6.96 | 10.75 | 2.72 | 0.47 | 0.82 |
| yolov8m | 640 | 0.25 | 7.03 | 11.87 | -6.98 | 0.44 | 0.81 |
| yolov8s+tiled | 640 | 0.35 | 7.05 | 11.50 | -6.82 | 0.40 | 0.79 |
| yolov8s | 1280 | 0.25 | 7.14 | 11.58 | -6.91 | 0.43 | 0.78 |
| yolov8m | 1280 | 0.25 | 7.19 | 11.61 | -6.92 | 0.42 | 0.80 |
| yolov8s | 640 | 0.25 | 7.43 | 12.05 | -7.28 | 0.39 | 0.81 |
| yolov8n | 1280 | 0.05 | 7.49 | 12.61 | 0.48 | 0.46 | 0.76 |
| yolov8n | 640 | 0.25 | 7.82 | 12.37 | -7.71 | 0.37 | 0.78 |
| yolov8m | 640 | 0.35 | 8.12 | 13.29 | -8.12 | 0.38 | 0.78 |
| yolov8n | 1280 | 0.25 | 8.23 | 12.85 | -8.13 | 0.37 | 0.72 |
| yolov8m | 1280 | 0.35 | 8.25 | 13.32 | -8.19 | 0.38 | 0.74 |
| yolov8s | 1280 | 0.35 | 8.46 | 13.36 | -8.35 | 0.35 | 0.74 |
| yolov8s | 640 | 0.35 | 8.47 | 13.53 | -8.44 | 0.37 | 0.77 |
| yolov8n | 640 | 0.35 | 9.04 | 14.03 | -9.01 | 0.34 | 0.75 |
| yolov8n | 1280 | 0.35 | 9.35 | 14.35 | -9.34 | 0.33 | 0.70 |

### Before tuning: yolov8s.pt @ 640px, conf 0.25

Count MAE by true count: 0-5: 1.55 (n=58), 6-20: 7.16 (n=44), 21+: 23.45 (n=22)

### Shipped setting (full frame): yolov8s.pt @ 640px, conf 0.08

Count MAE by true count: 0-5: 1.41 (n=58), 6-20: 4.66 (n=44), 21+: 12.59 (n=22)

### Best sliced (tiled) setting: yolov8s+tiled.pt @ 640px, conf 0.15

Count MAE by true count: 0-5: 1.40 (n=58), 6-20: 4.30 (n=44), 21+: 12.09 (n=22)

### Best setting by MAE: yolov8m.pt @ 640px, conf 0.08

Count MAE by true count: 0-5: 1.26 (n=58), 6-20: 3.82 (n=44), 21+: 12.45 (n=22)

## 2. Congestion label vs hand-labelled class

Always answering the most common class (Low) scores accuracy 0.74; compare balanced accuracy (mean recall per class) and macro-F1, which that shortcut cannot game.

### Before tuning: yolov8s.pt @ 640px, conf 0.25

| feature | accuracy | balanced acc | macro-F1 | off by 2+ |
|---|---|---|---|---|
| density (fixed 0.05/0.15/0.30) | 0.73 | 0.32 | 0.33 | 0.02 |
| density (fitted, CV) | 0.58 | 0.50 | 0.40 | 0.07 |
| count (fitted, CV) | 0.63 | 0.46 | 0.39 | 0.05 |

Thresholds fitted on all frames for density (fitted, CV): (0.004, 0.045, 0.148) (shipped numbers above are the cross-validated ones)

Thresholds fitted on all frames for count (fitted, CV): (2.0, 12.754, 19.863) (shipped numbers above are the cross-validated ones)

Confusion matrix, fixed density thresholds (rows = hand label):

| true | pred Low | pred Moderate | pred High | pred Severe |
|---|---|---|---|---|
| Low | 85 | 7 | 0 | 0 |
| Moderate | 17 | 5 | 0 | 0 |
| High | 2 | 5 | 1 | 0 |
| Severe | 0 | 1 | 1 | 0 |

### Shipped setting (full frame): yolov8s.pt @ 640px, conf 0.08

| feature | accuracy | balanced acc | macro-F1 | off by 2+ |
|---|---|---|---|---|
| density (fixed 0.05/0.15/0.30) | 0.75 | 0.62 | 0.64 | 0.02 |
| density (fitted, CV) | 0.69 | 0.62 | 0.54 | 0.03 |
| count (fitted, CV) | 0.67 | 0.47 | 0.39 | 0.05 |

Thresholds fitted on all frames for density (fitted, CV): (0.022, 0.134, 0.261) (shipped numbers above are the cross-validated ones)

Thresholds fitted on all frames for count (fitted, CV): (6.353, 18.0, 45.31) (shipped numbers above are the cross-validated ones)

Confusion matrix, fixed density thresholds (rows = hand label):

| true | pred Low | pred Moderate | pred High | pred Severe |
|---|---|---|---|---|
| Low | 77 | 14 | 1 | 0 |
| Moderate | 11 | 9 | 2 | 0 |
| High | 1 | 1 | 6 | 0 |
| Severe | 0 | 0 | 1 | 1 |

### Best sliced (tiled) setting: yolov8s+tiled.pt @ 640px, conf 0.15

| feature | accuracy | balanced acc | macro-F1 | off by 2+ |
|---|---|---|---|---|
| density (fixed 0.05/0.15/0.30) | 0.75 | 0.60 | 0.60 | 0.03 |
| density (fitted, CV) | 0.69 | 0.53 | 0.47 | 0.05 |
| count (fitted, CV) | 0.72 | 0.51 | 0.43 | 0.07 |

Thresholds fitted on all frames for density (fitted, CV): (0.042, 0.123, 0.309) (shipped numbers above are the cross-validated ones)

Thresholds fitted on all frames for count (fitted, CV): (9.369, 22.416, 53.7) (shipped numbers above are the cross-validated ones)

Confusion matrix, fixed density thresholds (rows = hand label):

| true | pred Low | pred Moderate | pred High | pred Severe |
|---|---|---|---|---|
| Low | 77 | 14 | 1 | 0 |
| Moderate | 9 | 10 | 2 | 1 |
| High | 1 | 2 | 5 | 0 |
| Severe | 0 | 1 | 0 | 1 |

### Best setting by MAE: yolov8m.pt @ 640px, conf 0.08

| feature | accuracy | balanced acc | macro-F1 | off by 2+ |
|---|---|---|---|---|
| density (fixed 0.05/0.15/0.30) | 0.76 | 0.57 | 0.62 | 0.01 |
| density (fitted, CV) | 0.62 | 0.50 | 0.42 | 0.04 |
| count (fitted, CV) | 0.76 | 0.55 | 0.50 | 0.03 |

Thresholds fitted on all frames for density (fitted, CV): (0.018, 0.095, 0.146) (shipped numbers above are the cross-validated ones)

Thresholds fitted on all frames for count (fitted, CV): (11.707, 22.0, 44.77) (shipped numbers above are the cross-validated ones)

Confusion matrix, fixed density thresholds (rows = hand label):

| true | pred Low | pred Moderate | pred High | pred Severe |
|---|---|---|---|---|
| Low | 80 | 12 | 0 | 0 |
| Moderate | 12 | 9 | 1 | 0 |
| High | 1 | 3 | 4 | 0 |
| Severe | 0 | 0 | 1 | 1 |
