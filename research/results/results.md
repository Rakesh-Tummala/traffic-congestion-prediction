# Historical-model research evaluation

Seeds: [42, 43, 44]. Chronological split 70/15/15; the test rows are the last 15% of the data. Congestion-level edges come from the training rows only. Recurrent-model rows are the average prediction of the seeds (an ensemble); the seed spread is listed separately.

## 1. Main comparison (same test rows for every model)

| Model | MAE | RMSE | R2 | LevelAcc | Within1 | MacroF1 |
|---|---|---|---|---|---|---|
| Persistence (last hour) | 590.034 | 815.573 | 0.831 | 0.722 | 0.973 | 0.722 |
| Seasonal naive (24 h ago) | 560.074 | 1,029 | 0.731 | 0.745 | 0.958 | 0.747 |
| Seasonal naive (1 week ago) | 322.170 | 626.152 | 0.900 | 0.843 | 0.988 | 0.845 |
| Hour x weekday mean | 261.359 | 448.869 | 0.949 | 0.879 | 0.996 | 0.881 |
| LinearRegression | 334.446 | 452.169 | 0.948 | 0.849 | 1.000 | 0.851 |
| SVR | 254.077 | 399.809 | 0.959 | 0.876 | 0.998 | 0.879 |
| RandomForest | 141.143 | 217.677 | 0.988 | 0.926 | 1.000 | 0.927 |
| GradientBoosting | 142.297 | 216.567 | 0.988 | 0.926 | 1.000 | 0.927 |
| LSTM (mean of 3 seeds) | 183.801 | 275.852 | 0.981 | 0.912 | 1.000 | 0.913 |
| GRU (mean of 3 seeds) | 176.055 | 273.759 | 0.981 | 0.913 | 1.000 | 0.914 |

Recurrent-model seed spread:

| Model | seeds | MAE_mean | MAE_std | R2_mean | R2_std |
|---|---|---|---|---|---|
| LSTM | 3 | 191.481 | 3.108 | 0.979 | 0.000 |
| GRU | 3 | 185.751 | 4.880 | 0.979 | 0.001 |

## 2. Moving-block bootstrap (24 h blocks, 2,000 resamples)

| Model | MAE | MAE_CI_low | MAE_CI_high | dMAE_vs_persistence | dMAE_CI_low | dMAE_CI_high | significant |
|---|---|---|---|---|---|---|---|
| Persistence (last hour) | 590.034 | 573.748 | 606.281 | nan | nan | nan | nan |
| Seasonal naive (24 h ago) | 560.074 | 506.414 | 609.641 | -29.961 | -92.742 | 27.047 | False |
| Seasonal naive (1 week ago) | 322.170 | 283.560 | 365.182 | -267.864 | -315.038 | -216.046 | True |
| Hour x weekday mean | 261.359 | 237.182 | 289.188 | -328.675 | -361.656 | -290.641 | True |
| LinearRegression | 334.446 | 325.310 | 344.248 | -255.589 | -269.592 | -240.842 | True |
| SVR | 254.077 | 241.515 | 267.527 | -335.958 | -355.563 | -315.280 | True |
| RandomForest | 141.143 | 135.493 | 147.043 | -448.891 | -466.773 | -431.399 | True |
| GradientBoosting | 142.297 | 136.967 | 147.811 | -447.737 | -465.973 | -430.580 | True |
| LSTM (mean of 3 seeds) | 183.801 | 174.590 | 192.539 | -406.233 | -426.636 | -385.763 | True |
| GRU (mean of 3 seeds) | 176.055 | 166.916 | 185.139 | -413.979 | -434.689 | -393.682 | True |

## 3. Feature ablation

| Features | Model | n_features | MAE | RMSE | R2 |
|---|---|---|---|---|---|
| All features | LinearRegression | 17 | 334.437 | 452.184 | 0.948 |
| All features | GradientBoosting | 17 | 142.508 | 217.070 | 0.988 |
| All features | SVR | 17 | 254.131 | 399.838 | 0.959 |
| No lag features | LinearRegression | 13 | 835.166 | 1,069 | 0.709 |
| No lag features | GradientBoosting | 13 | 245.785 | 431.958 | 0.952 |
| No lag features | SVR | 13 | 553.577 | 825.730 | 0.826 |
| Lag features only | LinearRegression | 4 | 368.174 | 503.276 | 0.935 |
| Lag features only | GradientBoosting | 4 | 192.375 | 287.040 | 0.979 |
| Lag features only | SVR | 4 | 273.784 | 409.630 | 0.957 |
| Calendar only | LinearRegression | 8 | 837.278 | 1,070 | 0.708 |
| Calendar only | GradientBoosting | 8 | 242.111 | 434.088 | 0.952 |
| Calendar only | SVR | 8 | 452.573 | 690.398 | 0.878 |
| Weather only | LinearRegression | 5 | 1,695 | 1,951 | 0.029 |
| Weather only | GradientBoosting | 5 | 1,623 | 1,891 | 0.088 |
| Weather only | SVR | 5 | 1,660 | 1,933 | 0.047 |
| No weather | LinearRegression | 12 | 334.656 | 452.795 | 0.948 |
| No weather | GradientBoosting | 12 | 142.495 | 218.744 | 0.988 |
| No weather | SVR | 12 | 215.977 | 350.713 | 0.969 |
| No calendar | LinearRegression | 9 | 368.365 | 503.362 | 0.935 |
| No calendar | GradientBoosting | 9 | 194.402 | 289.873 | 0.979 |
| No calendar | SVR | 9 | 337.785 | 507.616 | 0.934 |

## 4. Forecast horizon (targets are real observations only)

| Horizon_h | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| 1 | Persistence | 591.718 | 817.859 | 0.830 |
| 1 | Seasonal naive (24 h) | 551.892 | 1,022 | 0.735 |
| 1 | Hour x weekday mean | 257.850 | 435.760 | 0.952 |
| 1 | LinearRegression | 335.027 | 451.621 | 0.948 |
| 1 | GradientBoosting | 139.520 | 214.032 | 0.988 |
| 3 | Persistence | 1,507 | 1,966 | 0.021 |
| 3 | Seasonal naive (24 h) | 548.998 | 1,018 | 0.737 |
| 3 | Hour x weekday mean | 257.902 | 435.261 | 0.952 |
| 3 | LinearRegression | 754.323 | 989.678 | 0.752 |
| 3 | GradientBoosting | 188.049 | 296.685 | 0.978 |
| 6 | Persistence | 2,412 | 2,864 | -1.085 |
| 6 | Seasonal naive (24 h) | 556.146 | 1,027 | 0.732 |
| 6 | Hour x weekday mean | 258.406 | 434.695 | 0.952 |
| 6 | LinearRegression | 762.978 | 1,007 | 0.742 |
| 6 | GradientBoosting | 219.486 | 357.374 | 0.968 |
| 12 | Persistence | 3,260 | 3,571 | -2.230 |
| 12 | Seasonal naive (24 h) | 554.057 | 1,026 | 0.733 |
| 12 | Hour x weekday mean | 257.160 | 434.372 | 0.952 |
| 12 | LinearRegression | 828.128 | 1,044 | 0.724 |
| 12 | GradientBoosting | 225.550 | 391.874 | 0.961 |
| 24 | Persistence | 553.644 | 1,025 | 0.733 |
| 24 | Seasonal naive (24 h) | 553.644 | 1,025 | 0.733 |
| 24 | Hour x weekday mean | 258.827 | 435.858 | 0.952 |
| 24 | LinearRegression | 555.725 | 800.545 | 0.837 |
| 24 | GradientBoosting | 221.657 | 396.074 | 0.960 |

## 5. Expanding-window time-series CV (5 folds, first 85% of the data)

| Model | MAE_mean | MAE_std | RMSE_mean | R2_mean | R2_std |
|---|---|---|---|---|---|
| GradientBoosting | 168.480 | 12.280 | 261.228 | 0.982 | 0.002 |
| LinearRegression | 351.277 | 6.724 | 506.817 | 0.934 | 0.012 |
| Persistence | 577.052 | 11.029 | 814.175 | 0.830 | 0.008 |
| RandomForest | 166.094 | 11.027 | 263.626 | 0.982 | 0.002 |
| SVR | 299.592 | 32.418 | 467.813 | 0.943 | 0.013 |
| Seasonal naive (24 h) | 589.741 | 44.444 | 1,056 | 0.714 | 0.030 |

## 6. Split-conformal prediction intervals (models retrained here; the app uses research/conformal_intervals.py)

| Model | nominal | half_width | test_coverage |
|---|---|---|---|
| LinearRegression | 0.800 | 546.673 | 0.828 |
| LinearRegression | 0.900 | 787.048 | 0.910 |
| GradientBoosting | 0.800 | 230.597 | 0.812 |
| GradientBoosting | 0.900 | 338.958 | 0.914 |
| LSTM | 0.800 | 299.282 | 0.819 |
| LSTM | 0.900 | 422.693 | 0.915 |
