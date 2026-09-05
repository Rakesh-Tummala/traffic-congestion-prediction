"""One-off hyperparameter search for SVR. Run from the notebooks/ dir with the
project venv active. Prints the best params found; train_regression.py hardcodes
the result so training doesn't re-run this search every time."""
import os
import sys

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, train_val_test_split  # noqa: E402


def main():
    df, _, _ = load_dataset()
    train, _, _ = train_val_test_split(df)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    idx = np.random.RandomState(42).choice(len(X_train_s), min(4000, len(X_train_s)), replace=False)

    param_dist = {
        "C": [1, 10, 50, 100, 200],
        "epsilon": [10, 25, 50, 100],
        "gamma": ["scale", "auto", 0.01, 0.05],
    }
    search = RandomizedSearchCV(
        SVR(kernel="rbf"), param_dist, n_iter=12, cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
    )
    search.fit(X_train_s[idx], y_train.iloc[idx])
    print("Best params:", search.best_params_)
    print("Best CV RMSE:", -search.best_score_)


if __name__ == "__main__":
    main()
