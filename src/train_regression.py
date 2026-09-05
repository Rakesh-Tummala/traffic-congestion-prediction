"""Train and evaluate Linear Regression and SVR baselines on traffic volume."""
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, seasonal_profile, train_val_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>18s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:6.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df, _, _ = load_dataset()
    train, val, test = train_val_test_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n")

    lr = LinearRegression().fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_metrics = evaluate("LinearRegression", y_test, lr_pred)

    # SVR is O(n^2)-ish; subsample for training speed, evaluate on full test set.
    # Hyperparameters chosen via RandomizedSearchCV with TimeSeriesSplit CV (see
    # notebooks/tune_svr.py) over C/epsilon/gamma on a 4000-row subsample.
    svr_sample = min(8000, len(X_train_s))
    idx = np.random.RandomState(42).choice(len(X_train_s), svr_sample, replace=False)
    svr = SVR(kernel="rbf", C=100, epsilon=10, gamma="auto")
    svr.fit(X_train_s[idx], y_train.iloc[idx])
    svr_pred = svr.predict(X_test_s)
    svr_metrics = evaluate("SVR (rbf)", y_test, svr_pred)

    joblib.dump({"model": lr, "scaler": scaler, "features": FEATURE_COLUMNS}, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    joblib.dump({"model": svr, "scaler": scaler, "features": FEATURE_COLUMNS}, os.path.join(MODELS_DIR, "svr.joblib"))
    joblib.dump(seasonal_profile(train), os.path.join(MODELS_DIR, "seasonal_profile.joblib"))
    print(f"\nSaved models to {MODELS_DIR}")

    return {"linear_regression": lr_metrics, "svr": svr_metrics}


if __name__ == "__main__":
    main()
