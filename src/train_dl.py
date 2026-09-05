"""Train an LSTM to forecast traffic volume from a sliding window of past hours."""
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, train_val_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEQ_LEN = 24  # look back 24 hours to predict the next hour's volume
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main(epochs: int = 40, batch_size: int = 128, lr_rate: float = 1e-3, patience: int = 5):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df, _, _ = load_dataset()
    train, val, test = train_val_test_split(df)

    x_scaler = StandardScaler().fit(train[FEATURE_COLUMNS])
    y_scaler = StandardScaler().fit(train[[TARGET_COLUMN]])

    def prep(split):
        X = x_scaler.transform(split[FEATURE_COLUMNS]).astype(np.float32)
        y = y_scaler.transform(split[[TARGET_COLUMN]]).astype(np.float32).ravel()
        return make_sequences(X, y, SEQ_LEN)

    X_train, y_train = prep(train)
    X_val, y_val = prep(val)
    X_test, y_test = prep(test)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    model = TrafficLSTM(n_features=len(FEATURE_COLUMNS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss_fn = nn.MSELoss()

    print(f"Device: {DEVICE}  Train sequences: {len(X_train)}  Val sequences: {len(X_val)}  "
          f"Test sequences: {len(X_test)}\n")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:2d}/{epochs}  train MSE: {train_loss:.4f}  val MSE: {val_loss:.4f}{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping (no val improvement for {patience} epochs).")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(DEVICE)
        pred_scaled = model(X_test_t).cpu().numpy()

    y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'LSTM':>18s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:6.3f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm.pt"))
    joblib.dump(
        {"x_scaler": x_scaler, "y_scaler": y_scaler, "features": FEATURE_COLUMNS, "seq_len": SEQ_LEN},
        os.path.join(MODELS_DIR, "lstm_scalers.joblib"),
    )
    print(f"Saved model to {MODELS_DIR}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    main()
