"""Train all three models and print/plot a side-by-side comparison."""
import os

import matplotlib.pyplot as plt

import train_dl
import train_regression

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    print("=" * 60)
    print("Training Linear Regression + SVR")
    print("=" * 60)
    reg_metrics = train_regression.main()

    print("\n" + "=" * 60)
    print("Training LSTM")
    print("=" * 60)
    lstm_metrics = train_dl.main()

    all_metrics = {
        "Linear Regression": reg_metrics["linear_regression"],
        "SVR": reg_metrics["svr"],
        "LSTM": lstm_metrics,
    }

    print("\n" + "=" * 60)
    print(f"{'Model':<20s}{'MAE':>10s}{'RMSE':>10s}{'R2':>8s}")
    print("=" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<20s}{m['mae']:>10.2f}{m['rmse']:>10.2f}{m['r2']:>8.3f}")

    names = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, [all_metrics[n]["rmse"] for n in names], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(names, [all_metrics[n]["r2"] for n in names], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("R² (higher is better)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart to {out_path}")


if __name__ == "__main__":
    main()
