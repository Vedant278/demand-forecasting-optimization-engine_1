import pandas as pd
import numpy as np
from math import sqrt
import os

DATA_IN = "data/demand_model_features.parquet"

# Metrics
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def main():
    df = pd.read_parquet(DATA_IN)
    df = df.sort_values(["product_id", "region_id", "d"]).reset_index(drop=True)

    # ---- Time split (global cutoffs, easy + interview-friendly) ----
    # We'll use:
    # Train: up to 2025-09-30
    # Val:   2025-10-01 to 2025-11-30
    # Test:  2025-12-01 onward
    train_end = pd.Timestamp("2025-09-30")
    val_end = pd.Timestamp("2025-11-30")

    df["split"] = np.where(df["d"] <= train_end, "train",
                           np.where(df["d"] <= val_end, "val", "test"))

    # ---- Baselines ----
    # naive: predict y[t] = lag_1
    # seasonal naive: predict y[t] = lag_7
    for baseline_name, col in [("naive_lag1", "lag_1"), ("seasonal_naive_lag7", "lag_7")]:
        print("\n====", baseline_name, "====")
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split].copy()
            y_true = sub["units"].to_numpy()
            y_pred = sub[col].to_numpy()

            print(
                f"{split.upper():5s} | "
                f"MAE: {mae(y_true, y_pred):.3f}  "
                f"RMSE: {rmse(y_true, y_pred):.3f}  "
                f"SMAPE: {smape(y_true, y_pred):.2f}%  "
                f"(n={len(sub):,})"
            )

    # Quick sanity check of split sizes
    print("\nSplit counts:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
