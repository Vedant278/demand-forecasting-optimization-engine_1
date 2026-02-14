import os
import pandas as pd
import numpy as np

PARQUET_IN = "data/demand_features_base.parquet"
PARQUET_OUT = "data/demand_model_features.parquet"

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]


def main():
    df = pd.read_parquet(PARQUET_IN)
    df = df.sort_values(["product_id", "region_id", "d"]).reset_index(drop=True)

    # Group by each time series (product-region)
    g = df.groupby(["product_id", "region_id"], group_keys=False)

    # Lag features
    for lag in LAGS:
        df[f"lag_{lag}"] = g["units"].shift(lag)

    # Rolling features (use shifted units to prevent leakage)
    for w in ROLL_WINDOWS:
        shifted = g["units"].shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(w).std()

    # Optional: simple trend feature
    df["diff_1"] = df["units"] - df["lag_1"]

    # Drop rows that don't have enough history for features
    feature_cols = [f"lag_{l}" for l in LAGS] + \
                   [f"roll_mean_{w}" for w in ROLL_WINDOWS] + \
                   [f"roll_std_{w}" for w in ROLL_WINDOWS] + ["diff_1"]

    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    after = len(df)

    print(f"Rows before: {before:,}")
    print(f"Rows after dropping NA feature rows: {after:,}")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"✅ Saved: {PARQUET_OUT}")
    print(df.head(3))


if __name__ == "__main__":
    main()
