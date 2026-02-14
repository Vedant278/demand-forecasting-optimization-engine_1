import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

DATA_IN = "data/demand_model_features.parquet"


def train_model(df_train, feature_cols):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(df_train[feature_cols], df_train["units"])
    return model


def main():
    df = pd.read_parquet(DATA_IN)
    df["d"] = pd.to_datetime(df["d"])

    product_id = "P001"
    region_id = "R01"

    one = df[(df["product_id"] == product_id) & (df["region_id"] == region_id)].copy()
    one = one.sort_values("d").reset_index(drop=True)

    feature_cols = [
        "price", "promo_flag", "discount_pct",
        "dow", "weekofyear", "month", "holiday_flag",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7",
        "roll_mean_28", "roll_std_28"
    ]

    # Time split (same as before)
    train_end = pd.Timestamp("2025-09-30")
    train = one[one["d"] <= train_end].copy()
    scored = one.copy()  # score all dates we have features for

    model = train_model(train, feature_cols)

    scored["pred"] = model.predict(scored[feature_cols])
    scored["residual"] = scored["units"] - scored["pred"]
    scored["abs_residual"] = scored["residual"].abs()

    # Rolling residual volatility (use past residuals only)
    window = 28
    scored["resid_roll_mean"] = scored["residual"].shift(1).rolling(window).mean()
    scored["resid_roll_std"] = scored["residual"].shift(1).rolling(window).std()

    # Z-score style anomaly score
    scored["anomaly_score"] = (scored["residual"] - scored["resid_roll_mean"]) / scored["resid_roll_std"]
    scored["anomaly_score"] = scored["anomaly_score"].replace([np.inf, -np.inf], np.nan)

    # Flag anomalies (tune threshold)
    threshold = 3.0
    scored["is_anomaly"] = (scored["anomaly_score"].abs() >= threshold).astype(int)

    anomalies = scored[scored["is_anomaly"] == 1].copy()
    anomalies = anomalies.dropna(subset=["anomaly_score"]).sort_values("anomaly_score", key=lambda s: s.abs(),
                                                                       ascending=False)

    print(f"\n✅ Anomalies for {product_id}-{region_id} (threshold={threshold}): {len(anomalies)}")
    print(anomalies[["d", "units", "pred", "residual", "anomaly_score"]].head(10))

    # Save anomalies
    out_csv = f"data/anomalies_{product_id}_{region_id}.csv"
    anomalies.to_csv(out_csv, index=False)
    print(f"Saved anomalies to: {out_csv}")

    # Plot last 120 days with anomalies highlighted
    view = scored.tail(120).copy()

    plt.figure(figsize=(12, 6))
    plt.plot(view["d"], view["units"], label="Actual")
    plt.plot(view["d"], view["pred"], label="Predicted")

    an_view = view[view["is_anomaly"] == 1]
    plt.scatter(an_view["d"], an_view["units"], label="Anomaly", marker="o")

    plt.title(f"Anomaly Detection (Residual-based) — {product_id}-{region_id}")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
