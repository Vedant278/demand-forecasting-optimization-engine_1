import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

DATA_IN = "data/demand_model_features.parquet"

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def main():
    df = pd.read_parquet(DATA_IN)

    # Time split
    train_end = pd.Timestamp("2025-09-30")
    val_end = pd.Timestamp("2025-11-30")

    df["split"] = np.where(df["d"] <= train_end, "train",
                    np.where(df["d"] <= val_end, "val", "test"))

    feature_cols = [
        "price", "promo_flag", "discount_pct",
        "dow", "weekofyear", "month", "holiday_flag",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7",
        "roll_mean_28", "roll_std_28"
    ]

    X_train = df[df["split"] == "train"][feature_cols]
    y_train = df[df["split"] == "train"]["units"]

    X_val = df[df["split"] == "val"][feature_cols]
    y_val = df[df["split"] == "val"]["units"]

    X_test = df[df["split"] == "test"][feature_cols]
    y_test = df[df["split"] == "test"]["units"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    for split_name, X, y in [
        ("TRAIN", X_train, y_train),
        ("VAL", X_val, y_val),
        ("TEST", X_test, y_test),
    ]:
        preds = model.predict(X)
        print(f"\n{split_name}")
        print("MAE:", mean_absolute_error(y, preds))
        print("RMSE:", sqrt(mean_squared_error(y, preds)))
        print("SMAPE:", smape(y.to_numpy(), preds))

        # ============================
        # FEATURE IMPORTANCE SECTION
        # ============================

    important_features = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": important_features
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(feat_imp.head(10))

    plt.figure(figsize=(8, 6))
    plt.barh(feat_imp["feature"].head(10)[::-1],
             feat_imp["importance"].head(10)[::-1])
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
