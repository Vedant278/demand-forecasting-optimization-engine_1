import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

DATA_IN = "data/demand_model_features.parquet"

HORIZON = 28


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


def build_next_row(history: pd.DataFrame, next_date: pd.Timestamp) -> dict:
    """
    history contains past rows for ONE product-region, sorted by date, and includes
    the base columns needed to compute features.
    We create a single feature row for next_date using ONLY past info.
    """

    # Lags
    y = history["units"].to_numpy()
    lag_1 = y[-1]
    lag_7 = y[-7]
    lag_14 = y[-14]
    lag_28 = y[-28]

    # Rolling from last known actuals (shifted by 1 implicitly since we're forecasting next day)
    roll_mean_7 = np.mean(y[-7:])
    roll_std_7 = np.std(y[-7:], ddof=1) if len(y[-7:]) > 1 else 0.0
    roll_mean_28 = np.mean(y[-28:])
    roll_std_28 = np.std(y[-28:], ddof=1) if len(y[-28:]) > 1 else 0.0

    # Time features
    dow = next_date.dayofweek
    weekofyear = int(next_date.isocalendar().week)
    month = next_date.month
    holiday_flag = 1 if (month == 12 and next_date.day >= 20) else 0  # same rule as synthetic

    # Price/promo assumptions for future (simple baseline)
    # We'll carry forward last known price/promo.
    last = history.iloc[-1]
    price = float(last["price"])
    promo_flag = int(last["promo_flag"])
    discount_pct = float(last["discount_pct"])

    row = {
        "d": next_date,
        "price": price,
        "promo_flag": promo_flag,
        "discount_pct": discount_pct,
        "dow": dow,
        "weekofyear": weekofyear,
        "month": month,
        "holiday_flag": holiday_flag,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "roll_mean_7": roll_mean_7,
        "roll_std_7": roll_std_7,
        "roll_mean_28": roll_mean_28,
        "roll_std_28": roll_std_28,
    }
    return row


def main():
    df = pd.read_parquet(DATA_IN)
    df["d"] = pd.to_datetime(df["d"])

    # Choose one series (you can change these later)
    product_id = "P001"
    region_id = "R01"

    one = df[(df["product_id"] == product_id) & (df["region_id"] == region_id)].copy()
    one = one.sort_values("d").reset_index(drop=True)

    # Safety: need at least 28 history points
    if len(one) < 60:
        raise ValueError("Not enough history for this product-region.")

    # Train on all history up to the last available date (single-series demo)
    feature_cols = [
        "price", "promo_flag", "discount_pct",
        "dow", "weekofyear", "month", "holiday_flag",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7",
        "roll_mean_28", "roll_std_28"
    ]

    model = train_model(one, feature_cols)

    # We'll forecast forward from last known date
    last_date = one["d"].max()
    history = one[["d", "units", "price", "promo_flag", "discount_pct"]].copy()

    forecasts = []
    for step in range(1, HORIZON + 1):
        next_date = last_date + pd.Timedelta(days=step)

        # build feature row based on history units
        row = build_next_row(history, next_date)

        X_next = pd.DataFrame([row])[feature_cols]
        y_pred = float(model.predict(X_next)[0])

        # demand must be non-negative and integer-ish
        y_pred = max(0.0, y_pred)

        forecasts.append({"d": next_date, "pred_units": y_pred})

        # append predicted as if it happened (recursive forecasting)
        history = pd.concat(
            [history, pd.DataFrame([{
                "d": next_date,
                "units": y_pred,
                "price": row["price"],
                "promo_flag": row["promo_flag"],
                "discount_pct": row["discount_pct"],
            }])],
            ignore_index=True
        )

    fc = pd.DataFrame(forecasts)
    print(f"\n Forecast generated for {product_id} - {region_id} for next {HORIZON} days")
    print(fc.head(10))
    print("\nLast 5 forecast days:")
    print(fc.tail(5))

    # Save
    out_path = f"data/forecast_{product_id}_{region_id}_next{HORIZON}.csv"
    fc.to_csv(out_path, index=False)
    print(f"\nSaved forecast to: {out_path}")

    # ===============================
    # PLOT LAST 60 DAYS + FORECAST
    # ===============================

    import matplotlib.pyplot as plt

    # Get last 60 actual days
    actual = one.sort_values("d").tail(60)

    plt.figure(figsize=(12,6))

    # Plot actuals
    plt.plot(actual["d"], actual["units"], label="Actual (last 60 days)")

    # Plot forecast
    plt.plot(fc["d"], fc["pred_units"], label="Forecast (next 28 days)")

    # Vertical line separating history and forecast
    plt.axvline(x=actual["d"].max(), linestyle="--")

    plt.title(f"Demand Forecast for {product_id} - {region_id}")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
