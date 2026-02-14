import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from ortools.linear_solver import pywraplp

DATA_IN = "data/demand_model_features.parquet"

HORIZON = 28
TOTAL_INVENTORY = 20000  # you can change later


FEATURE_COLS = [
    "price", "promo_flag", "discount_pct",
    "dow", "weekofyear", "month", "holiday_flag",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7",
    "roll_mean_28", "roll_std_28"
]


def train_global_model(df: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(df[FEATURE_COLS], df["units"])
    return model


def recursive_forecast_one_series(model: XGBRegressor, series_df: pd.DataFrame, horizon=28) -> pd.DataFrame:
    """
    series_df must be for ONE product-region, sorted by d, contains:
    d, units, price, promo_flag, discount_pct
    plus enough history for 28-day lags.
    """
    series_df = series_df.sort_values("d").reset_index(drop=True)

    last_date = series_df["d"].max()
    history_units = series_df["units"].to_list()

    # Carry forward last known price/promo assumptions
    last = series_df.iloc[-1]
    price = float(last["price"])
    promo_flag = int(last["promo_flag"])
    discount_pct = float(last["discount_pct"])

    forecasts = []

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)

        # Lags (need at least 28)
        lag_1 = history_units[-1]
        lag_7 = history_units[-7]
        lag_14 = history_units[-14]
        lag_28 = history_units[-28]

        y_last_7 = history_units[-7:]
        y_last_28 = history_units[-28:]

        roll_mean_7 = float(np.mean(y_last_7))
        roll_std_7 = float(np.std(y_last_7, ddof=1)) if len(y_last_7) > 1 else 0.0
        roll_mean_28 = float(np.mean(y_last_28))
        roll_std_28 = float(np.std(y_last_28, ddof=1)) if len(y_last_28) > 1 else 0.0

        dow = next_date.dayofweek
        weekofyear = int(next_date.isocalendar().week)
        month = next_date.month
        holiday_flag = 1 if (month == 12 and next_date.day >= 20) else 0

        row = {
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

        X_next = pd.DataFrame([row])[FEATURE_COLS]
        pred = float(model.predict(X_next)[0])
        pred = max(0.0, pred)

        forecasts.append({"d": next_date, "pred_units": pred})
        history_units.append(pred)

    return pd.DataFrame(forecasts)


def optimize_allocation(product_forecasts: pd.DataFrame, total_inventory: int) -> pd.DataFrame:
    """
    product_forecasts columns:
      product_id, forecast_demand_28d, unit_profit
    Decision: allocate_units[product] (integer)
    Objective: maximize sum(unit_profit * min(allocate, demand))
    We'll model expected_sold <= demand and expected_sold <= allocate
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver not available. OR-Tools install issue.")

    products = product_forecasts["product_id"].tolist()
    min_service = 0.02  # 10% minimum coverage for every product

    alloc = {p: solver.IntVar(0, total_inventory, f"alloc_{p}") for p in products}
    sold = {p: solver.NumVar(0, solver.infinity(), f"sold_{p}") for p in products}

    # Constraints: total allocation
    solver.Add(sum(alloc[p] for p in products) <= total_inventory)

    # sold constraints per product
    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        demand = float(row["forecast_demand_28d"])
        solver.Add(alloc[p] >= min_service * demand)

        solver.Add(sold[p] <= alloc[p])
        solver.Add(sold[p] <= demand)

    # Objective: maximize profit
    objective = solver.Objective()
    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        profit = float(row["unit_profit"])
        objective.SetCoefficient(sold[p], profit)
    objective.SetMaximization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Optimization did not find an optimal solution.")

    results = []
    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        results.append({
            "product_id": p,
            "forecast_demand_28d": float(row["forecast_demand_28d"]),
            "unit_profit": float(row["unit_profit"]),
            "allocated_units": int(round(alloc[p].solution_value())),
            "expected_sold": float(sold[p].solution_value()),
            "expected_profit": float(sold[p].solution_value()) * float(row["unit_profit"]),
        })

    out = pd.DataFrame(results).sort_values("expected_profit", ascending=False).reset_index(drop=True)
    return out


def main():
    df = pd.read_parquet(DATA_IN)
    df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values(["product_id", "region_id", "d"]).reset_index(drop=True)

    # Train global model on TRAIN split only
    train_end = pd.Timestamp("2025-09-30")
    train_df = df[df["d"] <= train_end].copy()

    print("Training global XGBoost model...")
    model = train_global_model(train_df)

    # Forecast next 28 days for every product-region
    series_keys = df[["product_id", "region_id"]].drop_duplicates()
    last_date = df["d"].max()

    print(f"Forecasting next {HORIZON} days from last date = {last_date.date()} for all product-region series...")

    forecasts_all = []
    for _, key in series_keys.iterrows():
        pid = key["product_id"]
        rid = key["region_id"]

        series = df[(df["product_id"] == pid) & (df["region_id"] == rid)][
            ["d", "units", "price", "promo_flag", "discount_pct"]
        ].copy()

        if len(series) < 60:
            continue

        fc = recursive_forecast_one_series(model, series, horizon=HORIZON)
        fc["product_id"] = pid
        fc["region_id"] = rid
        forecasts_all.append(fc)

    fc_all = pd.concat(forecasts_all, ignore_index=True)

    # Aggregate forecast to product-level demand
    product_demand = (
        fc_all.groupby("product_id")["pred_units"].sum().reset_index()
        .rename(columns={"pred_units": "forecast_demand_28d"})
    )

    # Compute unit profit per product (avg from history)
    profit_by_product = (
        df.groupby("product_id")["unit_profit"].mean().reset_index()
    )

    product_forecasts = product_demand.merge(profit_by_product, on="product_id", how="left")
    product_forecasts["unit_profit"] = product_forecasts["unit_profit"].fillna(0)

    min_service = 0.02
    required_min = float((product_forecasts["forecast_demand_28d"] * min_service).sum())
    print(f"Minimum required inventory for {min_service*100:.0f}% service: {required_min:.0f}")
    print(f"Available inventory: {TOTAL_INVENTORY}")

    if required_min > TOTAL_INVENTORY:
        print("⚠️ Infeasible with current TOTAL_INVENTORY. Lower min_service or increase TOTAL_INVENTORY.")

    # Run optimization
    print(f"Running optimization with TOTAL_INVENTORY={TOTAL_INVENTORY} units...")
    alloc_plan = optimize_allocation(product_forecasts, total_inventory=TOTAL_INVENTORY)

    # Save outputs
    fc_path = f"data/forecast_all_products_next{HORIZON}.csv"
    fc_all.to_csv(fc_path, index=False)

    alloc_path = f"data/allocation_plan_total{TOTAL_INVENTORY}.csv"
    alloc_plan.to_csv(alloc_path, index=False)

    print("\nSaved forecasts:", fc_path)
    print("Saved allocation plan:", alloc_path)

    print("\nTop 10 products by expected_profit:")
    print(alloc_plan.head(10))


if __name__ == "__main__":
    main()
