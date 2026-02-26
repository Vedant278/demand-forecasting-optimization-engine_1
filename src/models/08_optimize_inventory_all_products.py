import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from ortools.linear_solver import pywraplp

DATA_IN = "data/demand_model_features.parquet"

HORIZON = 28
INVENTORY_SCENARIOS = [10000, 20000, 40000, 80000]  # scenario analysis
MIN_SERVICE = 0.02  # 2% minimum coverage for every product

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
    series_df = series_df.sort_values("d").reset_index(drop=True)

    last_date = series_df["d"].max()
    history_units = series_df["units"].to_list()

    # Carry forward last known price/promo assumptions (simple default)
    last = series_df.iloc[-1]
    price = float(last["price"])
    promo_flag = int(last["promo_flag"])
    discount_pct = float(last["discount_pct"])

    forecasts = []

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)

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


def optimize_allocation(product_forecasts: pd.DataFrame, total_inventory: int, min_service: float) -> pd.DataFrame:
    """
    product_forecasts columns:
      product_id, forecast_demand_28d, unit_profit
    Decision: alloc[product] (integer), sold[product] (continuous)
    Constraints:
      sum alloc <= total_inventory
      alloc >= min_service * demand
      sold <= alloc
      sold <= demand
    Objective:
      maximize sum(unit_profit * sold)
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver not available. OR-Tools install issue.")

    products = product_forecasts["product_id"].tolist()

    alloc = {p: solver.IntVar(0, total_inventory, f"alloc_{p}") for p in products}
    sold = {p: solver.NumVar(0, solver.infinity(), f"sold_{p}") for p in products}

    solver.Add(sum(alloc[p] for p in products) <= total_inventory)

    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        demand = float(row["forecast_demand_28d"])

        # Minimum coverage
        solver.Add(alloc[p] >= min_service * demand)

        # Can’t sell more than you allocate or demand
        solver.Add(sold[p] <= alloc[p])
        solver.Add(sold[p] <= demand)

    objective = solver.Objective()
    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        profit = float(row["unit_profit"])
        objective.SetCoefficient(sold[p], profit)
    objective.SetMaximization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Optimization did not find an optimal solution (likely infeasible).")

    results = []
    for _, row in product_forecasts.iterrows():
        p = row["product_id"]
        demand = float(row["forecast_demand_28d"])
        up = float(row["unit_profit"])
        a = float(alloc[p].solution_value())
        s = float(sold[p].solution_value())
        results.append({
            "product_id": p,
            "forecast_demand_28d": demand,
            "unit_profit": up,
            "allocated_units": int(round(a)),
            "expected_sold": s,
            "expected_profit": s * up
        })

    return pd.DataFrame(results).sort_values("expected_profit", ascending=False).reset_index(drop=True)


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
    profit_by_product = df.groupby("product_id")["unit_profit"].mean().reset_index()

    product_forecasts = product_demand.merge(profit_by_product, on="product_id", how="left")
    product_forecasts["unit_profit"] = product_forecasts["unit_profit"].fillna(0)

    # Save the forecast file once (common across scenarios)
    fc_path = f"data/forecast_all_products_next{HORIZON}.csv"
    fc_all.to_csv(fc_path, index=False)
    print("\nSaved forecasts:", fc_path)

    # Feasibility check for min service
    required_min = float((product_forecasts["forecast_demand_28d"] * MIN_SERVICE).sum())
    print(f"\nMinimum required inventory for {MIN_SERVICE*100:.0f}% service across all products: {required_min:.0f}")

    # Run scenarios
    scenario_rows = []
    for inv in INVENTORY_SCENARIOS:
        print(f"\n--- Scenario: TOTAL_INVENTORY={inv} ---")
        if required_min > inv:
            print("Infeasible with this inventory level. Skipping.")
            scenario_rows.append({
                "inventory": inv,
                "status": "infeasible",
                "total_expected_profit": None
            })
            continue

        alloc_plan = optimize_allocation(product_forecasts, total_inventory=inv, min_service=MIN_SERVICE)

        alloc_path = f"data/allocation_plan_total{inv}.csv"
        alloc_plan.to_csv(alloc_path, index=False)

        total_profit = float(alloc_plan["expected_profit"].sum())
        scenario_rows.append({
            "inventory": inv,
            "status": "optimal",
            "total_expected_profit": total_profit
        })

        print(f"Saved allocation plan: {alloc_path}")
        print(f"Total Expected Profit: {total_profit:,.0f}")
        print("Top 5 products by expected_profit:")
        print(alloc_plan.head(5))

    scenario_df = pd.DataFrame(scenario_rows)

    scenario_path = "data/optimization_scenarios_summary.csv"
    scenario_df.to_csv(scenario_path, index=False)

    print("\n=== Scenario Summary ===")
    print(scenario_df)
    print("\nSaved scenario summary:", scenario_path)


if __name__ == "__main__":
    main()
