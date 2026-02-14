import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# ----------------------------
# 1) CONFIG (edit if needed)
# ----------------------------
@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "demand_forecasting_db"
    user: str = "vedant"
    password: str = "33Patel33#"  # <-- put your password OR set env var PG_PASSWORD


def get_engine(cfg: DBConfig):
    pwd = cfg.password
    print(pwd)
    if not pwd:
        raise ValueError(
            "Postgres password not provided. Either set cfg.password in the script "
            "or set environment variable PG_PASSWORD."
        )
    url = f"postgresql+psycopg2://{cfg.user}:{pwd}@{cfg.host}:{cfg.port}/{cfg.dbname}"
    return create_engine(url, future=True)


# ----------------------------
# 2) SYNTHETIC DATA GENERATION
# ----------------------------
def make_calendar(start: date, end: date) -> pd.DataFrame:
    ds = pd.date_range(start=start, end=end, freq="D")
    cal = pd.DataFrame({"d": ds})
    cal["dow"] = cal["d"].dt.dayofweek  # 0=Mon
    cal["weekofyear"] = cal["d"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["d"].dt.month
    cal["year"] = cal["d"].dt.year
    # simple "holiday" flags (synthetic)
    cal["holiday_flag"] = ((cal["month"] == 12) & (cal["d"].dt.day >= 20)).astype(int)
    cal["event_name"] = np.where(cal["holiday_flag"] == 1, "HolidaySeason", None)
    return cal


def generate_synthetic_data(
    start: date,
    end: date,
    n_products: int = 30,
    n_regions: int = 5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    products = pd.DataFrame(
        {
            "product_id": [f"P{str(i).zfill(3)}" for i in range(1, n_products + 1)],
            "category": rng.choice(["A", "B", "C"], size=n_products, p=[0.5, 0.3, 0.2]),
            "base_price": rng.uniform(8, 60, size=n_products).round(2),
            "unit_cost": rng.uniform(3, 35, size=n_products).round(2),
        }
    )
    products["unit_cost"] = np.minimum(products["unit_cost"], products["base_price"] * 0.85)

    regions = pd.DataFrame(
        {
            "region_id": [f"R{str(i).zfill(2)}" for i in range(1, n_regions + 1)],
            "region_name": [f"Region {i}" for i in range(1, n_regions + 1)],
        }
    )

    cal = make_calendar(start, end)
    ds = cal["d"].to_numpy()

    # Cross join calendar x product x region
    grid = (
        cal.assign(key=1)
        .merge(products.assign(key=1), on="key")
        .merge(regions.assign(key=1), on="key")
        .drop(columns=["key"])
    )

    # Price with small random walk per product
    grid = grid.sort_values(["product_id", "region_id", "d"]).reset_index(drop=True)
    grid["price"] = grid["base_price"].astype(float)

    # Add weekly + holiday demand seasonality and region effects
    region_effect = {rid: eff for rid, eff in zip(regions["region_id"], rng.uniform(0.8, 1.2, n_regions))}
    category_mult = {"A": 1.0, "B": 0.8, "C": 1.2}

    # Promotions (random campaigns)
    promo_prob = 0.06  # 6% of days are promo days per product-region
    grid["promo_flag"] = (rng.random(len(grid)) < promo_prob).astype(int)
    grid["discount_pct"] = np.where(grid["promo_flag"] == 1, rng.uniform(0.05, 0.25, len(grid)), 0.0).round(3)

    # Apply discounts to price
    grid["price"] = (grid["price"] * (1 - grid["discount_pct"])).round(2)

    # Demand generation
    # baseline demand depends on category, region, and product popularity
    product_pop = {pid: pop for pid, pop in zip(products["product_id"], rng.uniform(20, 140, n_products))}
    dow_mult = {0: 0.95, 1: 0.98, 2: 1.00, 3: 1.02, 4: 1.05, 5: 1.10, 6: 1.08}  # weekend stronger

    # simple price elasticity effect
    # higher price => lower demand (log relationship)
    base_price_map = dict(zip(products["product_id"], products["base_price"]))
    elasticity = rng.uniform(0.8, 1.6, n_products)  # per product
    elast_map = dict(zip(products["product_id"], elasticity))

    def calc_mu(row):
        pid = row["product_id"]
        rid = row["region_id"]
        cat = row["category"]
        mu = product_pop[pid] * region_effect[rid] * category_mult[cat] * dow_mult[int(row["dow"])]
        # holiday uplift
        if row["holiday_flag"] == 1:
            mu *= 1.25
        # promo uplift
        if row["promo_flag"] == 1:
            mu *= 1.20
        # price elasticity (compare to base price)
        rel_price = max(row["price"] / base_price_map[pid], 0.3)
        mu *= rel_price ** (-elast_map[pid])
        return mu

    # Vectorized-ish apply (still okay for our dataset size)
    mu = grid.apply(calc_mu, axis=1).to_numpy()
    # Add noise and generate integer units (Poisson-like with overdispersion)
    noise = rng.normal(1.0, 0.18, size=len(mu))
    mu_noisy = np.clip(mu * noise, 0.1, None)
    units = rng.poisson(lam=np.clip(mu_noisy, 0.1, 500))

    # Add occasional anomalies (spikes/drops)
    anomaly_idx = rng.choice(len(grid), size=int(len(grid) * 0.008), replace=False)  # ~0.8% points
    spike = rng.choice([0.3, 2.5], size=len(anomaly_idx), p=[0.55, 0.45])  # drop or spike
    units[anomaly_idx] = np.clip((units[anomaly_idx] * spike).astype(int), 0, None)

    grid["units"] = units.astype(int)

    sales_daily = grid[["d", "product_id", "region_id", "units"]].copy()
    prices_daily = grid[["d", "product_id", "price"]].copy().drop_duplicates()
    promotions_daily = grid[["d", "product_id", "region_id", "promo_flag", "discount_pct"]].copy()

    return {
        "products": products,
        "regions": regions,
        "calendar": cal,
        "sales_daily": sales_daily,
        "prices_daily": prices_daily,
        "promotions_daily": promotions_daily,
    }


# ----------------------------
# 3) DDL + LOAD INTO POSTGRES
# ----------------------------
DDL = """
CREATE SCHEMA IF NOT EXISTS raw;

DROP TABLE IF EXISTS raw.products;
DROP TABLE IF EXISTS raw.regions;
DROP TABLE IF EXISTS raw.calendar;
DROP TABLE IF EXISTS raw.sales_daily;
DROP TABLE IF EXISTS raw.prices_daily;
DROP TABLE IF EXISTS raw.promotions_daily;

CREATE TABLE raw.products (
  product_id TEXT PRIMARY KEY,
  category TEXT,
  base_price NUMERIC(10,2),
  unit_cost NUMERIC(10,2)
);

CREATE TABLE raw.regions (
  region_id TEXT PRIMARY KEY,
  region_name TEXT
);

CREATE TABLE raw.calendar (
  d DATE PRIMARY KEY,
  dow INT,
  weekofyear INT,
  month INT,
  year INT,
  holiday_flag INT,
  event_name TEXT
);

CREATE TABLE raw.sales_daily (
  d DATE,
  product_id TEXT,
  region_id TEXT,
  units INT,
  PRIMARY KEY (d, product_id, region_id)
);

CREATE TABLE raw.prices_daily (
  d DATE,
  product_id TEXT,
  price NUMERIC(10,2),
  PRIMARY KEY (d, product_id)
);

CREATE TABLE raw.promotions_daily (
  d DATE,
  product_id TEXT,
  region_id TEXT,
  promo_flag INT,
  discount_pct NUMERIC(10,3),
  PRIMARY KEY (d, product_id, region_id)
);
"""


def main():
    cfg = DBConfig()
    engine = get_engine(cfg)

    # Generate 18 months of data
    start = date(2024, 8, 1)
    end = date(2026, 1, 31)

    data = generate_synthetic_data(start=start, end=end, n_products=30, n_regions=5, seed=42)

    with engine.begin() as conn:
        conn.execute(text(DDL))

    # Load dataframes
    data["products"].to_sql("products", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
    data["regions"].to_sql("regions", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
    data["calendar"].to_sql("calendar", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
    data["sales_daily"].to_sql("sales_daily", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
    data["prices_daily"].to_sql("prices_daily", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
    data["promotions_daily"].to_sql("promotions_daily", engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)

    print("✅ Loaded synthetic data into Postgres schema raw.")
    print("Tables: raw.products, raw.regions, raw.calendar, raw.sales_daily, raw.prices_daily, raw.promotions_daily")


if __name__ == "__main__":
    main()
