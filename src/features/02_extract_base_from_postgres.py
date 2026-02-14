import os
import pandas as pd
from sqlalchemy import create_engine

DB_NAME = "demand_forecasting_db"
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "vedant"

def get_engine():
    pwd = "33Patel33#"
    if not pwd:
        raise ValueError("Set PG_PASSWORD env var before running.")
    url = f"postgresql+psycopg2://{DB_USER}:{pwd}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, future=True)

def main():
    engine = get_engine()

    query = """
    SELECT
      d, product_id, region_id, units,
      price, unit_cost, unit_profit,
      promo_flag, discount_pct,
      dow, weekofyear, month, year,
      holiday_flag
    FROM marts.demand_features_base
    ORDER BY product_id, region_id, d;
    """

    df = pd.read_sql(query, engine, parse_dates=["d"])
    print("Rows:", len(df))
    print(df.head())

    # Save locally for faster next steps
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/demand_features_base.parquet", index=False)
    print("✅ Saved: data/demand_features_base.parquet")

if __name__ == "__main__":
    main()
