import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "src//models//data")

st.set_page_config(page_title="Demand Forecasting & Inventory Optimization", layout="wide")

st.title("Demand Forecasting & Inventory Optimization Engine")
st.caption("Forecast → Optimize → Scenario Analysis (XGBoost + OR-Tools)")

# ---- Load scenario summary ----
scenario_path = os.path.join(DATA_DIR, "optimization_scenarios_summary.csv")
if not os.path.exists(scenario_path):
    st.error("Missing data/optimization_scenarios_summary.csv. Run the optimization script first.")
    st.stop()

scenario_df = pd.read_csv(scenario_path)
scenario_df = scenario_df.sort_values("inventory")

# ---- Sidebar ----
st.sidebar.header("Controls")

inventories = scenario_df["inventory"].tolist()
selected_inventory = st.sidebar.selectbox("Select Inventory Scenario", inventories, index=inventories.index(20000) if 20000 in inventories else 0)

alloc_path = os.path.join(DATA_DIR, f"allocation_plan_total{selected_inventory}.csv")
if not os.path.exists(alloc_path):
    st.error(f"Missing {alloc_path}. Run the optimization script to generate this scenario.")
    st.stop()

alloc_df = pd.read_csv(alloc_path)

# ---- KPI Row ----
col1, col2, col3, col4 = st.columns(4)

total_profit = float(alloc_df["expected_profit"].sum())
total_alloc = int(alloc_df["allocated_units"].sum())
avg_profit = float(alloc_df["expected_profit"].mean())
covered_products = int((alloc_df["allocated_units"] > 0).sum())

col1.metric("Inventory", f"{selected_inventory:,}")
col2.metric("Total Allocated", f"{total_alloc:,}")
col3.metric("Total Expected Profit", f"{total_profit:,.0f}")
col4.metric("Products Covered", f"{covered_products}")

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 Scenarios", "📦 Allocation", "⬇️ Downloads"])

with tab1:
    st.subheader("Scenario Profit Curve")

    curve = scenario_df.copy()
    curve["total_expected_profit"] = pd.to_numeric(curve["total_expected_profit"], errors="coerce")

    left, right = st.columns([2, 1])

    with left:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(curve["inventory"], curve["total_expected_profit"], marker="o")
        plt.xlabel("Total Inventory")
        plt.ylabel("Total Expected Profit")
        plt.title("Profit vs Inventory Scenario")
        st.pyplot(fig)

    with right:
        st.markdown("### Executive Summary")
        st.write(
            f"""
            **Selected inventory:** {selected_inventory:,}  
            **Expected profit:** {total_profit:,.0f}  
            **Products covered:** {covered_products}  
            
            Use the left chart to compare how profit scales as inventory increases.
            """
        )

with tab2:
    st.subheader("Top Products by Expected Profit")
    top_n = st.sidebar.slider("Top N products", min_value=5, max_value=30, value=10, step=1)

    top_df = alloc_df.sort_values("expected_profit", ascending=False).head(top_n).copy()
    top_df["forecast_demand_28d"] = top_df["forecast_demand_28d"].round(0).astype(int)
    top_df["unit_profit"] = top_df["unit_profit"].round(2)
    top_df["expected_sold"] = top_df["expected_sold"].round(0).astype(int)
    top_df["expected_profit"] = top_df["expected_profit"].round(0).astype(int)

    st.dataframe(top_df, width="stretch")

    st.subheader("Allocation Distribution (Top 15)")
    alloc_sorted = alloc_df.sort_values("allocated_units", ascending=False).head(15).copy()

    fig2 = plt.figure(figsize=(10, 4))
    plt.bar(alloc_sorted["product_id"], alloc_sorted["allocated_units"])
    plt.xlabel("Product")
    plt.ylabel("Allocated Units")
    plt.title("Top 15 Allocations")
    st.pyplot(fig2)

with tab3:
    st.subheader("Download Outputs")

    st.download_button(
        "Download Allocation CSV",
        data=alloc_df.to_csv(index=False),
        file_name=f"allocation_plan_total{selected_inventory}.csv",
        mime="text/csv",
    )

    forecast_path = os.path.join(DATA_DIR, "forecast_all_products_next28.csv")
    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)
        st.download_button(
            "Download Forecast CSV (All Products)",
            data=forecast_df.to_csv(index=False),
            file_name="forecast_all_products_next28.csv",
            mime="text/csv",
        )
    else:
        st.info("Forecast CSV not found. Run optimization script to generate it.")

