# demand-forecasting-optimization-engine

## Optimization: Inventory Allocation (Forecast → Optimize)

**Goal:**
Allocate limited inventory across products to maximize expected profit while maintaining minimum service coverage.

---

### Inputs

* **28-day demand forecasts** (XGBoost, recursive multi-step)
* **Unit profit per product**
  *(average historical price − cost)*
* **Total inventory constraint**
* **Minimum service level:**
  **2% of forecast demand per product**

---

### Objective

Maximize:

∑(unit_profit×expected_sold)
---

### Constraints

1. Total allocation cannot exceed inventory:
  **∑allocated_units≤total_inventory**

2. Minimum service level per product:
  **allocated_units≥2%×forecast_demand**

3. Sales cannot exceed allocation:
   **expected_sold≤allocated_units**

4. Sales cannot exceed demand:
          **expected_sold≤allocated_units**

---

### Scenario Results

| Inventory | Status  | Total Expected Profit |
| --------- | ------- | --------------------- |
| 10,000    | Optimal | 264,864               |
| 20,000    | Optimal | 693,696               |
| 40,000    | Optimal | 1,508,534             |
| 80,000    | Optimal | 2,944,584             |



