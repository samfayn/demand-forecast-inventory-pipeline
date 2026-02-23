import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import (load_data, get_single_item, prepare_prophet_df,
                      train_forecast, evaluate_forecast, calculate_inventory,
                      STORE_LABELS, STORES_BY_STATE, STATE_LABELS,
                      STORE_SHORT_LABELS, CATEGORY_LABELS, DEPT_LABELS)

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Demand Forecast & Inventory Optimizer",
                   layout="wide")

st.title("Demand Forecast & Inventory Optimization Pipeline")
st.markdown("Built with Prophet forecasting and ISE inventory principles.")

# ── Load Data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_cached_data():
    df = load_data('data/sales_clean.parquet')
    return df[['item_id', 'store_id', 'state_id',
                'cat_id', 'dept_id', 'date', 'sales', 'sell_price']]

sales_clean = load_cached_data()

# ── Precompute avg demand per item for sorting the product dropdown ────────
@st.cache_data
def get_product_demand_summary(df):
    """Avg daily demand per item across all stores — used for sorting only."""
    summary = (
        df.groupby(['item_id', 'cat_id', 'dept_id'])['sales']
        .mean()
        .reset_index()
        .rename(columns={'sales': 'avg_daily_demand'})
    )
    return summary

demand_summary = get_product_demand_summary(sales_clean)

# ── Sidebar: Cascading Product Selection ──────────────────────────────────
st.sidebar.header("Product Selection")

# Step 1 — Category
st.sidebar.subheader("Category")
category_options = sorted(CATEGORY_LABELS.values())
selected_category_label = st.sidebar.selectbox(
    "Category", category_options, label_visibility="collapsed")
selected_cat_id = {v: k for k, v in CATEGORY_LABELS.items()}[selected_category_label]

# Step 2 — Department (filtered to selected category)
st.sidebar.subheader("Department")
dept_options_raw = sorted([k for k in DEPT_LABELS if k.startswith(selected_cat_id)])
dept_options_labels = [DEPT_LABELS[d] for d in dept_options_raw]
selected_dept_label = st.sidebar.selectbox(
    "Department", dept_options_labels, label_visibility="collapsed")
selected_dept_id = {v: k for k, v in DEPT_LABELS.items()}[selected_dept_label]

# Step 3 — Product (filtered to dept, sorted by avg demand descending)
st.sidebar.subheader("Product  (sorted by avg demand ↓)")
dept_products = demand_summary[demand_summary['dept_id'] == selected_dept_id].copy()
dept_products = dept_products.sort_values('avg_daily_demand', ascending=False)
dept_products['label'] = (
    dept_products['item_id'] +
    "  —  " +
    dept_products['avg_daily_demand'].map(lambda x: f"{x:.2f} units/day")
)
selected_product_label = st.sidebar.selectbox(
    "Product", dept_products['label'].tolist(), label_visibility="collapsed")
product_id = dept_products[
    dept_products['label'] == selected_product_label]['item_id'].values[0]
cat_id = selected_cat_id

# ── Sidebar: Cascading Store Selection ────────────────────────────────────
st.sidebar.header("Store Selection")

# Step 1 — State
st.sidebar.subheader("State")
state_options = sorted(STATE_LABELS.values())
selected_state_label = st.sidebar.selectbox(
    "State", state_options, label_visibility="collapsed")
selected_state_id = {v: k for k, v in STATE_LABELS.items()}[selected_state_label]

# Step 2 — Store (filtered to selected state)
st.sidebar.subheader("Store")
store_ids_in_state = STORES_BY_STATE[selected_state_id]
store_short_options = [STORE_SHORT_LABELS[s] for s in store_ids_in_state]
selected_store_short = st.sidebar.selectbox(
    "Store", store_short_options, label_visibility="collapsed")
store_id = store_ids_in_state[store_short_options.index(selected_store_short)]
selected_store_label = STORE_LABELS[store_id]  # full label, e.g. "California - Store 1"

# ── Sidebar: Inventory Parameters ────────────────────────────────────────
st.sidebar.header("Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 1, 100, 10)
holding_cost = st.sidebar.slider("Holding Cost (% of price/year)", 0.05, 0.50, 0.20)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)

# ── Cached Pipeline ───────────────────────────────────────────────────────
@st.cache_data
def run_pipeline(product_id, store_id, forecast_days,
                 lead_time, ordering_cost, holding_cost):
    item_df = get_single_item(sales_clean, product_id, store_id)
    if len(item_df) < 30:
        return None, None, None, None, None
    df_prophet = prepare_prophet_df(item_df)
    forecast = train_forecast(df_prophet, forecast_days)
    avg_price = item_df['sell_price'].mean()
    inv = calculate_inventory(forecast, df_prophet, avg_price,
                              lead_time_days=lead_time,
                              holding_cost=holding_cost,
                              ordering_cost=ordering_cost)
    eval_results = evaluate_forecast(df_prophet, holdout_days=90)
    return item_df, df_prophet, forecast, inv, eval_results

# ── Run ───────────────────────────────────────────────────────────────────
if st.sidebar.button("Run Forecast", type="primary"):

    with st.spinner("Training forecast model... this may take a moment"):
        item_df, df_prophet, forecast, inv, eval_results = run_pipeline(
            product_id, store_id, forecast_days,
            lead_time, ordering_cost, holding_cost)

    if inv is None:
        st.error("Not enough data for this product/store combination. Please select another.")
        st.stop()

    # ── Product Header ────────────────────────────────────────────────────
    st.subheader(f"{CATEGORY_LABELS.get(cat_id, cat_id)} | "
                 f"{product_id} at {selected_store_label}")

    # ── KPI Metrics ───────────────────────────────────────────────────────
    st.subheader("Inventory Policy Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Demand", f"{inv['avg_daily_demand']:.2f} units")
    col2.metric("Safety Stock", f"{inv['safety_stock']:.1f} units")
    col3.metric("Reorder Point", f"{inv['rop']:.1f} units")
    col4.metric("EOQ", f"{inv['eoq']:.1f} units")

    # ── Model Accuracy ────────────────────────────────────────────────────
    st.subheader("Model Accuracy (90-Day Holdout Backtest)")

    if eval_results is None:
        st.warning("Not enough historical data to run a backtest for this item.")
    else:
        mape = eval_results['mape']
        rmse = eval_results['rmse']
        avg_demand = inv['avg_daily_demand']

        # Detect intermittent demand — MAPE is structurally unreliable below ~1.5 units/day
        is_intermittent = avg_demand < 1.5

        if mape < 20:
            mape_label = f"✅ {mape:.1f}%"
        elif mape < 40:
            mape_label = f"⚠️ {mape:.1f}%"
        else:
            mape_label = f"🔴 {mape:.1f}%"

        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("MAPE (Mean Abs % Error)", mape_label,
                     help="Lower is better. Measures average % error on days with non-zero sales.")
        acol2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f} units",
                     help="Lower is better. Penalizes large errors more heavily than MAPE.")
        acol3.metric("Holdout Window", "Last 90 days",
                     help="Model trained on all data before this window, then tested against it.")

        # Contextual note for low-volume / intermittent demand items
        if is_intermittent:
            st.warning(
                f"**Intermittent demand detected** (avg {avg_demand:.2f} units/day). "
                f"MAPE is less meaningful for low-volume items — being off by 1 unit on a day "
                f"with 1 actual sale registers as 100% error, even though the absolute mistake "
                f"is small. For items like this, **RMSE ({rmse:.2f} units)** is a more reliable "
                f"accuracy signal, and the shape of the backtest chart below matters more than "
                f"the MAPE number. This is a known limitation of MAPE on intermittent retail "
                f"demand — it's why the M5 Kaggle competition used WRMSSE (weighted RMSE) instead."
            )
        elif mape > 40:
            st.info(
                f"**Note on this MAPE:** Retail demand forecasting at the individual item level "
                f"is genuinely difficult. The M5 dataset (real Walmart data) is a well-known "
                f"benchmark where even competition-winning ensemble models achieve MAPEs in this "
                f"range. A single-item Prophet model capturing weekly and yearly seasonality is "
                f"the right tool for interpretable, actionable forecasts — the backtest chart "
                f"below shows whether the model is tracking the right patterns."
            )

        comp = eval_results['comparison']
        fig_eval, ax_eval = plt.subplots(figsize=(12, 3))
        ax_eval.plot(comp['ds'], comp['y'],
                     color='steelblue', label='Actual Sales', alpha=0.7)
        ax_eval.plot(comp['ds'], comp['yhat'],
                     color='red', linewidth=2, label='Model Prediction')
        ax_eval.fill_between(comp['ds'], comp['yhat_lower'], comp['yhat_upper'],
                             alpha=0.15, color='red', label='Uncertainty Band')
        ax_eval.set_title(
            f'Backtest: Actual vs Predicted (last 90 days) — MAPE: {mape:.1f}%')
        ax_eval.set_xlabel('Date')
        ax_eval.set_ylabel('Units Sold')
        ax_eval.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_eval)

        st.caption(
            "The backtest trains the model on all data *except* the last 90 days, "
            "then forecasts that window and compares to what actually happened. "
            "This gives an honest estimate of how accurate future forecasts will be."
        )

    # ── Forecast Insights ─────────────────────────────────────────────────
    st.subheader("Forecast Insights")
    future_f = inv['future_forecast']

    peak_day = future_f.loc[future_f['yhat'].idxmax()]
    low_day = future_f.loc[future_f['yhat'].idxmin()]
    total_forecast = future_f['yhat'].sum()
    avg_upper = future_f['yhat_upper'].mean()

    icol1, icol2, icol3, icol4 = st.columns(4)
    icol1.metric("Total Forecasted Demand",
                 f"{total_forecast:.0f} units",
                 help=f"Over the next {forecast_days} days")
    icol2.metric("Peak Demand Day",
                 f"{peak_day['ds'].strftime('%b %d, %Y')}",
                 f"{peak_day['yhat']:.1f} units")
    icol3.metric("Lowest Demand Day",
                 f"{low_day['ds'].strftime('%b %d, %Y')}",
                 f"{low_day['yhat']:.1f} units")
    icol4.metric("Avg Worst-Case Daily Demand",
                 f"{avg_upper:.2f} units",
                 help="Upper bound — use this for conservative planning")

    # ── Forecast Chart ────────────────────────────────────────────────────
    st.subheader("Demand Forecast")

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    recent_history = df_prophet.tail(180)
    ax1.plot(recent_history['ds'], recent_history['y'],
             color='steelblue', alpha=0.5, label='Historical Sales (last 180 days)')
    ax1.plot(future_f['ds'], future_f['yhat'],
             color='red', linewidth=2, label='Forecast')
    ax1.fill_between(future_f['ds'],
                     future_f['yhat_lower'],
                     future_f['yhat_upper'],
                     alpha=0.2, color='red', label='Uncertainty Band')
    ax1.axvline(x=peak_day['ds'], color='orange',
                linestyle='--', alpha=0.7, label="Peak Day")
    ax1.set_title(f'Demand Forecast — {product_id} at {selected_store_label}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Units Sold')
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    # ── Plain English Summary ─────────────────────────────────────────────
    st.subheader("Summary")
    st.info(f"""
    **What this means for {product_id}:**
    - Over the next **{forecast_days} days**, expect to sell approximately **{total_forecast:.0f} units**
      at {selected_store_label}.
    - Demand peaks around **{peak_day['ds'].strftime('%B %d, %Y')}**
      at {peak_day['yhat']:.1f} units/day.
    - To maintain a **95% service level**, keep at least **{inv['safety_stock']:.0f} units**
      as safety stock.
    - Place a new order when inventory hits **{inv['rop']:.0f} units**
      (covers {lead_time} day lead time + safety stock).
    - Order **{inv['eoq']:.0f} units** at a time to minimize total inventory costs.
    """)

    # ── Inventory Simulation ──────────────────────────────────────────────
    st.subheader("Inventory Simulation")
    np.random.seed(42)
    simulated_demand = np.random.poisson(inv['avg_daily_demand'], size=forecast_days)
    inventory_levels = []
    current_inventory = inv['eoq']
    orders = []

    for day, demand in enumerate(simulated_demand):
        if current_inventory <= inv['rop']:
            current_inventory += inv['eoq']
            orders.append(day)
        current_inventory = max(0, current_inventory - demand)
        inventory_levels.append(current_inventory)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(inventory_levels, color='steelblue', linewidth=2,
             label='Inventory Level')
    ax2.axhline(y=inv['rop'], color='orange', linestyle='--',
                label=f"Reorder Point ({inv['rop']:.1f} units)")
    ax2.axhline(y=inv['safety_stock'], color='red', linestyle='--',
                label=f"Safety Stock ({inv['safety_stock']:.1f} units)")
    for order in orders:
        ax2.axvline(x=order, color='green', alpha=0.3, linestyle=':')
    ax2.set_title(f'Inventory Simulation — {product_id} at {selected_store_label}')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Units in Stock')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(
        f"Green dotted lines indicate order placement events. "
        f"{len(orders)} orders placed over {forecast_days} day simulation."
    )

else:
    st.info("Use the sidebar to select a category, department, product, and store — "
            "then click **Run Forecast** to begin.")