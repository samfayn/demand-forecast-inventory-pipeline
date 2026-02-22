import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import (load_data, get_single_item, prepare_prophet_df,
                      train_forecast, calculate_inventory,
                      STORE_LABELS, CATEGORY_LABELS, DEPT_LABELS)

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Demand Forecast & Inventory Optimizer",
                   layout="wide")

st.title("Demand Forecast & Inventory Optimization Pipeline")
st.markdown("Built with Prophet forecasting and ISE inventory principles.")

# ── Sidebar Controls ──────────────────────────────────────────────────────
st.sidebar.header("Product Selection")

@st.cache_data
def load_cached_data():
    df = load_data('data/sales_clean.parquet')
    # Keep only necessary columns to reduce memory
    return df[['item_id', 'store_id', 'state_id',
                'cat_id', 'dept_id', 'date', 'sales', 'sell_price']]

sales_clean = load_cached_data()

# Build readable product list
product_df = sales_clean[['item_id', 'cat_id', 'dept_id']].drop_duplicates()
product_df['label'] = (product_df['cat_id'].map(CATEGORY_LABELS) +
                       ' | ' + product_df['item_id'])
product_df = product_df.sort_values('label')

store_options = {v: k for k, v in STORE_LABELS.items()}
store_labels = sorted(STORE_LABELS.values())

selected_product_label = st.sidebar.selectbox(
    "Select Product", product_df['label'].tolist())
selected_store_label = st.sidebar.selectbox(
    "Select Store", store_labels)

product_id = product_df[product_df['label'] ==
                         selected_product_label]['item_id'].values[0]
store_id = store_options[selected_store_label]
cat_id = product_df[product_df['label'] ==
                     selected_product_label]['cat_id'].values[0]

st.sidebar.header("Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 1, 100, 10)
holding_cost = st.sidebar.slider("Holding Cost (% of price/year)", 0.05, 0.50, 0.20)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)

# ── Cache the forecast so it doesn't retrain on every interaction ─────────
@st.cache_data
def run_pipeline(product_id, store_id, forecast_days,
                 lead_time, ordering_cost, holding_cost):
    item_df = get_single_item(sales_clean, product_id, store_id)
    if len(item_df) < 30:
        return None, None, None, None
    df_prophet = prepare_prophet_df(item_df)
    forecast = train_forecast(df_prophet, forecast_days)
    avg_price = item_df['sell_price'].mean()
    inv = calculate_inventory(forecast, df_prophet, avg_price,
                              lead_time_days=lead_time,
                              holding_cost=holding_cost,
                              ordering_cost=ordering_cost)
    return item_df, df_prophet, forecast, inv

# ── Run Pipeline ──────────────────────────────────────────────────────────
if st.sidebar.button("Run Forecast", type="primary"):

    with st.spinner("Training forecast model... this may take a moment"):
        item_df, df_prophet, forecast, inv = run_pipeline(
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

    # Only show last 180 days of history to make chart cleaner
    recent_history = df_prophet.tail(180)
    ax1.plot(recent_history['ds'], recent_history['y'],
             color='steelblue', alpha=0.5, label='Historical Sales (last 180 days)')
    ax1.plot(future_f['ds'], future_f['yhat'],
             color='red', linewidth=2, label='Forecast')
    ax1.fill_between(future_f['ds'],
                     future_f['yhat_lower'],
                     future_f['yhat_upper'],
                     alpha=0.2, color='red', label='Uncertainty Band')

    # Mark peak demand day
    ax1.axvline(x=peak_day['ds'], color='orange',
                linestyle='--', alpha=0.7, label=f"Peak Day")

    ax1.set_title(f'Demand Forecast - {product_id} at {selected_store_label}')
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

    # ── Inventory Simulation Chart ────────────────────────────────────────
    st.subheader("Inventory Simulation")
    np.random.seed(42)
    simulated_demand = np.random.poisson(inv['avg_daily_demand'],
                                          size=forecast_days)
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
    ax2.set_title(f'Inventory Simulation - {product_id} at {selected_store_label}')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Units in Stock')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(f"Green dotted lines indicate order placement events. "
               f"{len(orders)} orders placed over {forecast_days} day simulation.")

else:
    st.info("Select a product and store from the sidebar, then click Run Forecast to begin.")