import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import (load_data, get_single_item, prepare_prophet_df,
                      train_forecast, evaluate_forecast, calculate_inventory,
                      save_results_to_db, load_all_runs, load_run_forecast,
                      ForecastingError, DatabaseError,
                      PARQUET_PATH, USING_DEMO_DATA,
                      STORE_LABELS, STORES_BY_STATE, STATE_LABELS,
                      STORE_SHORT_LABELS, CATEGORY_LABELS, DEPT_LABELS)
 
st.set_page_config(page_title="Demand Forecast & Inventory Optimizer",
                   layout="wide")
 
st.title("Demand Forecast & Inventory Optimization Pipeline")
st.markdown("Built with Prophet forecasting and ISE inventory principles.")

if not os.path.exists(PARQUET_PATH):
    st.error(
        f"No dataset found at `{PARQUET_PATH}`.\n\n"
        "Run `python src/data_prep.py` to build it from the raw M5 CSVs, or "
        "`python scripts/build_demo_data.py` to create the smaller demo subset."
    )
    st.stop()

if USING_DEMO_DATA:
    st.info(
        "**Demo dataset.** This deployment runs on a subset of the M5 data covering "
        "the item/store combinations that have saved forecast runs, because "
        "Streamlit Community Cloud cannot hold the full 46M-row dataset in memory. "
        "The pipeline, forecasting, and inventory logic are identical to the full "
        "version; only the catalog is smaller. Clone the repo and run "
        "`src/data_prep.py` to work with everything."
    )
 
@st.cache_data
def load_cached_data():
    df = load_data(PARQUET_PATH)
    return df[['item_id', 'store_id', 'state_id',
                'cat_id', 'dept_id', 'date', 'sales', 'sell_price']]
 
sales_clean = load_cached_data()
 
@st.cache_data
def get_product_demand_summary(df):
    """Avg daily demand per item across all stores, used to sort dropdowns."""
    summary = (
        df.groupby(['item_id', 'cat_id', 'dept_id'])['sales']
        .mean()
        .reset_index()
        .rename(columns={'sales': 'avg_daily_demand'})
    )
    return summary
 
demand_summary = get_product_demand_summary(sales_clean)


def _available_categories():
    """Categories that actually have items in the loaded dataset.

    Options are derived from the data rather than from CATEGORY_LABELS so a
    trimmed dataset (the deployed demo, or any filtered subset) can't produce
    an empty dropdown that breaks st.selectbox.
    """
    present = set(demand_summary['cat_id'].unique())
    return sorted(CATEGORY_LABELS[c] for c in CATEGORY_LABELS if c in present)


def _available_departments(cat_id):
    """Departments within a category that have items in the loaded dataset."""
    present = set(demand_summary[demand_summary['cat_id'] == cat_id]['dept_id'].unique())
    return sorted(DEPT_LABELS[d] for d in DEPT_LABELS if d in present)


def _available_stores(state_id):
    """Stores in a state that have items in the loaded dataset."""
    present = set(sales_clean['store_id'].unique())
    return [s for s in STORES_BY_STATE[state_id] if s in present]


def _available_states():
    present_stores = set(sales_clean['store_id'].unique())
    return sorted(
        STATE_LABELS[s] for s in STATE_LABELS
        if any(store in present_stores for store in STORES_BY_STATE[s])
    )


def render_product_selector(key_prefix, sidebar=False, label_suffix="", cat_index=0):
    """
    Render the Category -> Department -> Product cascade and return
    (product_id, cat_id).

    sidebar=True renders in the sidebar with a subheader above each collapsed
    widget (Tab 1 style); sidebar=False renders inline with a visible label,
    optionally suffixed (" A" / " B" for the side-by-side comparison in Tab 2).
    The selection logic is identical either way, only the presentation differs.
    """
    container = st.sidebar if sidebar else st

    def selectbox(label, options, index=0, key=None):
        if sidebar:
            container.subheader(label)
            return container.selectbox(label, options, index=index,
                                       label_visibility="collapsed", key=key)
        return container.selectbox(f"{label}{label_suffix}", options,
                                   index=index, key=key)

    cat_options = _available_categories()
    cat_label = selectbox("Category", cat_options,
                          index=min(cat_index, len(cat_options) - 1),
                          key=f"{key_prefix}_cat")
    cat_id = {v: k for k, v in CATEGORY_LABELS.items()}[cat_label]

    dept_label = selectbox("Department", _available_departments(cat_id),
                           key=f"{key_prefix}_dept")
    dept_id = {v: k for k, v in DEPT_LABELS.items()}[dept_label]

    dept_products = demand_summary[demand_summary['dept_id'] == dept_id].copy()
    dept_products = dept_products.sort_values('avg_daily_demand', ascending=False)
    dept_products['label'] = (
        dept_products['item_id'] + "  —  " +
        dept_products['avg_daily_demand'].map(lambda x: f"{x:.2f} units/day")
    )
    product_label = selectbox("Product (sorted by avg demand ↓)",
                              dept_products['label'].tolist(),
                              key=f"{key_prefix}_prod")
    product_id = dept_products[
        dept_products['label'] == product_label]['item_id'].values[0]

    return product_id, cat_id


def render_store_selector(key_prefix, sidebar=False, containers=None):
    """
    Render the State -> Store cascade and return (store_id, store_label).

    containers, if given, is a (state_container, store_container) tuple — two
    st.columns(), for instance — letting the caller control layout. Without it,
    both render into the sidebar or into the main body depending on `sidebar`.
    """
    if containers is not None:
        state_container, store_container = containers
    elif sidebar:
        state_container = store_container = st.sidebar
    else:
        state_container = store_container = st

    def selectbox(container, label, options, key=None):
        if sidebar:
            container.subheader(label)
            return container.selectbox(label, options,
                                       label_visibility="collapsed", key=key)
        return container.selectbox(label, options, key=key)

    state_label = selectbox(state_container, "State", _available_states(),
                            key=f"{key_prefix}_state")
    state_id = {v: k for k, v in STATE_LABELS.items()}[state_label]

    store_ids = _available_stores(state_id)
    short_options = [STORE_SHORT_LABELS[s] for s in store_ids]
    store_short = selectbox(store_container, "Store", short_options,
                            key=f"{key_prefix}_store")
    store_id = store_ids[short_options.index(store_short)]

    return store_id, STORE_LABELS[store_id]


 
@st.cache_data
def run_pipeline(product_id, store_id, forecast_days,
                 lead_time, ordering_cost, holding_cost, service_level):
    item_df = get_single_item(sales_clean, product_id, store_id)
    if len(item_df) < 30:
        return None, None, None, None, None
    df_prophet = prepare_prophet_df(item_df)
    forecast = train_forecast(df_prophet, forecast_days)
    avg_price = item_df['sell_price'].mean()
    inv = calculate_inventory(forecast, df_prophet, avg_price,
                              lead_time_days=lead_time,
                              holding_cost=holding_cost,
                              ordering_cost=ordering_cost,
                              service_level=service_level)
    eval_results = evaluate_forecast(df_prophet, holdout_days=90)
    return item_df, df_prophet, forecast, inv, eval_results
 
tab1, tab2, tab3 = st.tabs(["📈 Single Product", "🔍 Compare Products", "📂 Saved Results"])
 
 
# Tab 1 — Single Product
with tab1:
 
    st.sidebar.header("① Product Selection")
    product_id, cat_id = render_product_selector("t1", sidebar=True)

    st.sidebar.header("① Store Selection")
    store_id, selected_store_label = render_store_selector("t1", sidebar=True)
 
    st.sidebar.header("Inventory Parameters")
    lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
    ordering_cost = st.sidebar.slider("Ordering Cost ($)", 1, 100, 10)
    holding_cost = st.sidebar.slider("Holding Cost (% of price/year)", 0.05, 0.50, 0.20)
    service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95, step=0.01,
                                      help="Probability of not stocking out during lead time. "
                                           "Higher service level → more safety stock.")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)
 
    if st.sidebar.button("Run Forecast", type="primary", key="run_t1"):

        try:
            with st.spinner("Training forecast model... this may take a moment"):
                item_df, df_prophet, forecast, inv, eval_results = run_pipeline(
                    product_id, store_id, forecast_days,
                    lead_time, ordering_cost, holding_cost, service_level)
        except ForecastingError as e:
            st.error(f"Forecasting failed for this product/store combination: {e}")
            st.stop()

        if inv is None:
            st.error("Not enough data for this product/store combination. "
                     "Please select another.")
            st.stop()

        assert item_df is not None and df_prophet is not None and forecast is not None

        try:
            run_id = save_results_to_db(
                item_id=product_id,
                store_id=store_id,
                inv=inv,
                eval_results=eval_results,
                forecast_days=forecast_days,
                lead_time=lead_time,
                ordering_cost=ordering_cost,
                holding_cost=holding_cost,
                service_level=service_level
            )
            st.toast(f"✅ Run saved to database (run #{run_id})", icon="💾")
        except DatabaseError as e:
            st.warning(f"Forecast computed successfully, but saving this run "
                       f"to the database failed: {e}")
 
        st.subheader(f"{CATEGORY_LABELS.get(cat_id, cat_id)} | "
                     f"{product_id} at {selected_store_label}")
 
        st.subheader("Inventory Policy Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Daily Demand", f"{inv['avg_daily_demand']:.2f} units")
        col2.metric("Safety Stock", f"{inv['safety_stock']:.1f} units")
        col3.metric("Reorder Point", f"{inv['rop']:.1f} units")
        col4.metric("EOQ", f"{inv['eoq']:.1f} units")
 
        st.subheader("Model Accuracy (90-Day Holdout Backtest)")
 
        if eval_results is None:
            st.warning("Not enough historical data to run a backtest for this item.")
        else:
            mape = eval_results['mape']
            rmse = eval_results['rmse']
            mase = eval_results.get('mase')
            avg_demand = inv['avg_daily_demand']
            is_intermittent = avg_demand < 1.5
 
            if mase is None:
                mase_label = "N/A"
            elif mase < 1.0:
                mase_label = f"✅ {mase:.2f}"
            elif mase < 1.2:
                mase_label = f"⚠️ {mase:.2f}"
            else:
                mase_label = f"🔴 {mase:.2f}"
 
            if mape is None:
                mape_label = "N/A"
            elif mape < 20:
                mape_label = f"{mape:.1f}%"
            elif mape < 40:
                mape_label = f"{mape:.1f}%"
            else:
                mape_label = f"{mape:.1f}%"
 
            acol1, acol2, acol3, acol4 = st.columns(4)
            acol1.metric("MASE (vs seasonal naive)", mase_label,
                         help="The headline accuracy metric. Below 1.0 means the model beats a "
                              "seasonal naive forecast ('next Tuesday looks like last Tuesday'). "
                              "Above 1.0 means it doesn't. Unlike MAPE, MASE is defined when "
                              "sales are zero, so it works on intermittent demand.")
            acol2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f} units",
                         help="Average error in units. Interpretable on the same scale as demand.")
            acol3.metric("MAPE (Mean Abs % Error)", mape_label,
                         help="Shown for reference. Unreliable on low-volume items and undefined "
                              "on zero-sales days, so it is not the metric to judge this on.")
            acol4.metric("Holdout Window", "Last 90 days",
                         help="Model trained on all data before this window, then tested against it.")
 
            if mase is not None and mase >= 1.0:
                st.warning(
                    f"**This model does not beat a seasonal naive baseline** (MASE {mase:.2f}). "
                    f"For this item, simply repeating what happened the same weekday last week "
                    f"would be about as accurate as the fitted Prophet model. That is a real "
                    f"result worth knowing: the forecast machinery is not adding value here, and "
                    f"the inventory policy below is only as good as its demand estimate."
                )
            elif mase is not None:
                st.success(
                    f"**Beats the seasonal naive baseline** (MASE {mase:.2f}). The model's average "
                    f"error is {(1 - mase) * 100:.0f}% smaller than simply repeating the same "
                    f"weekday from the previous week."
                )
 
            if is_intermittent:
                st.info(
                    f"**Intermittent demand** (avg {avg_demand:.2f} units/day). Read MASE and RMSE "
                    f"here, not MAPE. Being off by 1 unit on a day with 1 actual sale registers as "
                    f"100% error even though the absolute miss is tiny, which is why MAPE reads so "
                    f"high across this dataset. Across a 200-combination sweep of this catalog the "
                    f"median MAPE was 73.6% while the median RMSE was 0.79 units, describing the "
                    f"same forecasts. The M5 competition avoided MAPE for the same reason and "
                    f"scored on WRMSSE instead."
                )
 
            comp = eval_results['comparison']
            fig_eval, ax_eval = plt.subplots(figsize=(12, 3))
            ax_eval.plot(comp['ds'], comp['y'],
                         color='steelblue', label='Actual Sales', alpha=0.7)
            ax_eval.plot(comp['ds'], comp['yhat'],
                         color='red', linewidth=2, label='Model Prediction')
            ax_eval.fill_between(comp['ds'], comp['yhat_lower'], comp['yhat_upper'],
                                 alpha=0.15, color='red', label='Uncertainty Band')
            title_metric = f"MASE: {mase:.2f}" if mase is not None else (
                f"MAPE: {mape:.1f}%" if mape is not None else "no scoreable days")
            ax_eval.set_title(
                f'Backtest: Actual vs Predicted (last 90 days) — {title_metric}')
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
 
        st.subheader("Forecast Insights")
        future_f = inv['future_forecast']
        peak_day = future_f.loc[future_f['yhat'].idxmax()]
        low_day = future_f.loc[future_f['yhat'].idxmin()]
        total_forecast = future_f['yhat'].sum()
        avg_upper = future_f['yhat_upper'].mean()
 
        icol1, icol2, icol3, icol4 = st.columns(4)
        icol1.metric("Total Forecasted Demand", f"{total_forecast:.0f} units",
                     help=f"Over the next {forecast_days} days")
        icol2.metric("Peak Demand Day",
                     f"{peak_day['ds'].strftime('%b %d, %Y')}",
                     f"{peak_day['yhat']:.1f} units")
        icol3.metric("Lowest Demand Day",
                     f"{low_day['ds'].strftime('%b %d, %Y')}",
                     f"{low_day['yhat']:.1f} units")
        icol4.metric("Avg Worst-Case Daily Demand", f"{avg_upper:.2f} units",
                     help="Upper bound — use this for conservative planning")
 
        st.subheader("Demand Forecast")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        recent_history = df_prophet.tail(180)
        ax1.plot(recent_history['ds'], recent_history['y'],
                 color='steelblue', alpha=0.5, label='Historical Sales (last 180 days)')
        ax1.plot(future_f['ds'], future_f['yhat'],
                 color='red', linewidth=2, label='Forecast')
        ax1.fill_between(future_f['ds'], future_f['yhat_lower'], future_f['yhat_upper'],
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
 
        st.subheader("Summary")
        st.info(f"""
        **What this means for {product_id}:**
        - Over the next **{forecast_days} days**, expect to sell approximately **{total_forecast:.0f} units**
          at {selected_store_label}.
        - Demand peaks around **{peak_day['ds'].strftime('%B %d, %Y')}**
          at {peak_day['yhat']:.1f} units/day.
        - To maintain a **{service_level:.0%} service level**, keep at least **{inv['safety_stock']:.0f} units**
          as safety stock.
        - Place a new order when inventory hits **{inv['rop']:.0f} units**
          (covers {lead_time} day lead time + safety stock).
        - Order **{inv['eoq']:.0f} units** at a time to minimize total inventory costs.
        """)
 
        st.subheader("Inventory Simulation")
        np.random.seed(42)
        simulated_demand = np.random.poisson(inv['avg_daily_demand'], size=forecast_days)
        inventory_levels, orders = [], []
        current_inventory = inv['eoq']
 
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
 
 
# Tab 2 — Compare Products
with tab2:
 
    st.markdown("### Compare Two Products Side by Side")
    st.markdown(
        "Select two products and one shared store. The inventory parameter sliders "
        "in the sidebar apply to both — this isolates the effect of demand patterns "
        "while holding cost assumptions constant."
    )
 
    pcol1, pcol2 = st.columns(2)
 
    with pcol1:
        st.markdown("#### Product A")
        product_a_id, cat_a_id = render_product_selector("cmp_a", label_suffix=" A")

    with pcol2:
        st.markdown("#### Product B")
        product_b_id, cat_b_id = render_product_selector("cmp_b", label_suffix=" B",
                                                         cat_index=1)

    st.markdown("#### Shared Store")
    scol1, scol2, _ = st.columns([1, 1, 2])
    cmp_store_id, cmp_store_label = render_store_selector("cmp", containers=(scol1, scol2))
 
    st.divider()
 
    if st.button("Run Comparison", type="primary", key="run_cmp"):

        try:
            with st.spinner("Training two forecast models... this may take a moment"):
                res_a = run_pipeline(product_a_id, cmp_store_id, forecast_days,
                                     lead_time, ordering_cost, holding_cost, service_level)
                res_b = run_pipeline(product_b_id, cmp_store_id, forecast_days,
                                     lead_time, ordering_cost, holding_cost, service_level)
        except ForecastingError as e:
            st.error(f"Forecasting failed for one of these product/store combinations: {e}")
            st.stop()

        item_a, prophet_a, fc_a, inv_a, eval_a = res_a
        item_b, prophet_b, fc_b, inv_b, eval_b = res_b

        if inv_a is None or inv_b is None:
            st.error("One or both products don't have enough data for this store. "
                     "Try a different combination.")
            st.stop()

        assert prophet_a is not None and prophet_b is not None

        try:
            run_id_a = save_results_to_db(product_a_id, cmp_store_id, inv_a, eval_a,
                                           forecast_days, lead_time, ordering_cost, holding_cost,
                                           service_level=service_level)
            run_id_b = save_results_to_db(product_b_id, cmp_store_id, inv_b, eval_b,
                                           forecast_days, lead_time, ordering_cost, holding_cost,
                                           service_level=service_level)
            st.toast(f"✅ Both runs saved (run #{run_id_a} & #{run_id_b})", icon="💾")
        except DatabaseError as e:
            st.warning(f"Comparison computed successfully, but saving these runs "
                       f"to the database failed: {e}")
 
        st.subheader("📊 Head-to-Head Comparison")
 
        def mase_str(eval_res):
            if eval_res is None: return "N/A"
            m = eval_res.get('mase')
            if m is None: return "N/A"
            if m < 1.0:   return f"✅ {m:.2f}"
            elif m < 1.2: return f"⚠️ {m:.2f}"
            else:         return f"🔴 {m:.2f}"
 
        def mape_str(eval_res):
            if eval_res is None: return "N/A"
            m = eval_res['mape']
            if m is None: return "N/A"
            return f"{m:.1f}%"
 
        def rmse_str(eval_res):
            if eval_res is None: return "N/A"
            return f"{eval_res['rmse']:.2f} units"
 
        comparison_data = {
            "Metric": [
                "Avg Daily Demand",
                "Demand Std Dev",
                f"Total Forecast ({forecast_days} days)",
                "Safety Stock",
                "Reorder Point (ROP)",
                "Economic Order Qty (EOQ)",
                "MASE (vs seasonal naive)",
                "RMSE (Backtest)",
                "MAPE (Backtest)",
            ],
            f"Product A — {product_a_id}": [
                f"{inv_a['avg_daily_demand']:.2f} units/day",
                f"{inv_a['std_daily_demand']:.2f} units",
                f"{inv_a['future_forecast']['yhat'].sum():.0f} units",
                f"{inv_a['safety_stock']:.1f} units",
                f"{inv_a['rop']:.1f} units",
                f"{inv_a['eoq']:.1f} units",
                mase_str(eval_a),
                rmse_str(eval_a),
                mape_str(eval_a),
            ],
            f"Product B — {product_b_id}": [
                f"{inv_b['avg_daily_demand']:.2f} units/day",
                f"{inv_b['std_daily_demand']:.2f} units",
                f"{inv_b['future_forecast']['yhat'].sum():.0f} units",
                f"{inv_b['safety_stock']:.1f} units",
                f"{inv_b['rop']:.1f} units",
                f"{inv_b['eoq']:.1f} units",
                mase_str(eval_b),
                rmse_str(eval_b),
                mape_str(eval_b),
            ],
        }
 
        cmp_df = pd.DataFrame(comparison_data)
        st.dataframe(cmp_df.set_index("Metric"), use_container_width=True)
 
        demand_ratio = inv_a['avg_daily_demand'] / max(inv_b['avg_daily_demand'], 0.01)
        higher_label = product_a_id if demand_ratio >= 1 else product_b_id
        lower_label = product_b_id if demand_ratio >= 1 else product_a_id
        ratio_val = max(demand_ratio, 1 / max(demand_ratio, 0.01))
        higher_ss = product_a_id if inv_a['safety_stock'] >= inv_b['safety_stock'] else product_b_id
 
        st.info(
            f"**Key takeaway:** {higher_label} has {ratio_val:.1f}× higher average daily demand "
            f"than {lower_label}. Despite this, {higher_ss} requires more safety stock — "
            f"reflecting higher demand *variability* (σ), not just higher demand volume. "
            f"This is a core ISE insight: safety stock is driven by uncertainty, not mean demand."
        )
 
        st.subheader("Demand Forecasts")
        fig_cmp, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
 
        for ax, prophet_df, inv_x, pid, color in [
            (ax_a, prophet_a, inv_a, product_a_id, 'steelblue'),
            (ax_b, prophet_b, inv_b, product_b_id, 'tomato'),
        ]:
            future_f = inv_x['future_forecast']
            recent = prophet_df.tail(180)
            ax.plot(recent['ds'], recent['y'],
                    color=color, alpha=0.4, label='Historical (last 180d)')
            ax.plot(future_f['ds'], future_f['yhat'],
                    color=color, linewidth=2, label='Forecast')
            ax.fill_between(future_f['ds'],
                            future_f['yhat_lower'], future_f['yhat_upper'],
                            alpha=0.15, color=color, label='Uncertainty Band')
            ax.set_title(f'{pid}\n@ {cmp_store_label}', fontsize=10)
            ax.set_xlabel('Date')
            ax.set_ylabel('Units Sold')
            ax.legend(fontsize=8)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
 
        plt.tight_layout()
        st.pyplot(fig_cmp)
        st.caption(
            "Charts use independent y-axes so each product's pattern is clearly visible. "
            "Absolute scale differences are captured in the comparison table above."
        )
 
    else:
        st.info(
            "Select two products and a shared store above, then click "
            "**Run Comparison** to see them side by side."
        )
 
 
# Tab 3 — Saved Results
with tab3:
 
    st.markdown("### Saved Forecast & Inventory Runs")
    st.markdown(
        "Every time you run a forecast in Tab 1 or a comparison in Tab 2, "
        "the results are automatically saved to a local DuckDB database. "
        "Query, filter, and re-inspect past runs here — no re-running Prophet needed."
    )
 
    all_runs = load_all_runs(limit=200)
 
    if all_runs.empty:
        st.info("No saved runs yet. Run a forecast in **📈 Single Product** "
                "or **🔍 Compare Products** to populate this tab.")
        st.stop()
 
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Runs Saved", len(all_runs))
    k2.metric("Unique Products", int(all_runs['item_id'].nunique()))
    k3.metric("Unique Stores", int(all_runs['store_id'].nunique()))

    valid_mase = all_runs['mase'].dropna() if 'mase' in all_runs.columns else pd.Series(dtype=float)
    if not valid_mase.empty:
        beat_naive = (valid_mase < 1.0).sum()
        k4.metric("Median MASE", f"{valid_mase.median():.2f}",
                  help="Below 1.0 beats a seasonal naive forecast.")
        k5.metric("Beat naive baseline", f"{beat_naive}/{len(valid_mase)}",
                  help="Runs where the model outperformed seasonal naive.")
    else:
        k4.metric("Median MASE", "N/A",
                  help="Re-run the batch script to populate MASE for saved runs.")
        valid_mapes = all_runs['mape'].dropna()
        k5.metric("Median MAPE",
                  f"{valid_mapes.median():.1f}%" if not valid_mapes.empty else "N/A")
 
    st.divider()
 
    st.markdown("#### Filter Saved Runs")
    fcol1, fcol2, fcol3 = st.columns(3)
 
    with fcol1:
        store_options = ["All stores"] + sorted(all_runs['store_id'].unique().tolist())
        selected_store_filter = st.selectbox("Store", store_options, key="db_store_filter")
 
    with fcol2:
        item_options = ["All products"] + sorted(all_runs['item_id'].unique().tolist())
        selected_item_filter = st.selectbox("Product", item_options, key="db_item_filter")
 
    with fcol3:
        sort_options = {
            "Most recent first": ("run_at", False),
            "Highest avg demand": ("avg_daily_demand", False),
            "Lowest MASE (best fit)": ("mase", True),
            "Lowest MAPE": ("mape", True),
            "Highest EOQ": ("eoq", False),
        }
        selected_sort_label = st.selectbox("Sort by", list(sort_options.keys()), key="db_sort")
        sort_col, sort_asc = sort_options[selected_sort_label]
 
    filtered: pd.DataFrame = all_runs.copy()
    if selected_store_filter != "All stores":
        filtered = filtered[filtered['store_id'] == selected_store_filter]  # type: ignore[assignment]
    if selected_item_filter != "All products":
        filtered = filtered[filtered['item_id'] == selected_item_filter]  # type: ignore[assignment]
    filtered = filtered.sort_values(sort_col, ascending=sort_asc, na_position='last')
 
    st.markdown(f"**{len(filtered)} run(s) shown**")
 
    display_cols = {
        'run_id': 'Run #',
        'run_at': 'Timestamp',
        'item_id': 'Product',
        'store_id': 'Store',
        'forecast_days': 'Horizon (days)',
        'lead_time_days': 'Lead Time',
        'avg_daily_demand': 'Avg Demand/day',
        'safety_stock': 'Safety Stock',
        'rop': 'ROP',
        'eoq': 'EOQ',
        'mase': 'MASE',
        'rmse': 'RMSE',
        'mape': 'MAPE (%)',
    }
    st.dataframe(
        filtered[list(display_cols.keys())].rename(columns=display_cols),  # type: ignore[call-overload]
        use_container_width=True,
        hide_index=True
    )
 
    st.divider()
    st.markdown("#### Re-view Forecast for a Saved Run")
    st.caption("Select a run ID to re-render its forecast chart instantly — "
               "no Prophet re-training needed.")
 
    if not filtered.empty:
        run_id_options = filtered['run_id'].tolist()
        selected_run_id = st.selectbox(
            "Select Run #", run_id_options, key="db_run_select")
 
        if st.button("Load Forecast Chart", key="db_load_chart"):
            daily_df = load_run_forecast(selected_run_id)
            run_meta = filtered[filtered['run_id'] == selected_run_id].iloc[0]
 
            if daily_df.empty:
                st.warning("No daily forecast data found for this run.")
            else:
                fig_saved, ax_saved = plt.subplots(figsize=(12, 4))
                ax_saved.plot(daily_df['ds'], daily_df['yhat'],
                              color='steelblue', linewidth=2, label='Forecast (yhat)')
                ax_saved.fill_between(daily_df['ds'],
                                      daily_df['yhat_lower'], daily_df['yhat_upper'],
                                      alpha=0.2, color='steelblue', label='Uncertainty Band')
                ax_saved.axhline(y=run_meta['rop'], color='orange', linestyle='--',
                                 label=f"ROP ({run_meta['rop']:.1f} units)")
                ax_saved.axhline(y=run_meta['safety_stock'], color='red', linestyle='--',
                                 label=f"Safety Stock ({run_meta['safety_stock']:.1f} units)")
                ax_saved.set_title(
                    f"Saved Run #{selected_run_id} — "
                    f"{run_meta['item_id']} @ {STORE_LABELS.get(run_meta['store_id'], run_meta['store_id'])}"
                )
                ax_saved.set_xlabel('Date')
                ax_saved.set_ylabel('Units Sold')
                ax_saved.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_saved)
 
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Avg Daily Demand", f"{run_meta['avg_daily_demand']:.2f} units")
                rcol2.metric("Safety Stock", f"{run_meta['safety_stock']:.1f} units")
                rcol3.metric("Reorder Point", f"{run_meta['rop']:.1f} units")
                rcol4.metric("EOQ", f"{run_meta['eoq']:.1f} units")
 
                parts = []
                if 'mase' in run_meta.index and pd.notna(run_meta['mase']):
                    parts.append(f"MASE: {run_meta['mase']:.2f}")
                if pd.notna(run_meta['rmse']):
                    parts.append(f"RMSE: {run_meta['rmse']:.2f} units")
                if pd.notna(run_meta['mape']):
                    parts.append(f"MAPE: {run_meta['mape']:.1f}%")
                if parts:
                    st.caption("Model accuracy on this run — " + "  |  ".join(parts))