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
                      STORE_LABELS, STORES_BY_STATE, STATE_LABELS,
                      STORE_SHORT_LABELS, CATEGORY_LABELS, DEPT_LABELS)

st.set_page_config(page_title="Demand Forecast & Inventory Optimizer",
                   layout="wide")

st.title("Demand Forecast & Inventory Optimization Pipeline")
st.markdown("Built with Prophet forecasting and ISE inventory principles.")

@st.cache_data
def load_cached_data():
    df = load_data('data/sales_clean.parquet')
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

tab1, tab2, tab3 = st.tabs(["📈 Single Product", "🔍 Compare Products", "📂 Saved Results"])


# Tab 1 — Single Product
with tab1:

    st.sidebar.header("① Product Selection")

    st.sidebar.subheader("Category")
    selected_category_label = st.sidebar.selectbox(
        "Category", sorted(CATEGORY_LABELS.values()),
        label_visibility="collapsed", key="t1_cat")
    selected_cat_id = {v: k for k, v in CATEGORY_LABELS.items()}[selected_category_label]

    st.sidebar.subheader("Department")
    dept_options_raw = sorted([k for k in DEPT_LABELS if k.startswith(selected_cat_id)])
    dept_options_labels = [DEPT_LABELS[d] for d in dept_options_raw]
    selected_dept_label = st.sidebar.selectbox(
        "Department", dept_options_labels,
        label_visibility="collapsed", key="t1_dept")
    selected_dept_id = {v: k for k, v in DEPT_LABELS.items()}[selected_dept_label]

    st.sidebar.subheader("Product  (sorted by avg demand ↓)")
    dept_products = demand_summary[demand_summary['dept_id'] == selected_dept_id].copy()
    dept_products = dept_products.sort_values('avg_daily_demand', ascending=False)
    dept_products['label'] = (
        dept_products['item_id'] + "  —  " +
        dept_products['avg_daily_demand'].map(lambda x: f"{x:.2f} units/day")
    )
    selected_product_label = st.sidebar.selectbox(
        "Product", dept_products['label'].tolist(),
        label_visibility="collapsed", key="t1_prod")
    product_id = dept_products[
        dept_products['label'] == selected_product_label]['item_id'].values[0]
    cat_id = selected_cat_id

    st.sidebar.header("① Store Selection")

    st.sidebar.subheader("State")
    state_options = sorted(STATE_LABELS.values())
    selected_state_label = st.sidebar.selectbox(
        "State", state_options,
        label_visibility="collapsed", key="t1_state")
    selected_state_id = {v: k for k, v in STATE_LABELS.items()}[selected_state_label]

    st.sidebar.subheader("Store")
    store_ids_in_state = STORES_BY_STATE[selected_state_id]
    store_short_options = [STORE_SHORT_LABELS[s] for s in store_ids_in_state]
    selected_store_short = st.sidebar.selectbox(
        "Store", store_short_options,
        label_visibility="collapsed", key="t1_store")
    store_id = store_ids_in_state[store_short_options.index(selected_store_short)]
    selected_store_label = STORE_LABELS[store_id]

    st.sidebar.header("Inventory Parameters")
    lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
    ordering_cost = st.sidebar.slider("Ordering Cost ($)", 1, 100, 10)
    holding_cost = st.sidebar.slider("Holding Cost (% of price/year)", 0.05, 0.50, 0.20)
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)

    if st.sidebar.button("Run Forecast", type="primary", key="run_t1"):

        with st.spinner("Training forecast model... this may take a moment"):
            item_df, df_prophet, forecast, inv, eval_results = run_pipeline(
                product_id, store_id, forecast_days,
                lead_time, ordering_cost, holding_cost)

        if inv is None:
            st.error("Not enough data for this product/store combination. "
                     "Please select another.")
            st.stop()

        assert item_df is not None and df_prophet is not None and forecast is not None

        run_id = save_results_to_db(
            item_id=product_id,
            store_id=store_id,
            inv=inv,
            eval_results=eval_results,
            forecast_days=forecast_days,
            lead_time=lead_time,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost
        )
        st.toast(f"✅ Run saved to database (run #{run_id})", icon="💾")

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
            avg_demand = inv['avg_daily_demand']
            is_intermittent = avg_demand < 1.5

            if mape < 20:   mape_label = f"✅ {mape:.1f}%"
            elif mape < 40: mape_label = f"⚠️ {mape:.1f}%"
            else:           mape_label = f"🔴 {mape:.1f}%"

            acol1, acol2, acol3 = st.columns(3)
            acol1.metric("MAPE (Mean Abs % Error)", mape_label,
                         help="Lower is better. Measures average % error on non-zero sales days.")
            acol2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f} units",
                         help="Lower is better. Penalizes large errors more heavily than MAPE.")
            acol3.metric("Holdout Window", "Last 90 days",
                         help="Model trained on all data before this window, then tested against it.")

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
        - To maintain a **95% service level**, keep at least **{inv['safety_stock']:.0f} units**
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
        cat_a_label = st.selectbox(
            "Category A", sorted(CATEGORY_LABELS.values()), key="cmp_cat_a")
        cat_a_id = {v: k for k, v in CATEGORY_LABELS.items()}[cat_a_label]

        dept_a_raw = sorted([k for k in DEPT_LABELS if k.startswith(cat_a_id)])
        dept_a_label = st.selectbox(
            "Department A", [DEPT_LABELS[d] for d in dept_a_raw], key="cmp_dept_a")
        dept_a_id = {v: k for k, v in DEPT_LABELS.items()}[dept_a_label]

        prods_a = demand_summary[demand_summary['dept_id'] == dept_a_id].copy()
        prods_a = prods_a.sort_values('avg_daily_demand', ascending=False)
        prods_a['label'] = (
            prods_a['item_id'] + "  —  " +
            prods_a['avg_daily_demand'].map(lambda x: f"{x:.2f} units/day")
        )
        prod_a_label = st.selectbox(
            "Product A (sorted by avg demand ↓)", prods_a['label'].tolist(), key="cmp_prod_a")
        product_a_id = prods_a[prods_a['label'] == prod_a_label]['item_id'].values[0]

    with pcol2:
        st.markdown("#### Product B")
        cat_b_label = st.selectbox(
            "Category B", sorted(CATEGORY_LABELS.values()),
            index=1, key="cmp_cat_b")
        cat_b_id = {v: k for k, v in CATEGORY_LABELS.items()}[cat_b_label]

        dept_b_raw = sorted([k for k in DEPT_LABELS if k.startswith(cat_b_id)])
        dept_b_label = st.selectbox(
            "Department B", [DEPT_LABELS[d] for d in dept_b_raw], key="cmp_dept_b")
        dept_b_id = {v: k for k, v in DEPT_LABELS.items()}[dept_b_label]

        prods_b = demand_summary[demand_summary['dept_id'] == dept_b_id].copy()
        prods_b = prods_b.sort_values('avg_daily_demand', ascending=False)
        prods_b['label'] = (
            prods_b['item_id'] + "  —  " +
            prods_b['avg_daily_demand'].map(lambda x: f"{x:.2f} units/day")
        )
        prod_b_label = st.selectbox(
            "Product B (sorted by avg demand ↓)", prods_b['label'].tolist(), key="cmp_prod_b")
        product_b_id = prods_b[prods_b['label'] == prod_b_label]['item_id'].values[0]

    st.markdown("#### Shared Store")
    scol1, scol2, _ = st.columns([1, 1, 2])
    with scol1:
        cmp_state_label = st.selectbox(
            "State", sorted(STATE_LABELS.values()), key="cmp_state")
        cmp_state_id = {v: k for k, v in STATE_LABELS.items()}[cmp_state_label]
    with scol2:
        cmp_store_ids = STORES_BY_STATE[cmp_state_id]
        cmp_short_labels = [STORE_SHORT_LABELS[s] for s in cmp_store_ids]
        cmp_store_short = st.selectbox("Store", cmp_short_labels, key="cmp_store")
        cmp_store_id = cmp_store_ids[cmp_short_labels.index(cmp_store_short)]
        cmp_store_label = STORE_LABELS[cmp_store_id]

    st.divider()

    if st.button("Run Comparison", type="primary", key="run_cmp"):

        with st.spinner("Training two forecast models... this may take a moment"):
            res_a = run_pipeline(product_a_id, cmp_store_id, forecast_days,
                                 lead_time, ordering_cost, holding_cost)
            res_b = run_pipeline(product_b_id, cmp_store_id, forecast_days,
                                 lead_time, ordering_cost, holding_cost)

        item_a, prophet_a, fc_a, inv_a, eval_a = res_a
        item_b, prophet_b, fc_b, inv_b, eval_b = res_b

        if inv_a is None or inv_b is None:
            st.error("One or both products don't have enough data for this store. "
                     "Try a different combination.")
            st.stop()

        assert prophet_a is not None and prophet_b is not None

        run_id_a = save_results_to_db(product_a_id, cmp_store_id, inv_a, eval_a,
                                       forecast_days, lead_time, ordering_cost, holding_cost)
        run_id_b = save_results_to_db(product_b_id, cmp_store_id, inv_b, eval_b,
                                       forecast_days, lead_time, ordering_cost, holding_cost)
        st.toast(f"✅ Both runs saved (run #{run_id_a} & #{run_id_b})", icon="💾")

        st.subheader("📊 Head-to-Head Comparison")

        def mape_str(eval_res):
            if eval_res is None: return "N/A"
            m = eval_res['mape']
            if m < 20:   return f"✅ {m:.1f}%"
            elif m < 40: return f"⚠️ {m:.1f}%"
            else:        return f"🔴 {m:.1f}%"

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
                "MAPE (Backtest)",
                "RMSE (Backtest)",
            ],
            f"Product A — {product_a_id}": [
                f"{inv_a['avg_daily_demand']:.2f} units/day",
                f"{inv_a['std_daily_demand']:.2f} units",
                f"{inv_a['future_forecast']['yhat'].sum():.0f} units",
                f"{inv_a['safety_stock']:.1f} units",
                f"{inv_a['rop']:.1f} units",
                f"{inv_a['eoq']:.1f} units",
                mape_str(eval_a),
                rmse_str(eval_a),
            ],
            f"Product B — {product_b_id}": [
                f"{inv_b['avg_daily_demand']:.2f} units/day",
                f"{inv_b['std_daily_demand']:.2f} units",
                f"{inv_b['future_forecast']['yhat'].sum():.0f} units",
                f"{inv_b['safety_stock']:.1f} units",
                f"{inv_b['rop']:.1f} units",
                f"{inv_b['eoq']:.1f} units",
                mape_str(eval_b),
                rmse_str(eval_b),
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Runs Saved", len(all_runs))
    k2.metric("Unique Products", int(all_runs['item_id'].nunique()))
    k3.metric("Unique Stores", int(all_runs['store_id'].nunique()))
    valid_mapes = all_runs['mape'].dropna()
    k4.metric("Avg MAPE (where available)",
              f"{valid_mapes.mean():.1f}%" if not valid_mapes.empty else "N/A")

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
            "Lowest MAPE (best fit)": ("mape", True),
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
        'mape': 'MAPE (%)',
        'rmse': 'RMSE',
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

                mape_val = run_meta['mape']
                if pd.notna(mape_val):
                    st.caption(f"Model accuracy on this run — MAPE: {mape_val:.1f}%  |  "
                               f"RMSE: {run_meta['rmse']:.2f} units")
