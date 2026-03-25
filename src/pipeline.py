import pandas as pd
import numpy as np
from prophet import Prophet
import duckdb
import os
from datetime import datetime

STORE_LABELS = {
    'CA_1': 'California - Store 1',
    'CA_2': 'California - Store 2',
    'CA_3': 'California - Store 3',
    'CA_4': 'California - Store 4',
    'TX_1': 'Texas - Store 1',
    'TX_2': 'Texas - Store 2',
    'TX_3': 'Texas - Store 3',
    'WI_1': 'Wisconsin - Store 1',
    'WI_2': 'Wisconsin - Store 2',
    'WI_3': 'Wisconsin - Store 3'
}

STATE_LABELS = {
    'CA': 'California',
    'TX': 'Texas',
    'WI': 'Wisconsin'
}

STORES_BY_STATE = {
    'CA': ['CA_1', 'CA_2', 'CA_3', 'CA_4'],
    'TX': ['TX_1', 'TX_2', 'TX_3'],
    'WI': ['WI_1', 'WI_2', 'WI_3']
}

STORE_SHORT_LABELS = {
    'CA_1': 'Store 1', 'CA_2': 'Store 2', 'CA_3': 'Store 3', 'CA_4': 'Store 4',
    'TX_1': 'Store 1', 'TX_2': 'Store 2', 'TX_3': 'Store 3',
    'WI_1': 'Store 1', 'WI_2': 'Store 2', 'WI_3': 'Store 3'
}

CATEGORY_LABELS = {
    'FOODS': 'Foods',
    'HOBBIES': 'Hobbies',
    'HOUSEHOLD': 'Household'
}

DEPT_LABELS = {
    'FOODS_1': 'Foods - Dept 1',
    'FOODS_2': 'Foods - Dept 2',
    'FOODS_3': 'Foods - Dept 3',
    'HOBBIES_1': 'Hobbies - Dept 1',
    'HOBBIES_2': 'Hobbies - Dept 2',
    'HOUSEHOLD_1': 'Household - Dept 1',
    'HOUSEHOLD_2': 'Household - Dept 2'
}

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'inventory.duckdb')


def load_data(parquet_path):
    return pd.read_parquet(parquet_path)


def get_single_item(df, product_id, store_id):
    return df[
        (df['item_id'] == product_id) &
        (df['store_id'] == store_id)
    ].copy()


def prepare_prophet_df(item_df):
    return item_df[['date', 'sales']].rename(columns={
        'date': 'ds',
        'sales': 'y'
    }).reset_index(drop=True)


def train_forecast(df_prophet, forecast_days=90):
    model = Prophet(
        yearly_seasonality=True,  # type: ignore[arg-type]
        weekly_seasonality=True,  # type: ignore[arg-type]
        daily_seasonality=False,  # type: ignore[arg-type]
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(df_prophet, holdout_days=90):
    """
    Holdout backtest: train on all data except the last holdout_days,
    then compare forecast to actuals. Returns None if not enough data.
    """
    if len(df_prophet) < holdout_days + 60:
        return None

    train_df = df_prophet.iloc[:-holdout_days].copy()
    actual_df = df_prophet.iloc[-holdout_days:].copy()

    model = Prophet(
        yearly_seasonality=True,  # type: ignore[arg-type]
        weekly_seasonality=True,  # type: ignore[arg-type]
        daily_seasonality=False,  # type: ignore[arg-type]
        changepoint_prior_scale=0.05
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=holdout_days)
    forecast = model.predict(future)

    holdout_forecast = forecast[forecast['ds'].isin(actual_df['ds'])][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()

    comparison = actual_df.merge(holdout_forecast, on='ds', how='inner')
    comparison['yhat'] = comparison['yhat'].clip(lower=0)

    nonzero = comparison[comparison['y'] > 0].copy()

    if len(nonzero) == 0:
        return None

    mape = (np.abs(nonzero['y'] - nonzero['yhat']) / nonzero['y']).mean() * 100
    rmse = np.sqrt(((comparison['y'] - comparison['yhat']) ** 2).mean())

    return {
        'mape': mape,
        'rmse': rmse,
        'comparison': comparison,
        'holdout_days': holdout_days
    }


def calculate_inventory(forecast, df_prophet, avg_price,
                         lead_time_days=7, service_level=0.95,
                         holding_cost=0.20, ordering_cost=10.0):
    z_score = 1.645  # 95% service level

    future_forecast = forecast[
        forecast['ds'] > df_prophet['ds'].max()
    ].copy()

    future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)
    future_forecast['yhat_upper'] = future_forecast['yhat_upper'].clip(lower=0)
    future_forecast['yhat_lower'] = future_forecast['yhat_lower'].clip(lower=0)

    avg_daily_demand = future_forecast['yhat'].mean()
    std_daily_demand = future_forecast['yhat'].std()

    safety_stock = z_score * std_daily_demand * np.sqrt(lead_time_days)
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    annual_demand = avg_daily_demand * 365
    eoq = np.sqrt((2 * annual_demand * ordering_cost) /
                  (holding_cost * avg_price))

    return {
        'avg_daily_demand': avg_daily_demand,
        'std_daily_demand': std_daily_demand,
        'safety_stock': safety_stock,
        'rop': rop,
        'eoq': eoq,
        'future_forecast': future_forecast
    }


def _init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id           INTEGER PRIMARY KEY,
            run_at           TIMESTAMP,
            item_id          VARCHAR,
            store_id         VARCHAR,
            forecast_days    INTEGER,
            lead_time_days   INTEGER,
            ordering_cost    DOUBLE,
            holding_cost     DOUBLE,
            avg_daily_demand DOUBLE,
            std_daily_demand DOUBLE,
            safety_stock     DOUBLE,
            rop              DOUBLE,
            eoq              DOUBLE,
            mape             DOUBLE,
            rmse             DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_daily (
            run_id     INTEGER,
            ds         DATE,
            yhat       DOUBLE,
            yhat_lower DOUBLE,
            yhat_upper DOUBLE
        )
    """)


def save_results_to_db(item_id, store_id, inv, eval_results,
                        forecast_days, lead_time, ordering_cost, holding_cost):
    con = duckdb.connect(DB_PATH)
    _init_db(con)

    # DuckDB doesn't support SERIAL/AUTOINCREMENT
    result = con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM forecast_runs").fetchone()
    run_id = result[0] if result is not None else 1

    mape = eval_results['mape'] if eval_results else None
    rmse = eval_results['rmse'] if eval_results else None

    con.execute("""
        INSERT INTO forecast_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id, datetime.now(), item_id, store_id,
        forecast_days, lead_time, ordering_cost, holding_cost,
        inv['avg_daily_demand'], inv['std_daily_demand'],
        inv['safety_stock'], inv['rop'], inv['eoq'],
        mape, rmse
    ])

    daily_df = inv['future_forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    daily_df.insert(0, 'run_id', run_id)
    con.execute("INSERT INTO forecast_daily SELECT * FROM daily_df")

    con.close()
    return run_id


def load_all_runs(limit=200):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT
            run_id, run_at, item_id, store_id,
            forecast_days, lead_time_days, ordering_cost, holding_cost,
            ROUND(avg_daily_demand, 2) AS avg_daily_demand,
            ROUND(std_daily_demand, 2) AS std_daily_demand,
            ROUND(safety_stock, 1)     AS safety_stock,
            ROUND(rop, 1)              AS rop,
            ROUND(eoq, 1)              AS eoq,
            ROUND(mape, 1)             AS mape,
            ROUND(rmse, 2)             AS rmse
        FROM forecast_runs
        ORDER BY run_at DESC
        LIMIT {limit}
    """).df()
    con.close()
    return df


def load_run_forecast(run_id):
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT ds, yhat, yhat_lower, yhat_upper
        FROM forecast_daily
        WHERE run_id = ?
        ORDER BY ds
    """, [run_id]).df()
    con.close()
    return df


def query_runs(item_id=None, store_id=None, top_n_by_demand=None):
    """
    Filter saved runs by item and/or store.

    Examples:
        query_runs(store_id='CA_1')
        query_runs(item_id='FOODS_1_001_CA_1')
        query_runs(store_id='TX_2', top_n_by_demand=10)
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    con = duckdb.connect(DB_PATH, read_only=True)

    filters = []
    if item_id:
        filters.append(f"item_id = '{item_id}'")
    if store_id:
        filters.append(f"store_id = '{store_id}'")
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    limit_clause = f"LIMIT {top_n_by_demand}" if top_n_by_demand else ""

    df = con.execute(f"""
        SELECT
            run_id, run_at, item_id, store_id,
            ROUND(avg_daily_demand, 2) AS avg_daily_demand,
            ROUND(safety_stock, 1)     AS safety_stock,
            ROUND(rop, 1)              AS rop,
            ROUND(eoq, 1)              AS eoq,
            ROUND(mape, 1)             AS mape,
            ROUND(rmse, 2)             AS rmse
        FROM forecast_runs
        {where_clause}
        ORDER BY avg_daily_demand DESC
        {limit_clause}
    """).df()
    con.close()
    return df
