import pandas as pd
import numpy as np
from scipy.stats import norm
from prophet import Prophet
import duckdb
import os
from datetime import datetime


class ForecastingError(Exception):
    """Raised when Prophet model fitting or prediction fails."""


class DatabaseError(Exception):
    """Raised when a DuckDB read or write operation fails."""

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

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')


def _resolve_data_path(full_name, demo_name):
    """
    Prefer the full local dataset, fall back to the committed demo subset.

    data/ holds the real 46M-row output of data_prep.py and is gitignored, so
    it exists on a developer machine but never in a fresh clone or on
    Streamlit Community Cloud. demo_data/ holds a small committed subset
    covering the item/store pairs that have saved forecast runs, which is what
    the deployed app runs on. Returning the full path when neither exists
    keeps the resulting error message pointing at the thing the user is
    expected to build.
    """
    full = os.path.join(_REPO_ROOT, 'data', full_name)
    demo = os.path.join(_REPO_ROOT, 'demo_data', demo_name)
    if os.path.exists(full):
        return full
    if os.path.exists(demo):
        return demo
    return full


DB_PATH = _resolve_data_path('inventory.duckdb', 'inventory_demo.duckdb')
PARQUET_PATH = _resolve_data_path('sales_clean.parquet', 'sales_demo.parquet')

# True when running against the trimmed demo subset rather than the full
# dataset, so the UI can say so instead of implying it has everything.
USING_DEMO_DATA = 'demo_data' in PARQUET_PATH


def load_data(parquet_path):
    return pd.read_parquet(parquet_path)


def get_single_item(df, product_id, store_id):
    return df[
        (df['item_id'] == product_id) &
        (df['store_id'] == store_id)
    ].copy()


def build_series_index(df):
    """
    Build a sorted MultiIndex on (item_id, store_id) for fast repeated lookups.

    get_single_item scans the whole frame on every call, which is fine for the
    dashboard (one selection at a time) but costs roughly a second per lookup
    on a 46M-row frame. A batch sweep doing hundreds of lookups pays that
    repeatedly. Sorting once up front turns each subsequent lookup into a
    binary search.

    Use with get_indexed_item(). The one-time sort costs a few seconds and
    pays for itself after a handful of lookups.
    """
    return df.set_index(['item_id', 'store_id']).sort_index()


def get_indexed_item(indexed_df, product_id, store_id):
    """
    Look up one product/store series from a frame prepared by
    build_series_index(). Returns an empty DataFrame if the combination
    isn't present, matching get_single_item's behavior rather than raising.
    """
    try:
        result = indexed_df.loc[[(product_id, store_id)]]
    except KeyError:
        return pd.DataFrame(columns=indexed_df.columns)
    return result.reset_index()


def prepare_prophet_df(item_df):
    return item_df[['date', 'sales']].rename(columns={
        'date': 'ds',
        'sales': 'y'
    }).reset_index(drop=True)


def train_forecast(df_prophet, forecast_days=90):
    try:
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
    except Exception as e:
        raise ForecastingError(
            f"Prophet failed to fit or predict for this product/store "
            f"combination ({len(df_prophet)} data points): {e}"
        ) from e


def compute_mase(actual, predicted, train_series, seasonal_period=7):
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales the model's mean absolute error by the mean absolute error of a
    seasonal naive forecast ("next Tuesday looks like last Tuesday") computed
    on the *training* data. The result reads directly as a comparison against
    that baseline:

        MASE < 1  -> the model beats seasonal naive
        MASE = 1  -> the model matches it
        MASE > 1  -> the model is worse than doing nothing clever

    Unlike MAPE, MASE is defined when actuals are zero, which is why it's the
    standard recommendation for intermittent demand. seasonal_period defaults
    to 7 for daily retail data with weekly seasonality.

    Returns None when the scaling factor can't be computed: too little
    training history, or a perfectly periodic training series (a flat or
    all-zero series makes the naive error zero, so the ratio is undefined).
    """
    train_series = np.asarray(train_series, dtype=float)

    if len(train_series) <= seasonal_period:
        return None

    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    scale = naive_errors.mean()

    if scale == 0 or not np.isfinite(scale):
        return None

    mae = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)).mean()
    return mae / scale


def compute_backtest_metrics(comparison, train_series=None, seasonal_period=7):
    """
    Pure function: given a comparison DataFrame with 'y' (actual) and 'yhat'
    (predicted, already clipped to >= 0) columns, computes MAPE, RMSE and
    (when train_series is supplied) MASE.

    MAPE is computed only over rows where actual sales > 0, since percentage
    error is undefined at zero and intermittent retail demand contains many
    zero-sales days; it comes back as None when no such rows exist. RMSE and
    MASE use all rows and are reported regardless, so a metric that is well
    defined isn't discarded just because MAPE isn't.

    Returns None only when there is nothing to score at all.
    """
    if len(comparison) == 0:
        return None

    nonzero = comparison[comparison['y'] > 0]
    if len(nonzero) > 0:
        mape = (np.abs(nonzero['y'] - nonzero['yhat']) / nonzero['y']).mean() * 100
    else:
        mape = None

    rmse = np.sqrt(((comparison['y'] - comparison['yhat']) ** 2).mean())

    mase = None
    if train_series is not None:
        mase = compute_mase(comparison['y'], comparison['yhat'],
                            train_series, seasonal_period=seasonal_period)

    return {'mape': mape, 'rmse': rmse, 'mase': mase}


def evaluate_forecast(df_prophet, holdout_days=90, seasonal_period=7):
    """
    Holdout backtest: train on all data except the last holdout_days,
    then compare forecast to actuals. Returns None if not enough data.
    """
    if len(df_prophet) < holdout_days + 60:
        return None

    train_df = df_prophet.iloc[:-holdout_days].copy()
    actual_df = df_prophet.iloc[-holdout_days:].copy()

    try:
        model = Prophet(
            yearly_seasonality=True,  # type: ignore[arg-type]
            weekly_seasonality=True,  # type: ignore[arg-type]
            daily_seasonality=False,  # type: ignore[arg-type]
            changepoint_prior_scale=0.05
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=holdout_days)
        forecast = model.predict(future)
    except Exception as e:
        raise ForecastingError(
            f"Prophet failed during backtest fit/predict "
            f"({len(train_df)} training points): {e}"
        ) from e

    holdout_forecast = forecast[forecast['ds'].isin(actual_df['ds'])][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()

    comparison = actual_df.merge(holdout_forecast, on='ds', how='inner')
    comparison['yhat'] = comparison['yhat'].clip(lower=0)

    metrics = compute_backtest_metrics(comparison, train_series=train_df['y'],
                                       seasonal_period=seasonal_period)
    if metrics is None:
        return None

    return {
        'mape': metrics['mape'],
        'rmse': metrics['rmse'],
        'mase': metrics['mase'],
        'comparison': comparison,
        'holdout_days': holdout_days
    }


def calculate_inventory(forecast, df_prophet, avg_price,
                         lead_time_days=7, service_level=0.95,
                         holding_cost=0.20, ordering_cost=10.0):
    z_score = norm.ppf(service_level)

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
            service_level    DOUBLE,
            avg_daily_demand DOUBLE,
            std_daily_demand DOUBLE,
            safety_stock     DOUBLE,
            rop              DOUBLE,
            eoq              DOUBLE,
            mape             DOUBLE,
            rmse             DOUBLE,
            mase             DOUBLE
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
                        forecast_days, lead_time, ordering_cost, holding_cost,
                        service_level=0.95):
    try:
        with duckdb.connect(DB_PATH) as con:
            _init_db(con)

            # DuckDB doesn't support SERIAL/AUTOINCREMENT
            result = con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM forecast_runs").fetchone()
            run_id = result[0] if result is not None else 1

            mape = eval_results['mape'] if eval_results else None
            rmse = eval_results['rmse'] if eval_results else None
            mase = eval_results.get('mase') if eval_results else None

            con.execute("""
                INSERT INTO forecast_runs (
                    run_id, run_at, item_id, store_id,
                    forecast_days, lead_time_days, ordering_cost, holding_cost, service_level,
                    avg_daily_demand, std_daily_demand, safety_stock, rop, eoq, mape, rmse, mase
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, datetime.now(), item_id, store_id,
                forecast_days, lead_time, ordering_cost, holding_cost, service_level,
                inv['avg_daily_demand'], inv['std_daily_demand'],
                inv['safety_stock'], inv['rop'], inv['eoq'],
                mape, rmse, mase
            ])

            daily_df = inv['future_forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            daily_df.insert(0, 'run_id', run_id)
            con.execute("INSERT INTO forecast_daily SELECT * FROM daily_df")

        return run_id
    except duckdb.Error as e:
        raise DatabaseError(f"Failed to save forecast run to database: {e}") from e


def get_run_summary():
    """
    Aggregate statistics over *every* saved run, computed in SQL.

    load_all_runs() caps how many rows it returns so the UI isn't rendering an
    unbounded table. Computing summary figures from that capped result would
    silently describe only the most recent page rather than the whole history,
    so these are aggregated database-side where the limit doesn't apply.

    Returns None when there's no database or no runs yet.
    """
    if not os.path.exists(DB_PATH):
        return None

    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            row = con.execute("""
                SELECT
                    COUNT(*)                                       AS total_runs,
                    COUNT(DISTINCT item_id)                        AS unique_items,
                    COUNT(DISTINCT store_id)                       AS unique_stores,
                    COUNT(mase)                                    AS runs_with_mase,
                    MEDIAN(mase)                                   AS median_mase,
                    COUNT(*) FILTER (WHERE mase < 1.0)             AS beat_naive,
                    COUNT(mape)                                    AS runs_with_mape,
                    MEDIAN(mape)                                   AS median_mape,
                    MEDIAN(rmse)                                   AS median_rmse
                FROM forecast_runs
            """).fetchone()
    except duckdb.Error as e:
        raise DatabaseError(f"Failed to summarize saved runs: {e}") from e

    if row is None or row[0] == 0:
        return None

    keys = ['total_runs', 'unique_items', 'unique_stores', 'runs_with_mase',
            'median_mase', 'beat_naive', 'runs_with_mape', 'median_mape',
            'median_rmse']
    return dict(zip(keys, row))


def load_all_runs(limit=200):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            df = con.execute(f"""
                SELECT
                    run_id, run_at, item_id, store_id,
                    forecast_days, lead_time_days, ordering_cost, holding_cost, service_level,
                    ROUND(avg_daily_demand, 2) AS avg_daily_demand,
                    ROUND(std_daily_demand, 2) AS std_daily_demand,
                    ROUND(safety_stock, 1)     AS safety_stock,
                    ROUND(rop, 1)              AS rop,
                    ROUND(eoq, 1)              AS eoq,
                    ROUND(mape, 1)             AS mape,
                    ROUND(rmse, 2)             AS rmse,
                    ROUND(mase, 3)             AS mase
                FROM forecast_runs
                ORDER BY run_at DESC
                LIMIT {limit}
            """).df()
        return df
    except duckdb.Error as e:
        raise DatabaseError(f"Failed to load saved runs: {e}") from e


def load_run_forecast(run_id):
    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            df = con.execute("""
                SELECT ds, yhat, yhat_lower, yhat_upper
                FROM forecast_daily
                WHERE run_id = ?
                ORDER BY ds
            """, [run_id]).df()
        return df
    except duckdb.Error as e:
        raise DatabaseError(f"Failed to load forecast for run #{run_id}: {e}") from e


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

    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            filters = []
            params = []
            if item_id:
                filters.append("item_id = ?")
                params.append(item_id)
            if store_id:
                filters.append("store_id = ?")
                params.append(store_id)
            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

            limit_clause = ""
            if top_n_by_demand is not None:
                limit_clause = "LIMIT ?"
                params.append(int(top_n_by_demand))

            df = con.execute(f"""
                SELECT
                    run_id, run_at, item_id, store_id,
                    ROUND(avg_daily_demand, 2) AS avg_daily_demand,
                    ROUND(safety_stock, 1)     AS safety_stock,
                    ROUND(rop, 1)              AS rop,
                    ROUND(eoq, 1)              AS eoq,
                    ROUND(mape, 1)             AS mape,
                    ROUND(rmse, 2)             AS rmse,
                    ROUND(mase, 3)             AS mase
                FROM forecast_runs
                {where_clause}
                ORDER BY avg_daily_demand DESC
                {limit_clause}
            """, params).df()
        return df
    except duckdb.Error as e:
        raise DatabaseError(f"Failed to query saved runs: {e}") from e