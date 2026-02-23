import pandas as pd
import numpy as np
from prophet import Prophet

# ── Label Mappings ────────────────────────────────────────────────────────
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

# State grouping for cascading store selection
STATE_LABELS = {
    'CA': 'California',
    'TX': 'Texas',
    'WI': 'Wisconsin'
}

# Maps state code → list of store IDs in that state
STORES_BY_STATE = {
    'CA': ['CA_1', 'CA_2', 'CA_3', 'CA_4'],
    'TX': ['TX_1', 'TX_2', 'TX_3'],
    'WI': ['WI_1', 'WI_2', 'WI_3']
}

# Short store labels for use inside a state context (e.g. "Store 1")
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


def load_data(parquet_path):
    """Load the cleaned sales data from parquet."""
    return pd.read_parquet(parquet_path)


def get_single_item(df, product_id, store_id):
    """Filter dataframe to a single product and store."""
    return df[
        (df['item_id'] == product_id) &
        (df['store_id'] == store_id)
    ].copy()


def prepare_prophet_df(item_df):
    """Convert item dataframe to Prophet format."""
    return item_df[['date', 'sales']].rename(columns={
        'date': 'ds',
        'sales': 'y'
    }).reset_index(drop=True)


def train_forecast(df_prophet, forecast_days=90):
    """Train Prophet model and return forecast."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(df_prophet, holdout_days=90):
    """
    Holdout backtest: train on all data except last holdout_days,
    forecast that window, and compare to actuals.

    Returns a dict with MAPE, RMSE, and a comparison dataframe.
    Returns None if there is not enough data to run a backtest.
    """
    if len(df_prophet) < holdout_days + 60:
        # Need at least 60 days of training data beyond the holdout
        return None

    train_df = df_prophet.iloc[:-holdout_days].copy()
    actual_df = df_prophet.iloc[-holdout_days:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=holdout_days)
    forecast = model.predict(future)

    # Align forecast to the holdout window
    holdout_forecast = forecast[forecast['ds'].isin(actual_df['ds'])][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()

    comparison = actual_df.merge(holdout_forecast, on='ds', how='inner')
    comparison['yhat'] = comparison['yhat'].clip(lower=0)

    # Filter to days with actual sales > 0 for MAPE (avoid division by zero)
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
    """Calculate inventory policy from forecast output."""
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