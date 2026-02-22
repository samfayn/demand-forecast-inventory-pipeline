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