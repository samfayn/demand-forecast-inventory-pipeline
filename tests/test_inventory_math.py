"""
Unit tests for calculate_inventory() — pure inventory math (EOQ, safety
stock, reorder point). No Prophet or DuckDB involved: these are plain
deterministic functions of a forecast DataFrame and some parameters, so
we build small synthetic forecast frames by hand rather than training a
real model or touching a database.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import calculate_inventory


def make_df_prophet(n_days=30, start='2025-01-01'):
    """A minimal historical actuals frame — only 'ds' matters for
    calculate_inventory, which uses it solely to find the cutoff date
    between history and future forecast."""
    dates = pd.date_range(start, periods=n_days)
    return pd.DataFrame({'ds': dates, 'y': [1.0] * n_days})


def make_future_forecast(df_prophet, daily_values):
    """A forecast frame containing only the future window, with the given
    per-day yhat values. yhat_lower/yhat_upper are set to a fixed +/-1 band
    since calculate_inventory doesn't use their exact values for EOQ/ROP/
    safety stock — only yhat itself feeds those formulas."""
    start = df_prophet['ds'].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=len(daily_values))
    return pd.DataFrame({
        'ds': dates,
        'yhat': daily_values,
        'yhat_lower': [v - 1 for v in daily_values],
        'yhat_upper': [v + 1 for v in daily_values],
    })


VARYING_DEMAND = [3.0, 5.0, 7.0, 4.0, 6.0, 8.0, 2.0]


def test_higher_service_level_increases_safety_stock():
    """Safety stock should grow with service level, since it's driven by
    the z-score for that confidence level (z increases as service level
    approaches 100%)."""
    df_prophet = make_df_prophet()
    forecast = make_future_forecast(df_prophet, VARYING_DEMAND)

    low = calculate_inventory(forecast, df_prophet, avg_price=10.0, service_level=0.80)
    high = calculate_inventory(forecast, df_prophet, avg_price=10.0, service_level=0.99)

    assert high['safety_stock'] > low['safety_stock']


def test_default_service_level_matches_95_percent_z_score():
    """Regression guard for the service_level bug fix: safety_stock at the
    default service_level should match the closed-form z=1.645 result,
    not some other hardcoded value."""
    df_prophet = make_df_prophet()
    forecast = make_future_forecast(df_prophet, VARYING_DEMAND)

    inv = calculate_inventory(forecast, df_prophet, avg_price=10.0, lead_time_days=7)

    # ddof=1 matches pandas.Series.std()'s default (sample std), which is
    # what calculate_inventory uses internally via future_forecast['yhat'].std()
    std = np.std(VARYING_DEMAND, ddof=1)
    expected_safety_stock = 1.645 * std * np.sqrt(7)
    assert inv['safety_stock'] == pytest.approx(expected_safety_stock, rel=1e-3)


def test_reorder_point_equals_lead_time_demand_plus_safety_stock():
    """ROP = (avg daily demand * lead time) + safety stock, per the README's
    stated formula — verify the implementation actually matches it."""
    df_prophet = make_df_prophet()
    forecast = make_future_forecast(df_prophet, VARYING_DEMAND)

    inv = calculate_inventory(forecast, df_prophet, avg_price=10.0, lead_time_days=7)

    expected_rop = inv['avg_daily_demand'] * 7 + inv['safety_stock']
    assert inv['rop'] == pytest.approx(expected_rop)


def test_eoq_decreases_as_holding_cost_increases():
    """EOQ = sqrt(2 * annual_demand * ordering_cost / (holding_cost * price)),
    so higher holding cost should mean smaller optimal order quantity."""
    df_prophet = make_df_prophet()
    forecast = make_future_forecast(df_prophet, VARYING_DEMAND)

    low_holding = calculate_inventory(forecast, df_prophet, avg_price=10.0, holding_cost=0.10)
    high_holding = calculate_inventory(forecast, df_prophet, avg_price=10.0, holding_cost=0.50)

    assert high_holding['eoq'] < low_holding['eoq']


def test_eoq_matches_closed_form_formula():
    df_prophet = make_df_prophet()
    forecast = make_future_forecast(df_prophet, VARYING_DEMAND)

    inv = calculate_inventory(forecast, df_prophet, avg_price=10.0,
                               holding_cost=0.20, ordering_cost=15.0)

    avg_daily_demand = np.mean(VARYING_DEMAND)
    annual_demand = avg_daily_demand * 365
    expected_eoq = np.sqrt((2 * annual_demand * 15.0) / (0.20 * 10.0))
    assert inv['eoq'] == pytest.approx(expected_eoq, rel=1e-6)


def test_negative_forecast_values_are_clipped_to_zero():
    """Negative demand isn't physically meaningful — calculate_inventory
    should clip yhat/yhat_lower/yhat_upper to >= 0 before using them."""
    df_prophet = make_df_prophet()
    # a forecast that dips negative on some days
    forecast = make_future_forecast(df_prophet, [-2.0, 5.0, -1.0, 4.0])

    inv = calculate_inventory(forecast, df_prophet, avg_price=10.0)

    assert (inv['future_forecast']['yhat'] >= 0).all()
    assert (inv['future_forecast']['yhat_lower'] >= 0).all()
    assert (inv['future_forecast']['yhat_upper'] >= 0).all()


def test_only_future_rows_are_used_not_historical():
    """calculate_inventory should filter forecast to ds > df_prophet['ds'].max()
    — if historical rows leaked into the calculation, avg_daily_demand would
    be skewed by them."""
    df_prophet = make_df_prophet()

    historical_leak = pd.DataFrame({
        'ds': df_prophet['ds'],           # same dates as history — should be excluded
        'yhat': [999.0] * len(df_prophet),  # obviously wrong value if leaked in
        'yhat_lower': [999.0] * len(df_prophet),
        'yhat_upper': [999.0] * len(df_prophet),
    })
    future = make_future_forecast(df_prophet, VARYING_DEMAND)
    forecast = pd.concat([historical_leak, future], ignore_index=True)

    inv = calculate_inventory(forecast, df_prophet, avg_price=10.0)

    assert inv['avg_daily_demand'] == pytest.approx(np.mean(VARYING_DEMAND))
    assert len(inv['future_forecast']) == len(VARYING_DEMAND)