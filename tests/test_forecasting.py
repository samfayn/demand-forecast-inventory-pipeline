import pandas as pd
import pytest

from pipeline import (compute_backtest_metrics, evaluate_forecast,
                       train_forecast, ForecastingError)


# ---------------------------------------------------------------------------
# compute_backtest_metrics — pure function, no Prophet involved
# ---------------------------------------------------------------------------

def test_mape_ignores_zero_actual_days():
    """MAPE is undefined at y=0 (division by zero), so days with zero actual
    sales must be excluded from the MAPE calculation — even though they still
    count toward RMSE."""
    comparison = pd.DataFrame({
        'y':    [10.0, 0.0,  5.0],
        'yhat': [ 8.0, 9.0,  5.0],   # the y=0 row has a huge prediction error
    })
    metrics = compute_backtest_metrics(comparison)

    # MAPE should reflect only the two nonzero-actual rows: (|10-8|/10 + |5-5|/5)/2 * 100 = 10.0
    assert metrics['mape'] == pytest.approx(10.0)

    # RMSE should include all three rows, including the y=0 outlier
    expected_rmse = ((2**2 + 9**2 + 0**2) / 3) ** 0.5
    assert metrics['rmse'] == pytest.approx(expected_rmse)


def test_returns_none_when_all_actuals_are_zero():
    comparison = pd.DataFrame({'y': [0.0, 0.0, 0.0], 'yhat': [1.0, 2.0, 0.5]})
    assert compute_backtest_metrics(comparison) is None


def test_perfect_prediction_gives_zero_mape_and_rmse():
    comparison = pd.DataFrame({'y': [5.0, 8.0, 3.0], 'yhat': [5.0, 8.0, 3.0]})
    metrics = compute_backtest_metrics(comparison)
    assert metrics['mape'] == pytest.approx(0.0)
    assert metrics['rmse'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ForecastingError — verify Prophet failures are translated, not left raw
# ---------------------------------------------------------------------------

class _FailingProphet:
    """Stands in for prophet.Prophet, but always raises on fit() — lets us
    deterministically test the failure path without depending on finding a
    real series that happens to break Prophet's optimizer."""
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, df):
        raise RuntimeError("simulated optimizer failure")


def test_train_forecast_wraps_prophet_failure(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "Prophet", _FailingProphet)

    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2025-01-01', periods=100),
        'y': [1.0] * 100,
    })

    with pytest.raises(ForecastingError):
        train_forecast(df_prophet)


def test_evaluate_forecast_wraps_prophet_failure(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "Prophet", _FailingProphet)

    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2025-01-01', periods=200),
        'y': [1.0] * 200,
    })

    with pytest.raises(ForecastingError):
        evaluate_forecast(df_prophet, holdout_days=90)


def test_evaluate_forecast_returns_none_for_insufficient_data():
    """This guard runs before Prophet is ever touched, so it should return
    None (not raise) regardless of whether Prophet would succeed or fail."""
    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2025-01-01', periods=50),
        'y': [1.0] * 50,
    })
    assert evaluate_forecast(df_prophet, holdout_days=90) is None


class _AllZeroHoldoutProphet:
    """A fake Prophet whose predictions land inside the holdout window but
    whose real-world counterpart would be a series with zero actual sales
    for the entire holdout — exercises evaluate_forecast's 'no nonzero days
    to compute MAPE over' branch without needing to find a real such series."""
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, df):
        self._last_ds = df['ds']

    def make_future_dataframe(self, periods):
        # extend by `periods` days past the training data's last date
        last = self._last_ds.max()
        future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=periods)
        return pd.DataFrame({'ds': pd.concat([self._last_ds, pd.Series(future_dates)],
                                              ignore_index=True)})

    def predict(self, future):
        return pd.DataFrame({
            'ds': future['ds'],
            'yhat': [0.0] * len(future),
            'yhat_lower': [0.0] * len(future),
            'yhat_upper': [0.0] * len(future),
        })


def test_evaluate_forecast_returns_none_when_holdout_actuals_all_zero(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "Prophet", _AllZeroHoldoutProphet)

    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2025-01-01', periods=200),
        'y': [1.0] * 110 + [0.0] * 90,  # holdout window (last 90 days) is all zero
    })
    assert evaluate_forecast(df_prophet, holdout_days=90) is None


# ---------------------------------------------------------------------------
# train_forecast — one real (slow) integration test against actual Prophet,
# to confirm the happy path still works end to end, not just the failure path
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_forecast_real_prophet_produces_expected_horizon():
    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=400),
        'y': [10 + (i % 7) for i in range(400)],  # mild weekly pattern
    })
    forecast = train_forecast(df_prophet, forecast_days=30)

    assert len(forecast) == 400 + 30
    assert 'yhat' in forecast.columns
    assert forecast['ds'].max() > df_prophet['ds'].max()


@pytest.mark.slow
def test_evaluate_forecast_real_prophet_success_path():
    """Confirms the full backtest path — real Prophet fit, real predict,
    real metric computation — returns a properly-shaped result, not just
    that the failure path is handled."""
    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=300),
        'y': [10 + (i % 7) for i in range(300)],
    })
    result = evaluate_forecast(df_prophet, holdout_days=90)

    assert result is not None
    assert 'mape' in result and 'rmse' in result
    assert result['mape'] >= 0
    assert result['rmse'] >= 0
    assert result['holdout_days'] == 90
    assert len(result['comparison']) <= 90