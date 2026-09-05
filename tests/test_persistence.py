"""
Integration tests for the DuckDB persistence layer. Unlike the pure-function
tests elsewhere, these genuinely hit a database — but always a temp one
(via pytest's tmp_path fixture + monkeypatching pipeline.DB_PATH), never the
real data/inventory.duckdb, so running the test suite can never corrupt or
wipe real saved runs.
"""
import pandas as pd
import pytest

import pipeline
from pipeline import (save_results_to_db, load_all_runs, load_run_forecast,
                       query_runs, DatabaseError)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every test in this file at an isolated temp database file."""
    monkeypatch.setattr(pipeline, "DB_PATH", str(tmp_path / "test_inventory.duckdb"))


def make_fake_inv(avg_demand):
    return {
        'avg_daily_demand': avg_demand,
        'std_daily_demand': 1.5,
        'safety_stock': 12.0,
        'rop': 55.0,
        'eoq': 30.0,
        'future_forecast': pd.DataFrame({
            'ds': pd.date_range('2026-01-01', periods=3),
            'yhat': [avg_demand, avg_demand, avg_demand],
            'yhat_lower': [avg_demand - 1] * 3,
            'yhat_upper': [avg_demand + 1] * 3,
        })
    }


FAKE_EVAL = {'mape': 12.5, 'rmse': 1.1}


def test_save_and_load_round_trip():
    run_id = save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                                 forecast_days=90, lead_time=7, ordering_cost=10.0,
                                 holding_cost=0.2, service_level=0.90)

    runs = load_all_runs()
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row['run_id'] == run_id
    assert row['item_id'] == 'FOODS_1_001'
    assert row['store_id'] == 'CA_1'
    assert row['service_level'] == pytest.approx(0.90)
    assert row['avg_daily_demand'] == pytest.approx(6.0)


def test_run_ids_increment_across_saves():
    id_1 = save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                               90, 7, 10.0, 0.2, service_level=0.95)
    id_2 = save_results_to_db('FOODS_1_002', 'CA_1', make_fake_inv(3.0), FAKE_EVAL,
                               90, 7, 10.0, 0.2, service_level=0.95)
    assert id_2 == id_1 + 1


def test_load_run_forecast_returns_correct_daily_rows_for_that_run():
    id_1 = save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                               90, 7, 10.0, 0.2, service_level=0.95)
    id_2 = save_results_to_db('FOODS_1_002', 'CA_1', make_fake_inv(3.0), FAKE_EVAL,
                               90, 7, 10.0, 0.2, service_level=0.95)

    daily_1 = load_run_forecast(id_1)
    daily_2 = load_run_forecast(id_2)

    assert len(daily_1) == 3
    assert (daily_1['yhat'] == 6.0).all()
    assert (daily_2['yhat'] == 3.0).all()


def test_query_runs_filters_by_store():
    save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)
    save_results_to_db('FOODS_1_002', 'TX_1', make_fake_inv(3.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)

    ca_only = query_runs(store_id='CA_1')
    assert len(ca_only) == 1
    assert ca_only.iloc[0]['item_id'] == 'FOODS_1_001'


def test_query_runs_top_n_by_demand_limits_and_sorts():
    save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(3.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)
    save_results_to_db('FOODS_1_002', 'CA_1', make_fake_inv(8.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)
    save_results_to_db('FOODS_1_003', 'CA_1', make_fake_inv(5.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)

    top1 = query_runs(store_id='CA_1', top_n_by_demand=1)
    assert len(top1) == 1
    assert top1.iloc[0]['item_id'] == 'FOODS_1_002'  # the 8.0-demand item


def test_query_runs_rejects_sql_injection_attempt():
    """Regression test for the parameterized-query fix: a malicious item_id
    should be treated as a literal string, never as SQL that alters the
    query's logic (e.g. an OR '1'='1' style payload matching every row)."""
    save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                        90, 7, 10.0, 0.2, service_level=0.95)

    payload = "nonexistent' OR '1'='1"
    result = query_runs(item_id=payload)
    assert len(result) == 0


def test_load_all_runs_returns_empty_frame_when_no_db_exists():
    # temp_db fixture points at a path that doesn't exist yet — nothing has
    # been saved in this particular test, so the file genuinely isn't there.
    result = load_all_runs()
    assert result.empty


def test_query_runs_returns_empty_frame_when_no_db_exists():
    result = query_runs(store_id='CA_1')
    assert result.empty


def test_database_error_raised_on_corrupt_file(tmp_path, monkeypatch):
    corrupt_path = tmp_path / "corrupt.duckdb"
    corrupt_path.write_bytes(b"this is not a valid duckdb file")
    monkeypatch.setattr(pipeline, "DB_PATH", str(corrupt_path))

    with pytest.raises(DatabaseError):
        load_all_runs()


def test_save_results_to_db_raises_database_error_on_corrupt_file(tmp_path, monkeypatch):
    corrupt_path = tmp_path / "corrupt.duckdb"
    corrupt_path.write_bytes(b"this is not a valid duckdb file")
    monkeypatch.setattr(pipeline, "DB_PATH", str(corrupt_path))

    with pytest.raises(DatabaseError):
        save_results_to_db('FOODS_1_001', 'CA_1', make_fake_inv(6.0), FAKE_EVAL,
                            90, 7, 10.0, 0.2, service_level=0.95)


def test_load_run_forecast_raises_database_error_on_corrupt_file(tmp_path, monkeypatch):
    corrupt_path = tmp_path / "corrupt.duckdb"
    corrupt_path.write_bytes(b"this is not a valid duckdb file")
    monkeypatch.setattr(pipeline, "DB_PATH", str(corrupt_path))

    with pytest.raises(DatabaseError):
        load_run_forecast(1)


def test_query_runs_raises_database_error_on_corrupt_file(tmp_path, monkeypatch):
    corrupt_path = tmp_path / "corrupt.duckdb"
    corrupt_path.write_bytes(b"this is not a valid duckdb file")
    monkeypatch.setattr(pipeline, "DB_PATH", str(corrupt_path))

    with pytest.raises(DatabaseError):
        query_runs(store_id='CA_1')