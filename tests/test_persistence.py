"""
Integration tests for the DuckDB persistence layer. Unlike the pure-function
tests elsewhere, these genuinely hit a database — but always a temp one
(via pytest's tmp_path fixture + monkeypatching pipeline.DB_PATH), never the
real data/inventory.duckdb, so running the test suite can never corrupt or
wipe real saved runs.
"""
import os

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


FAKE_EVAL = {'mape': 12.5, 'rmse': 1.1, 'mase': 0.83}


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
    assert row['mase'] == pytest.approx(0.83)


def test_mase_persists_as_null_when_undefined():
    """A backtest that couldn't produce a MASE (perfectly periodic training
    series, too little history) should store NULL rather than a placeholder
    number that would pollute aggregate statistics."""
    eval_no_mase = {'mape': 20.0, 'rmse': 2.0, 'mase': None}
    save_results_to_db('FOODS_1_003', 'CA_1', make_fake_inv(4.0), eval_no_mase,
                        90, 7, 10.0, 0.2, service_level=0.95)

    runs = load_all_runs()
    row = runs[runs['item_id'] == 'FOODS_1_003'].iloc[0]
    assert pd.isna(row['mase'])
    assert row['mape'] == pytest.approx(20.0)


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

# ---------------------------------------------------------------------------
# get_run_summary — aggregates must cover every run, not just a loaded page
# ---------------------------------------------------------------------------

def test_run_summary_counts_all_runs_not_just_the_load_limit():
    """Regression test: the Saved Results tab used to compute its totals from
    load_all_runs(limit=200), so a database with more runs than the limit
    reported the limit as the total. Summary figures must come from SQL over
    the whole table."""
    from pipeline import get_run_summary

    for i in range(250):
        save_results_to_db(f'ITEM_{i:03d}', 'CA_1', make_fake_inv(1.0 + i * 0.01),
                            {'mape': 50.0, 'rmse': 1.0, 'mase': 0.5 + i * 0.004},
                            90, 7, 10.0, 0.2, service_level=0.95)

    summary = get_run_summary()
    loaded = load_all_runs(limit=200)

    assert summary['total_runs'] == 250
    assert len(loaded) == 200, "load_all_runs should still respect its limit"
    assert summary['unique_items'] == 250
    assert summary['runs_with_mase'] == 250


def test_run_summary_median_differs_from_truncated_slice():
    """The summary should reflect the full distribution. With MASE rising
    monotonically across runs, the newest 200 have a different median than
    all 250, so equality would mean the summary is reading the slice."""
    from pipeline import get_run_summary

    for i in range(250):
        save_results_to_db(f'ITEM_{i:03d}', 'CA_1', make_fake_inv(1.0),
                            {'mape': 50.0, 'rmse': 1.0, 'mase': 0.2 + i * 0.01},
                            90, 7, 10.0, 0.2, service_level=0.95)

    summary = get_run_summary()
    truncated_median = load_all_runs(limit=200)['mase'].median()

    assert summary['median_mase'] != pytest.approx(truncated_median)


def test_run_summary_beat_naive_counts_only_sub_one_mase():
    from pipeline import get_run_summary

    for i, mase in enumerate([0.5, 0.9, 1.0, 1.5, 0.99]):
        save_results_to_db(f'ITEM_{i}', 'CA_1', make_fake_inv(1.0),
                            {'mape': 50.0, 'rmse': 1.0, 'mase': mase},
                            90, 7, 10.0, 0.2, service_level=0.95)

    summary = get_run_summary()
    assert summary['beat_naive'] == 3   # 0.5, 0.9, 0.99
    assert summary['runs_with_mase'] == 5


def test_run_summary_returns_none_for_empty_database():
    from pipeline import get_run_summary
    assert get_run_summary() is None


def test_run_summary_raises_database_error_on_corrupt_file(tmp_path, monkeypatch):
    from pipeline import get_run_summary

    corrupt = tmp_path / "corrupt.duckdb"
    corrupt.write_bytes(b"not a duckdb file")
    monkeypatch.setattr(pipeline, "DB_PATH", str(corrupt))

    with pytest.raises(DatabaseError):
        get_run_summary()


def test_data_path_resolution_prefers_full_then_demo(tmp_path, monkeypatch):
    """The deployed app has no data/ directory, only the committed demo_data/.
    Resolution must prefer the full dataset locally and fall back to the demo
    subset when it's absent."""
    monkeypatch.setattr(pipeline, "_REPO_ROOT", str(tmp_path))

    (tmp_path / "data").mkdir()
    (tmp_path / "demo_data").mkdir()

    # neither present -> returns the full path, so the error message points at
    # the file the user is expected to build
    assert pipeline._resolve_data_path('full.parquet', 'demo.parquet').endswith(
        os.path.join('data', 'full.parquet'))

    # only demo present -> falls back
    (tmp_path / "demo_data" / "demo.parquet").write_text("x")
    assert pipeline._resolve_data_path('full.parquet', 'demo.parquet').endswith(
        os.path.join('demo_data', 'demo.parquet'))

    # both present -> prefers full
    (tmp_path / "data" / "full.parquet").write_text("x")
    assert pipeline._resolve_data_path('full.parquet', 'demo.parquet').endswith(
        os.path.join('data', 'full.parquet'))


def test_run_summary_returns_none_when_table_exists_but_is_empty(tmp_path, monkeypatch):
    """Distinct from a missing database file: here the schema exists but no
    runs have been saved, so COUNT(*) is 0 and the aggregates are all NULL."""
    from pipeline import get_run_summary
    import duckdb as ddb

    db = tmp_path / "empty.duckdb"
    with ddb.connect(str(db)) as con:
        pipeline._init_db(con)
    monkeypatch.setattr(pipeline, "DB_PATH", str(db))

    assert get_run_summary() is None