"""
Tests for the analytical SQL in sql/.

These build a small synthetic Parquet and forecast_runs table with known
properties, then check that each query runs and returns the shape and the
relationships it claims to. The point is not to re-derive the arithmetic in
Python (that would just restate the SQL) but to catch queries that stop
parsing, reference dropped columns, or silently return nothing after a schema
change.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from analysis import run_query, load_query, QUERIES, AnalysisError


@pytest.fixture
def synthetic_parquet(tmp_path):
    """Three items with deliberately different demand patterns: one
    continuous high-volume, one moderate, one highly intermittent."""
    rng = np.random.default_rng(3)
    dates = pd.date_range('2024-01-01', periods=200)

    specs = [
        ('FOODS_1_001', 'FOODS', 'FOODS_1', 12.0, 2.0, 3.0),
        ('FOODS_1_002', 'FOODS', 'FOODS_1', 3.0, 1.0, 2.0),
        ('HOBBIES_1_001', 'HOBBIES', 'HOBBIES_1', 0.15, 0.4, 9.0),
    ]

    frames = []
    for store in ['CA_1', 'TX_1']:
        for item, cat, dept, base, noise, price in specs:
            weekly = 1 + 0.3 * np.sin(np.arange(200) * 2 * np.pi / 7)
            vals = np.maximum(0, rng.normal(base * weekly, noise)).round()
            frames.append(pd.DataFrame({
                'item_id': item, 'store_id': store, 'state_id': store.split('_')[0],
                'cat_id': cat, 'dept_id': dept, 'date': dates,
                'sales': vals, 'sell_price': price,
            }))

    path = tmp_path / 'sales.parquet'
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_db(tmp_path):
    """A forecast_runs table with a known MASE spread: high-volume items
    beat naive, the intermittent one does not."""
    path = tmp_path / 'inventory.duckdb'
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE forecast_runs (
            run_id INTEGER PRIMARY KEY, run_at TIMESTAMP,
            item_id VARCHAR, store_id VARCHAR,
            forecast_days INTEGER, lead_time_days INTEGER,
            ordering_cost DOUBLE, holding_cost DOUBLE, service_level DOUBLE,
            avg_daily_demand DOUBLE, std_daily_demand DOUBLE,
            safety_stock DOUBLE, rop DOUBLE, eoq DOUBLE,
            mape DOUBLE, rmse DOUBLE, mase DOUBLE
        )
    """)

    rows = [
        (1, 'FOODS_1_001', 'CA_1', 12.0, 2.0, 25.0, 2.1, 0.70),
        (2, 'FOODS_1_002', 'CA_1', 3.0, 1.0, 40.0, 1.0, 0.85),
        (3, 'HOBBIES_1_001', 'CA_1', 0.15, 0.35, 95.0, 0.4, 1.30),
        (4, 'FOODS_1_001', 'TX_1', 11.5, 2.2, 27.0, 2.2, 0.75),
        (5, 'FOODS_1_002', 'TX_1', 3.2, 1.1, 42.0, 1.1, 0.90),
        (6, 'HOBBIES_1_001', 'TX_1', 0.18, 0.40, 99.0, 0.5, 1.25),
    ]
    for run_id, item, store, demand, std, mape, rmse, mase in rows:
        con.execute(
            "INSERT INTO forecast_runs VALUES (?, now(), ?, ?, 90, 7, 10.0, 0.2, 0.95, "
            "?, ?, ?, ?, ?, ?, ?, ?)",
            [run_id, item, store, demand, std, std * 1.645 * 2.65,
             demand * 7 + std * 4.35, 30.0, mape, rmse, mase])
    con.close()
    return str(path)


# ---------------------------------------------------------------------------
# Loader behavior
# ---------------------------------------------------------------------------

def test_every_registered_query_has_a_sql_file():
    """QUERIES and sql/ must not drift apart."""
    for name in QUERIES:
        assert load_query(name).strip(), f"{name}.sql is empty or missing"


def test_every_sql_file_is_registered():
    sql_dir = os.path.join(os.path.dirname(__file__), '..', 'sql')
    on_disk = {f[:-4] for f in os.listdir(sql_dir) if f.endswith('.sql')}
    assert on_disk == set(QUERIES), (
        f"sql/ and QUERIES disagree: only on disk {on_disk - set(QUERIES)}, "
        f"only registered {set(QUERIES) - on_disk}")


def test_unknown_query_raises():
    with pytest.raises(AnalysisError):
        run_query('no_such_query')


def test_missing_parquet_raises_actionable_error(tmp_path, synthetic_db):
    with pytest.raises(AnalysisError, match="data_prep"):
        run_query('abc_classification', db_path=synthetic_db,
                  parquet_path=str(tmp_path / 'absent.parquet'))


def test_missing_database_raises_actionable_error(tmp_path, synthetic_parquet):
    with pytest.raises(AnalysisError, match="run_batch_forecasts"):
        run_query('when_model_beats_naive', db_path=str(tmp_path / 'absent.duckdb'),
                  parquet_path=synthetic_parquet)


# ---------------------------------------------------------------------------
# Individual queries
# ---------------------------------------------------------------------------

def test_abc_classification_partitions_the_whole_catalog(synthetic_parquet, synthetic_db):
    df = run_query('abc_classification', db_path=synthetic_db,
                   parquet_path=synthetic_parquet)

    assert not df.empty
    assert set(df['abc_class']) <= {'A', 'B', 'C'}
    # every item/store pair lands in exactly one class: 3 items x 2 stores
    assert df['item_store_pairs'].sum() == 6
    # shares are percentages of a whole
    assert df['pct_of_catalog'].sum() == pytest.approx(100.0, abs=0.2)
    assert df['pct_of_revenue'].sum() == pytest.approx(100.0, abs=0.2)


def test_abc_classes_are_ordered_by_value(synthetic_parquet, synthetic_db):
    """A items should carry more revenue per pair than C items — that's the
    entire premise of the classification."""
    df = run_query('abc_classification', db_path=synthetic_db,
                   parquet_path=synthetic_parquet).set_index('abc_class')

    if 'A' in df.index and 'C' in df.index:
        assert df.loc['A', 'avg_daily_units'] > df.loc['C', 'avg_daily_units']


def test_weekly_seasonality_returns_seven_days_per_category(synthetic_parquet, synthetic_db):
    df = run_query('weekly_seasonality_by_category', db_path=synthetic_db,
                   parquet_path=synthetic_parquet)

    assert not df.empty
    for cat, group in df.groupby('cat_id'):
        assert len(group) == 7, f"{cat} should have one row per weekday"
        # the index is relative to each category's own mean, so it centers on 100
        assert group['demand_index'].mean() == pytest.approx(100.0, abs=1.0)


def test_volume_quartiles_split_runs_into_four_groups(synthetic_db, synthetic_parquet):
    df = run_query('when_model_beats_naive', db_path=synthetic_db,
                   parquet_path=synthetic_parquet)

    assert len(df) == 4
    assert df['runs'].sum() == 6
    assert list(df['volume_quartile']) == [1, 2, 3, 4]
    # quartile boundaries must be monotonic
    assert df['max_demand'].is_monotonic_increasing


def test_higher_volume_quartiles_forecast_better(synthetic_db, synthetic_parquet):
    """The fixture is built so high-volume items have low MASE and the
    intermittent one has MASE > 1. The query should surface that."""
    df = run_query('when_model_beats_naive', db_path=synthetic_db,
                   parquet_path=synthetic_parquet)

    assert df.iloc[-1]['median_mase'] < df.iloc[0]['median_mase']


def test_store_quality_returns_best_and_worst_per_store(synthetic_db, synthetic_parquet):
    df = run_query('store_forecast_quality', db_path=synthetic_db,
                   parquet_path=synthetic_parquet, n_per_store=1)

    assert set(df['store_id']) == {'CA_1', 'TX_1'}
    assert set(df['end_of_range']) <= {'best', 'worst'}
    # with 3 runs per store and n=1, each store contributes a best and a worst
    for store, group in df.groupby('store_id'):
        assert len(group) == 2
        assert group.iloc[0]['mase'] < group.iloc[-1]['mase']


def test_store_quality_verdict_matches_mase(synthetic_db, synthetic_parquet):
    df = run_query('store_forecast_quality', db_path=synthetic_db,
                   parquet_path=synthetic_parquet, n_per_store=3)

    for _, row in df.iterrows():
        expected = 'beats naive' if row['mase'] < 1.0 else 'loses to naive'
        assert row['verdict'] == expected


def test_accuracy_vs_demand_pattern_joins_both_sources(synthetic_db, synthetic_parquet):
    """This query is the one that needs Parquet and DuckDB together; a broken
    join would show up as an empty result or lost rows."""
    df = run_query('accuracy_vs_demand_pattern', db_path=synthetic_db,
                   parquet_path=synthetic_parquet)

    assert not df.empty
    assert df['runs'].sum() == 6, "join dropped or duplicated runs"
    assert df['avg_zero_day_pct'].is_monotonic_increasing


def test_all_queries_execute_without_error(synthetic_db, synthetic_parquet):
    """Smoke test over the whole set, so adding a query without wiring it up
    correctly fails here rather than in front of someone."""
    for name in QUERIES:
        df = run_query(name, db_path=synthetic_db, parquet_path=synthetic_parquet)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, f"{name} returned no rows"


# ---------------------------------------------------------------------------
# CLI — invoked as a real subprocess, matching how it's documented
# ---------------------------------------------------------------------------

import subprocess
import sys as _sys

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis.py')


def test_cli_list_shows_every_query():
    result = subprocess.run([_sys.executable, SCRIPT, '--list'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    for name in QUERIES:
        assert name in result.stdout


def test_cli_runs_a_single_query(synthetic_db, synthetic_parquet):
    result = subprocess.run(
        [_sys.executable, SCRIPT, '--query', 'when_model_beats_naive',
         '--db', synthetic_db, '--parquet', synthetic_parquet],
        capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert 'volume_quartile' in result.stdout


def test_cli_runs_all_queries(synthetic_db, synthetic_parquet):
    result = subprocess.run(
        [_sys.executable, SCRIPT, '--db', synthetic_db, '--parquet', synthetic_parquet],
        capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    for spec in QUERIES.values():
        assert spec['title'] in result.stdout


def test_cli_exits_nonzero_for_named_query_with_missing_inputs(tmp_path):
    result = subprocess.run(
        [_sys.executable, SCRIPT, '--query', 'abc_classification',
         '--db', str(tmp_path / 'nope.duckdb'),
         '--parquet', str(tmp_path / 'nope.parquet')],
        capture_output=True, text=True)

    assert result.returncode == 1
    assert 'skipped' in result.stdout.lower()
    assert 'Traceback' not in result.stderr


def test_load_query_rejects_unknown_name():
    with pytest.raises(AnalysisError, match="No such query"):
        load_query('definitely_not_a_query')


def test_run_query_wraps_sql_errors(synthetic_db, synthetic_parquet, monkeypatch):
    """A broken query should surface as AnalysisError with context, not as a
    raw duckdb exception."""
    import analysis
    monkeypatch.setattr(analysis, 'load_query', lambda name: "SELECT * FROM does_not_exist")

    with pytest.raises(AnalysisError, match="failed"):
        run_query('when_model_beats_naive', db_path=synthetic_db,
                  parquet_path=synthetic_parquet)
