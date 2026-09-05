import os

import pandas as pd
import pytest

from data_prep import (load_raw_data, melt_sales_to_long, merge_calendar,
                        merge_prices, filter_unstocked_periods, run_pipeline,
                        DataPrepError)


def make_sales_wide():
    """Two items at one store, three days each — wide format like the
    real sales_train_validation.csv (one column per day)."""
    return pd.DataFrame({
        'id':        ['ITEM_A_CA_1_validation', 'ITEM_B_CA_1_validation'],
        'item_id':   ['ITEM_A', 'ITEM_B'],
        'dept_id':   ['FOODS_1', 'FOODS_1'],
        'cat_id':    ['FOODS', 'FOODS'],
        'store_id':  ['CA_1', 'CA_1'],
        'state_id':  ['CA', 'CA'],
        'd_1': [0, 4],   # ITEM_A hasn't launched yet on day 1 (sells 0, no price will exist)
        'd_2': [5, 6],
        'd_3': [3, 2],
    })


def make_calendar():
    return pd.DataFrame({
        'd':         ['d_1', 'd_2', 'd_3'],
        'date':      ['2026-01-01', '2026-01-02', '2026-01-03'],
        'wm_yr_wk':  [11101, 11102, 11102],
    })


def make_prices():
    """ITEM_A has no price row for wm_yr_wk=11101 — simulates it not being
    stocked yet in that week. ITEM_B has a price for every week."""
    return pd.DataFrame({
        'store_id':   ['CA_1', 'CA_1', 'CA_1'],
        'item_id':    ['ITEM_A', 'ITEM_B', 'ITEM_B'],
        'wm_yr_wk':   [11102,    11101,    11102],
        'sell_price': [2.50,     1.00,     1.00],
    })


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

def test_load_raw_data_reads_all_three_files(tmp_path):
    make_sales_wide().to_csv(tmp_path / 'sales_train_validation.csv', index=False)
    make_calendar().to_csv(tmp_path / 'calendar.csv', index=False)
    make_prices().to_csv(tmp_path / 'sell_prices.csv', index=False)

    sales, calendar, prices = load_raw_data(str(tmp_path))

    assert len(sales) == 2
    assert len(calendar) == 3
    assert len(prices) == 3


def test_load_raw_data_raises_clear_error_when_files_missing(tmp_path):
    # only create one of the three required files
    make_sales_wide().to_csv(tmp_path / 'sales_train_validation.csv', index=False)

    with pytest.raises(DataPrepError, match="calendar.*prices|prices.*calendar"):
        load_raw_data(str(tmp_path))


# ---------------------------------------------------------------------------
# melt_sales_to_long
# ---------------------------------------------------------------------------

def test_melt_produces_one_row_per_item_per_day():
    sales_long = melt_sales_to_long(make_sales_wide())

    # 2 items x 3 days = 6 rows
    assert len(sales_long) == 6
    assert set(sales_long.columns) >= {'item_id', 'store_id', 'd', 'sales'}

    item_a_rows = sales_long[sales_long['item_id'] == 'ITEM_A'].sort_values('d')
    assert item_a_rows['sales'].tolist() == [0, 5, 3]


# ---------------------------------------------------------------------------
# merge_calendar
# ---------------------------------------------------------------------------

def test_merge_calendar_attaches_dates_and_week_key():
    sales_long = melt_sales_to_long(make_sales_wide())
    merged = merge_calendar(sales_long, make_calendar())

    assert 'date' in merged.columns
    assert 'wm_yr_wk' in merged.columns
    assert pd.api.types.is_datetime64_any_dtype(merged['date'])

    day1 = merged[merged['d'] == 'd_1']
    assert (day1['wm_yr_wk'] == 11101).all()


# ---------------------------------------------------------------------------
# merge_prices + filter_unstocked_periods — the correctness-critical pair
# ---------------------------------------------------------------------------

def test_merge_prices_leaves_null_for_prelaunch_rows():
    sales_long = melt_sales_to_long(make_sales_wide())
    sales_long = merge_calendar(sales_long, make_calendar())
    merged = merge_prices(sales_long, make_prices())

    # ITEM_A on day 1 (wm_yr_wk=11101) has no matching price row
    item_a_day1 = merged[(merged['item_id'] == 'ITEM_A') & (merged['d'] == 'd_1')]
    assert item_a_day1['sell_price'].isna().all()

    # ITEM_B has a price every day
    item_b_rows = merged[merged['item_id'] == 'ITEM_B']
    assert item_b_rows['sell_price'].notna().all()


def test_filter_drops_only_prelaunch_rows_not_real_zero_sales():
    """This is the correctness property the README calls out explicitly:
    a pre-launch row (no price) must be dropped, but a genuine zero-sales
    day for an already-launched item must NOT be dropped."""
    sales_long = melt_sales_to_long(make_sales_wide())
    sales_long = merge_calendar(sales_long, make_calendar())
    sales_long = merge_prices(sales_long, make_prices())

    clean = filter_unstocked_periods(sales_long)

    # ITEM_A's day 1 (pre-launch, sales=0, no price) should be gone
    assert not ((clean['item_id'] == 'ITEM_A') & (clean['d'] == 'd_1')).any()

    # ITEM_A's day 2 (sales=5, has a price) should remain
    item_a_day2 = clean[(clean['item_id'] == 'ITEM_A') & (clean['d'] == 'd_2')]
    assert len(item_a_day2) == 1
    assert item_a_day2['sales'].iloc[0] == 5

    # every remaining row must have a non-null price
    assert clean['sell_price'].notna().all()


def test_filter_removes_exactly_the_expected_row_count():
    sales_long = melt_sales_to_long(make_sales_wide())
    sales_long = merge_calendar(sales_long, make_calendar())
    sales_long = merge_prices(sales_long, make_prices())

    clean = filter_unstocked_periods(sales_long)

    # 6 total rows, 1 pre-launch row (ITEM_A/d_1) dropped -> 5 remain
    assert len(clean) == 5


# ---------------------------------------------------------------------------
# run_pipeline — full end-to-end integration test
# ---------------------------------------------------------------------------

def test_run_pipeline_end_to_end(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    make_sales_wide().to_csv(data_dir / 'sales_train_validation.csv', index=False)
    make_calendar().to_csv(data_dir / 'calendar.csv', index=False)
    make_prices().to_csv(data_dir / 'sell_prices.csv', index=False)

    output_path = tmp_path / 'output' / 'sales_clean.parquet'
    result = run_pipeline(str(data_dir), str(output_path))

    assert output_path.exists()
    assert len(result) == 5
    assert result['sell_price'].notna().all()

    # confirm what was written to disk matches what was returned
    reloaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True), result.reset_index(drop=True)
    )


def test_run_pipeline_raises_data_prep_error_for_missing_input(tmp_path):
    empty_data_dir = tmp_path / 'empty'
    empty_data_dir.mkdir()

    with pytest.raises(DataPrepError):
        run_pipeline(str(empty_data_dir), str(tmp_path / 'out.parquet'))


# ---------------------------------------------------------------------------
# CLI entrypoint — invoked as a real subprocess, since that's the documented
# way this script is actually run (python src/data_prep.py --data-dir ...)
# ---------------------------------------------------------------------------

import subprocess
import sys as _sys

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'data_prep.py')


def test_cli_runs_successfully_and_writes_output(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    make_sales_wide().to_csv(data_dir / 'sales_train_validation.csv', index=False)
    make_calendar().to_csv(data_dir / 'calendar.csv', index=False)
    make_prices().to_csv(data_dir / 'sell_prices.csv', index=False)

    output_path = tmp_path / 'sales_clean.parquet'
    result = subprocess.run(
        [_sys.executable, SCRIPT_PATH,
         '--data-dir', str(data_dir), '--output', str(output_path)],
        capture_output=True, text=True
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert len(written) == 5


def test_cli_exits_nonzero_with_clear_message_on_missing_input(tmp_path):
    empty_data_dir = tmp_path / 'empty'
    empty_data_dir.mkdir()

    result = subprocess.run(
        [_sys.executable, SCRIPT_PATH,
         '--data-dir', str(empty_data_dir), '--output', str(tmp_path / 'out.parquet')],
        capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "Missing required input file" in result.stderr
    # should NOT dump a raw Python traceback for this expected failure mode
    assert "Traceback" not in result.stderr