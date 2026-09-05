import pandas as pd
import pytest

from pipeline import get_single_item, prepare_prophet_df, load_data


def make_sales_frame():
    """Mirrors the column set produced by src/data_prep.py, so tests exercise
    the same shape the app actually loads."""
    return pd.DataFrame({
        'item_id':   ['FOODS_1_001', 'FOODS_1_001', 'FOODS_1_002', 'FOODS_1_001'],
        'store_id':  ['CA_1',        'CA_2',        'CA_1',        'CA_1'],
        'state_id':  ['CA',          'CA',          'CA',          'CA'],
        'cat_id':    ['FOODS',       'FOODS',       'FOODS',       'FOODS'],
        'dept_id':   ['FOODS_1',     'FOODS_1',     'FOODS_1',     'FOODS_1'],
        'date':      pd.to_datetime(['2026-01-01', '2026-01-01', '2026-01-01', '2026-01-02']),
        'sales':     [5, 8, 3, 6],
        'sell_price': [2.5, 2.5, 4.0, 2.5],
    })


def test_get_single_item_filters_on_both_item_and_store():
    df = make_sales_frame()
    result = get_single_item(df, product_id='FOODS_1_001', store_id='CA_1')

    # only the two CA_1 rows for FOODS_1_001 should remain
    assert len(result) == 2
    assert (result['item_id'] == 'FOODS_1_001').all()
    assert (result['store_id'] == 'CA_1').all()


def test_get_single_item_returns_empty_for_nonexistent_combination():
    df = make_sales_frame()
    result = get_single_item(df, product_id='FOODS_1_999', store_id='CA_1')
    assert len(result) == 0


def test_get_single_item_does_not_mutate_original_frame():
    """get_single_item returns .copy() — verify a caller mutating the result
    can't corrupt the original DataFrame."""
    df = make_sales_frame()
    result = get_single_item(df, product_id='FOODS_1_001', store_id='CA_1')
    result['sales'] = 0
    assert (df.loc[(df['item_id'] == 'FOODS_1_001') & (df['store_id'] == 'CA_1'), 'sales'] != 0).all()


def test_prepare_prophet_df_renames_columns_correctly():
    df = make_sales_frame()
    item_df = get_single_item(df, product_id='FOODS_1_001', store_id='CA_1')
    prophet_df = prepare_prophet_df(item_df)

    assert list(prophet_df.columns) == ['ds', 'y']
    assert prophet_df['y'].tolist() == [5, 6]


def test_prepare_prophet_df_resets_index():
    df = make_sales_frame()
    item_df = get_single_item(df, product_id='FOODS_1_001', store_id='CA_1')
    prophet_df = prepare_prophet_df(item_df)

    assert list(prophet_df.index) == list(range(len(prophet_df)))


def test_load_data_reads_parquet_file(tmp_path):
    df = make_sales_frame()
    parquet_path = tmp_path / "sales_clean.parquet"
    df.to_parquet(parquet_path)

    # optimize_memory=False keeps the original dtypes, so this stays a pure
    # round-trip check rather than also asserting the categorical conversion
    loaded = load_data(str(parquet_path), optimize_memory=False)

    pd.testing.assert_frame_equal(loaded.reset_index(drop=True),
                                   df.reset_index(drop=True))

# ---------------------------------------------------------------------------
# Indexed lookup path used by the batch runner
# ---------------------------------------------------------------------------

from pipeline import build_series_index, get_indexed_item


def test_indexed_lookup_matches_scan_lookup():
    """build_series_index + get_indexed_item is an optimization, so it must
    return exactly what the straightforward scan returns."""
    df = make_sales_frame()
    indexed = build_series_index(df)

    cols = ['item_id', 'store_id', 'date', 'sales', 'sell_price']
    scanned = get_single_item(df, 'FOODS_1_001', 'CA_1')[cols].sort_values('date').reset_index(drop=True)
    looked_up = get_indexed_item(indexed, 'FOODS_1_001', 'CA_1')[cols].sort_values('date').reset_index(drop=True)

    pd.testing.assert_frame_equal(scanned, looked_up)


def test_indexed_lookup_returns_empty_for_missing_combination():
    """Must match get_single_item's behavior of returning an empty frame
    rather than raising, so callers can use one 'not enough data' branch."""
    indexed = build_series_index(make_sales_frame())
    result = get_indexed_item(indexed, 'NO_SUCH_ITEM', 'CA_1')

    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_indexed_lookup_does_not_leak_other_stores():
    indexed = build_series_index(make_sales_frame())
    result = get_indexed_item(indexed, 'FOODS_1_001', 'CA_1')

    assert (result['store_id'] == 'CA_1').all()
    assert (result['item_id'] == 'FOODS_1_001').all()

# ---------------------------------------------------------------------------
# load_data memory behavior — the deployed app has a hard memory ceiling
# ---------------------------------------------------------------------------

def test_load_data_reads_only_analysis_columns_by_default(tmp_path):
    """The pipeline's output carries columns nothing downstream reads (id, d,
    wm_yr_wk). Loading them wastes memory that the deployed app doesn't have."""
    from pipeline import load_data, ANALYSIS_COLUMNS

    df = make_sales_frame()
    df['id'] = 'ITEM_CA_1_validation'
    df['d'] = 'd_1'
    df['wm_yr_wk'] = 11101

    path = tmp_path / 'sales.parquet'
    df.to_parquet(path, index=False)

    loaded = load_data(str(path))
    assert set(loaded.columns) == set(ANALYSIS_COLUMNS)
    for unused in ('id', 'd', 'wm_yr_wk'):
        assert unused not in loaded.columns


def test_load_data_uses_categorical_dtypes_for_repeated_columns(tmp_path):
    from pipeline import load_data, CATEGORICAL_COLUMNS

    path = tmp_path / 'sales.parquet'
    make_sales_frame().to_parquet(path, index=False)

    loaded = load_data(str(path))
    for col in CATEGORICAL_COLUMNS:
        assert str(loaded[col].dtype) == 'category', f"{col} should be categorical"


def test_load_data_optimization_preserves_values(tmp_path):
    """Memory optimization must not change the data itself."""
    from pipeline import load_data

    df = make_sales_frame()
    path = tmp_path / 'sales.parquet'
    df.to_parquet(path, index=False)

    optimized = load_data(str(path))
    raw = load_data(str(path), optimize_memory=False)

    for col in ['item_id', 'store_id']:
        assert list(optimized[col].astype(str)) == list(raw[col].astype(str))
    assert list(optimized['sales']) == list(raw['sales'])


def test_load_data_can_read_all_columns_when_asked(tmp_path):
    from pipeline import load_data

    df = make_sales_frame()
    df['wm_yr_wk'] = 11101
    path = tmp_path / 'sales.parquet'
    df.to_parquet(path, index=False)

    loaded = load_data(str(path), columns=None)
    assert 'wm_yr_wk' in loaded.columns


def test_load_data_raises_clear_error_for_unrecognized_parquet(tmp_path):
    """A Parquet with none of the expected columns should say so plainly
    rather than surfacing a pyarrow FieldRef error."""
    from pipeline import load_data

    path = tmp_path / 'wrong.parquet'
    pd.DataFrame({'foo': [1, 2], 'bar': ['a', 'b']}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="none of the expected columns"):
        load_data(str(path))