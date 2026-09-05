import pandas as pd

from pipeline import get_single_item, prepare_prophet_df, load_data


def make_sales_frame():
    return pd.DataFrame({
        'item_id':   ['FOODS_1_001', 'FOODS_1_001', 'FOODS_1_002', 'FOODS_1_001'],
        'store_id':  ['CA_1',        'CA_2',        'CA_1',        'CA_1'],
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

    loaded = load_data(str(parquet_path))

    pd.testing.assert_frame_equal(loaded.reset_index(drop=True),
                                   df.reset_index(drop=True))