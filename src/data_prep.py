"""
Batch data preparation pipeline: raw M5 CSVs -> analysis-ready Parquet.

Reshapes daily sales from wide format (one row per item/store, one column
per day) to long format (one row per item/store/day), joins in calendar
dates and sell prices, and filters out pre-launch periods (days before an
item was stocked at a given store) before persisting a single clean
Parquet file for downstream forecasting.

This is the production path — it supersedes notebooks/01_data_preparation.ipynb,
which now calls run_pipeline() from here for its production cells and keeps
only exploratory analysis (null checks, category breakdowns, a sales plot)
as notebook-native content.

Usage:
    python src/data_prep.py --data-dir data --output data/sales_clean.parquet

Expects three files in --data-dir (downloaded from the M5 Forecasting
competition on Kaggle — see README.md Setup):
    sales_train_validation.csv
    calendar.csv
    sell_prices.csv
"""
import argparse
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SALES_ID_VARS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


class DataPrepError(Exception):
    """Raised when the data preparation pipeline fails to load or process
    its inputs."""


def load_raw_data(data_dir):
    """
    Load the three raw M5 CSVs from data_dir.

    Raises DataPrepError with an actionable message if any expected file
    is missing, rather than letting a generic pandas FileNotFoundError
    propagate with a less clear path/context.
    """
    paths = {
        'sales': os.path.join(data_dir, 'sales_train_validation.csv'),
        'calendar': os.path.join(data_dir, 'calendar.csv'),
        'prices': os.path.join(data_dir, 'sell_prices.csv'),
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise DataPrepError(
            f"Missing required input file(s): {', '.join(missing)}. "
            f"Download them from the M5 Forecasting competition on Kaggle "
            f"into {data_dir}/ (see README.md Setup)."
        )

    logger.info("Loading raw CSVs from %s", data_dir)
    sales = pd.read_csv(paths['sales'])
    calendar = pd.read_csv(paths['calendar'])
    prices = pd.read_csv(paths['prices'])
    logger.info("Loaded: sales=%s calendar=%s prices=%s",
                sales.shape, calendar.shape, prices.shape)
    return sales, calendar, prices


def melt_sales_to_long(sales):
    """Reshape sales from one-row-per-item-store (with a 'd_N' column per
    day) into one-row-per-item-store-day, with the day identifier in a
    'd' column and the sales value in a 'sales' column."""
    sales_long = sales.melt(
        id_vars=SALES_ID_VARS,
        var_name='d',
        value_name='sales'
    )
    logger.info("Melted sales: %s wide rows -> %s long rows",
                sales.shape[0], sales_long.shape[0])
    return sales_long


def merge_calendar(sales_long, calendar):
    """Attach the real calendar date and the wm_yr_wk key (needed to join
    prices in the next step) onto each day-level sales row, via the 'd'
    day-identifier column shared with the calendar table."""
    merged = sales_long.merge(
        calendar[['d', 'date', 'wm_yr_wk']],
        on='d',
        how='left'
    )
    merged['date'] = pd.to_datetime(merged['date'])
    return merged


def merge_prices(sales_long, prices):
    """
    Attach sell_price via the composite key (store_id, item_id, wm_yr_wk).

    Rows with no matching price are periods before the item was stocked
    at that store — structurally absent from the catalog, not a real
    zero-demand observation. filter_unstocked_periods() removes these;
    this function only performs the join and logs how many such rows
    exist.
    """
    merged = sales_long.merge(
        prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    n_missing_price = merged['sell_price'].isna().sum()
    logger.info("Rows with no matching price (pre-launch periods): %s", n_missing_price)
    return merged


def filter_unstocked_periods(sales_long):
    """
    Drop rows with no price. These carry a sales value of 0, but the item
    did not exist at that store yet during that period. Treating them as
    real zero-demand observations would bias every downstream forecast
    and inflate safety stock requirements.
    """
    before = len(sales_long)
    clean = sales_long[sales_long['sell_price'].notna()].copy()
    logger.info("Filtered pre-launch rows: %s -> %s rows (dropped %s)",
                before, len(clean), before - len(clean))
    return clean


def run_pipeline(data_dir, output_path):
    """Run the full ingest -> reshape -> join -> filter -> persist pipeline
    and return the resulting clean DataFrame (also written to output_path
    as Parquet)."""
    sales, calendar, prices = load_raw_data(data_dir)

    sales_long = melt_sales_to_long(sales)
    sales_long = merge_calendar(sales_long, calendar)
    sales_long = merge_prices(sales_long, prices)
    sales_clean = filter_unstocked_periods(sales_long)

    logger.info(
        "Final dataset: %s products across %s stores, %s to %s",
        sales_clean['item_id'].nunique(), sales_clean['store_id'].nunique(),
        sales_clean['date'].min().date(), sales_clean['date'].max().date()
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sales_clean.to_parquet(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(sales_clean), output_path)

    return sales_clean


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data',
                         help="Directory containing the raw M5 CSVs (default: data)")
    parser.add_argument('--output', default='data/sales_clean.parquet',
                         help="Output Parquet path (default: data/sales_clean.parquet)")
    args = parser.parse_args()

    try:
        run_pipeline(args.data_dir, args.output)
    except DataPrepError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()