"""
Build a deployable demo dataset.

Streamlit Community Cloud gives an app roughly 1 GB of memory, so the full
46M-row sales_clean.parquet cannot be loaded there. This extracts just the
item/store combinations that already have saved forecast runs, writes them to
a much smaller Parquet, and copies the DuckDB alongside it.

Pairing the two matters: a visitor who lands on the deployed app and never
clicks "Run Forecast" still sees a populated Saved Results tab with real runs
in it, rather than an empty screen that looks broken.

The demo data is committed to the repository (see .gitignore), unlike the
full dataset which stays local.

Usage:
    python scripts/build_demo_data.py --all-stores --max-pairs 150
    python scripts/build_demo_data.py --max-pairs 500
"""
import argparse
import logging
import os
import shutil
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
FULL_PARQUET = os.path.join(REPO_ROOT, 'data', 'sales_clean.parquet')
FULL_DB = os.path.join(REPO_ROOT, 'data', 'inventory.duckdb')
DEMO_DIR = os.path.join(REPO_ROOT, 'demo_data')
DEMO_PARQUET = os.path.join(DEMO_DIR, 'sales_demo.parquet')
DEMO_DB = os.path.join(DEMO_DIR, 'inventory_demo.duckdb')


def forecast_pairs(db_path, max_pairs=None):
    """The (item_id, store_id) pairs that have saved runs, most-forecast
    first so that trimming keeps the best-covered combinations."""
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute("""
            SELECT item_id, store_id, COUNT(*) AS runs
            FROM forecast_runs
            GROUP BY item_id, store_id
            ORDER BY runs DESC, item_id
        """).df()

    if max_pairs is not None:
        df = df.head(max_pairs)
    return df


def build(full_parquet, full_db, max_pairs=None, all_stores=False):
    for path, what in [(full_parquet, 'cleaned Parquet'), (full_db, 'forecast database')]:
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing {what}: {path}\n"
                f"Run src/data_prep.py and scripts/run_batch_forecasts.py first.")

    pairs = forecast_pairs(full_db, max_pairs)
    logger.info("Found %s item/store pair(s) with saved runs.", len(pairs))
    if pairs.empty:
        raise SystemExit("No saved forecast runs — nothing to build a demo from.")

    logger.info("Reading %s ...", full_parquet)
    df = pd.read_parquet(full_parquet)
    logger.info("Full dataset: %s rows", f"{len(df):,}")

    if all_stores:
        # Keep every store that stocks each selected item, not just the
        # store it happened to be forecast at. Without this, most items in
        # the demo exist at a single store, so the comparison tab (which
        # needs two products sharing one store) almost always has nothing
        # to offer.
        wanted_items = set(pairs['item_id'])
        demo = df[df['item_id'].isin(wanted_items)].copy()
        logger.info("Keeping all stores for %s item(s).", len(wanted_items))
    else:
        wanted = set(zip(pairs['item_id'], pairs['store_id']))
        mask = pd.Series(list(zip(df['item_id'], df['store_id']))).isin(wanted)
        demo = df[mask.values].copy()

    if demo.empty:
        raise SystemExit(
            "No rows matched the saved runs. Are the Parquet and database "
            "from the same pipeline run?")

    os.makedirs(DEMO_DIR, exist_ok=True)
    demo.to_parquet(DEMO_PARQUET, index=False)
    shutil.copy2(full_db, DEMO_DB)

    parquet_mb = os.path.getsize(DEMO_PARQUET) / 1e6
    db_mb = os.path.getsize(DEMO_DB) / 1e6

    logger.info("Demo dataset: %s rows (%.1f%% of full), %s items, %s stores",
                f"{len(demo):,}", 100 * len(demo) / len(df),
                demo['item_id'].nunique(), demo['store_id'].nunique())
    logger.info("Wrote %s (%.1f MB)", DEMO_PARQUET, parquet_mb)
    logger.info("Wrote %s (%.1f MB)", DEMO_DB, db_mb)

    total_mb = parquet_mb + db_mb
    if total_mb > 90:
        logger.warning(
            "Demo data is %.0f MB. GitHub rejects files over 100 MB and warns "
            "above 50 MB. Re-run with a smaller --max-pairs.", total_mb)
    elif total_mb > 45:
        logger.warning(
            "Demo data is %.0f MB, above GitHub's 50 MB soft warning. It will "
            "push, but consider --max-pairs to trim it.", total_mb)

    return demo


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--parquet', default=FULL_PARQUET)
    parser.add_argument('--db', default=FULL_DB)
    parser.add_argument('--max-pairs', type=int, default=None,
                        help="Cap the number of item/store pairs to keep")
    parser.add_argument('--all-stores', action='store_true',
                        help="Keep every store that stocks each selected item, not "
                             "just the store it was forecast at. Larger output, but "
                             "makes the comparison tab usable — it needs two products "
                             "sharing one store.")
    args = parser.parse_args()

    build(args.parquet, args.db, args.max_pairs, args.all_stores)


if __name__ == '__main__':
    main()