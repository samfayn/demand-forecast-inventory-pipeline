"""
Batch forecast runner: sweeps many product/store combinations, computes an
inventory policy for each, and persists every run to DuckDB.

Runs the same code path as the Streamlit app (train -> backtest -> inventory
-> save), just headless and over many combinations instead of one at a time.
Useful for populating the Saved Results tab with enough runs to compare
across, and for pulling aggregate accuracy numbers off a real sweep rather
than a handful of hand-picked items.

Usage:
    # 5 highest-demand products in every store (50 runs)
    python scripts/run_batch_forecasts.py --top-n 5

    # random sample of 20 products per store, reproducible via seed
    python scripts/run_batch_forecasts.py --sample 20 --seed 42

    # restrict to specific stores
    python scripts/run_batch_forecasts.py --top-n 10 --stores CA_1 CA_3 TX_2

    # re-run combinations even if already saved
    python scripts/run_batch_forecasts.py --top-n 5 --no-skip-existing
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import (  # noqa: E402
    load_data, get_single_item, prepare_prophet_df, train_forecast,
    evaluate_forecast, calculate_inventory, save_results_to_db,
    load_all_runs, ForecastingError, DatabaseError,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MIN_ROWS_PER_SERIES = 30


def select_combinations(df, top_n=None, sample=None, stores=None, seed=42):
    """
    Build the list of (item_id, store_id) pairs to forecast.

    top_n picks the highest-average-demand products within each store;
    sample takes a random sample per store (reproducible via seed). Exactly
    one of the two must be given — they answer different questions, and
    silently combining them would make the resulting sweep hard to interpret.
    """
    if (top_n is None) == (sample is None):
        raise ValueError("Specify exactly one of --top-n or --sample.")

    if stores:
        df = df[df['store_id'].isin(stores)]
        if df.empty:
            raise ValueError(f"No data for stores: {stores}")

    per_store_demand = (
        df.groupby(['store_id', 'item_id'])['sales']
        .mean()
        .reset_index()
        .rename(columns={'sales': 'avg_daily_demand'})
    )

    combos = []
    for store_id, group in per_store_demand.groupby('store_id'):
        if top_n is not None:
            chosen = group.nlargest(top_n, 'avg_daily_demand')
        else:
            n = min(sample, len(group))
            chosen = group.sample(n=n, random_state=seed)
        combos.extend((row.item_id, store_id) for row in chosen.itertuples())

    return combos


def already_saved_combinations():
    """Return the set of (item_id, store_id) pairs already in the database,
    so a resumed sweep can skip them."""
    try:
        runs = load_all_runs(limit=100_000)
    except DatabaseError as e:
        logger.warning("Could not read existing runs (%s) — not skipping anything.", e)
        return set()

    if runs.empty:
        return set()
    return set(zip(runs['item_id'], runs['store_id']))


def run_one(sales_clean, item_id, store_id, args):
    """Forecast a single product/store combination and persist it.
    Returns a result dict, or None if the combination was skipped or failed."""
    item_df = get_single_item(sales_clean, item_id, store_id)
    if len(item_df) < MIN_ROWS_PER_SERIES:
        logger.warning("Skipping %s @ %s — only %s rows (need %s).",
                       item_id, store_id, len(item_df), MIN_ROWS_PER_SERIES)
        return None

    df_prophet = prepare_prophet_df(item_df)
    forecast = train_forecast(df_prophet, args.forecast_days)
    avg_price = item_df['sell_price'].mean()

    inv = calculate_inventory(
        forecast, df_prophet, avg_price,
        lead_time_days=args.lead_time,
        service_level=args.service_level,
        holding_cost=args.holding_cost,
        ordering_cost=args.ordering_cost,
    )
    eval_results = evaluate_forecast(df_prophet, holdout_days=args.holdout_days)

    run_id = save_results_to_db(
        item_id=item_id, store_id=store_id, inv=inv, eval_results=eval_results,
        forecast_days=args.forecast_days, lead_time=args.lead_time,
        ordering_cost=args.ordering_cost, holding_cost=args.holding_cost,
        service_level=args.service_level,
    )

    return {
        'run_id': run_id,
        'item_id': item_id,
        'store_id': store_id,
        'avg_daily_demand': inv['avg_daily_demand'],
        'std_daily_demand': inv['std_daily_demand'],
        'safety_stock': inv['safety_stock'],
        'mape': eval_results['mape'] if eval_results else None,
        'rmse': eval_results['rmse'] if eval_results else None,
    }


def print_summary(results, elapsed, n_failed, n_skipped):
    """Print aggregate stats over the sweep — the numbers worth quoting."""
    if not results:
        logger.warning("No successful runs to summarize.")
        return

    df = pd.DataFrame(results)
    with_mape = df[df['mape'].notna()]

    print("\n" + "=" * 68)
    print("BATCH SUMMARY")
    print("=" * 68)
    print(f"Successful runs:      {len(df)}")
    print(f"Failed:               {n_failed}")
    print(f"Skipped:              {n_skipped}")
    print(f"Elapsed:              {elapsed / 60:.1f} min "
          f"({elapsed / max(len(df), 1):.1f}s per run)")

    if not with_mape.empty:
        print(f"\nMAPE  — median {with_mape['mape'].median():.1f}%  "
              f"mean {with_mape['mape'].mean():.1f}%  "
              f"best {with_mape['mape'].min():.1f}%  "
              f"worst {with_mape['mape'].max():.1f}%")
        print(f"RMSE  — median {with_mape['rmse'].median():.2f} units")

        # Does forecast accuracy actually track demand volume? This is the
        # kind of aggregate question a single run can't answer.
        high = with_mape[with_mape['avg_daily_demand'] >= 1.5]
        low = with_mape[with_mape['avg_daily_demand'] < 1.5]
        if not high.empty and not low.empty:
            print("\nMAPE by demand volume:")
            print(f"  >= 1.5 units/day (n={len(high)}): median {high['mape'].median():.1f}%")
            print(f"  <  1.5 units/day (n={len(low)}):  median {low['mape'].median():.1f}%")

    # Variability vs volume: the ISE point the app's compare tab makes for a
    # single pair, checked here across the whole sweep.
    if len(df) >= 2 and df['avg_daily_demand'].std() > 0:
        corr_volume = df['safety_stock'].corr(df['avg_daily_demand'])
        corr_variability = df['safety_stock'].corr(df['std_daily_demand'])
        print("\nSafety stock correlation:")
        print(f"  with mean demand (volume):      {corr_volume:.3f}")
        print(f"  with demand std dev (variability): {corr_variability:.3f}")

    print("=" * 68)
    print("All runs saved to DuckDB — query them via the Saved Results tab,")
    print("or with query_runs() / load_all_runs() from src/pipeline.py.")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--parquet', default='data/sales_clean.parquet',
                        help="Cleaned parquet from data_prep.py (default: data/sales_clean.parquet)")
    parser.add_argument('--top-n', type=int,
                        help="Forecast the N highest-demand products in each store")
    parser.add_argument('--sample', type=int,
                        help="Forecast a random sample of N products per store")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for --sample (default: 42)")
    parser.add_argument('--stores', nargs='+',
                        help="Limit to specific store IDs (e.g. CA_1 TX_2)")
    parser.add_argument('--forecast-days', type=int, default=90)
    parser.add_argument('--holdout-days', type=int, default=90)
    parser.add_argument('--lead-time', type=int, default=7)
    parser.add_argument('--ordering-cost', type=float, default=10.0)
    parser.add_argument('--holding-cost', type=float, default=0.20)
    parser.add_argument('--service-level', type=float, default=0.95)
    parser.add_argument('--no-skip-existing', action='store_true',
                        help="Re-run combinations already saved in the database")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        logger.error("Parquet not found: %s. Run src/data_prep.py first.", args.parquet)
        sys.exit(1)

    logger.info("Loading %s ...", args.parquet)
    sales_clean = load_data(args.parquet)
    logger.info("Loaded %s rows.", f"{len(sales_clean):,}")

    try:
        combos = select_combinations(sales_clean, top_n=args.top_n, sample=args.sample,
                                     stores=args.stores, seed=args.seed)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    n_skipped = 0
    if not args.no_skip_existing:
        existing = already_saved_combinations()
        before = len(combos)
        combos = [c for c in combos if c not in existing]
        n_skipped = before - len(combos)
        if n_skipped:
            logger.info("Skipping %s combination(s) already in the database.", n_skipped)

    if not combos:
        logger.info("Nothing to do — every selected combination is already saved.")
        return

    logger.info("Forecasting %s combination(s). This is sequential and can take a "
                "while; it is safe to interrupt and resume.", len(combos))

    results, n_failed = [], 0
    start = time.time()

    for i, (item_id, store_id) in enumerate(combos, start=1):
        logger.info("[%s/%s] %s @ %s", i, len(combos), item_id, store_id)
        try:
            result = run_one(sales_clean, item_id, store_id, args)
            if result is not None:
                results.append(result)
        except ForecastingError as e:
            logger.warning("  forecast failed, continuing: %s", e)
            n_failed += 1
        except DatabaseError as e:
            logger.error("  database write failed: %s", e)
            n_failed += 1
        except KeyboardInterrupt:
            logger.info("Interrupted — %s run(s) already saved. Re-run to resume.",
                        len(results))
            break

    print_summary(results, time.time() - start, n_failed, n_skipped)


if __name__ == '__main__':
    main()
