"""
Migration 002: add mase to forecast_runs.

Context: MASE (Mean Absolute Scaled Error) was added as a third accuracy
metric alongside MAPE and RMSE. Unlike MAPE it is defined when actuals are
zero, and it reads directly as a comparison against a seasonal naive
baseline, which makes it the appropriate headline metric for intermittent
retail demand.

Existing rows are backfilled with NULL rather than a computed value. MASE
depends on the training series that produced each forecast, and that series
is not stored in the database, so it cannot be reconstructed after the fact.
Re-run scripts/run_batch_forecasts.py --no-skip-existing to populate MASE for
combinations you care about.

Safe to run multiple times: checks for the column before altering.

Usage:
    python scripts/migrations/002_add_mase.py
"""
import os
import duckdb

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'inventory.duckdb')


def column_exists(con, table, column):
    result = con.execute(f"""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name = '{table}' AND column_name = '{column}'
    """).fetchone()
    return result[0] > 0


def main():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH} — nothing to migrate. "
              f"It will be created with the correct schema on next run.")
        return

    con = duckdb.connect(DB_PATH)

    if column_exists(con, 'forecast_runs', 'mase'):
        print("mase column already present — nothing to do.")
        con.close()
        return

    before_count = con.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]

    con.execute("ALTER TABLE forecast_runs ADD COLUMN mase DOUBLE")

    con.close()
    print(f"Migration complete: added mase to forecast_runs. "
          f"{before_count} existing row(s) left with NULL mase — re-run the batch "
          f"script with --no-skip-existing to populate them.")


if __name__ == "__main__":
    main()