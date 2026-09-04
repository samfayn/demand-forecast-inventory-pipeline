"""
Migration 001: add service_level to forecast_runs.

Context: calculate_inventory() previously hardcoded z_score=1.645 (95% service
level) regardless of what was passed in. That bug is fixed in pipeline.py, but
any forecast_runs rows saved before the fix were computed at an implicit 95%
service level with no column recording that fact. This migration adds the
column and backfills existing rows with 0.95, which is the true value they
were computed at — not a guess.

Safe to run multiple times: checks for the column before altering.

Usage:
    python scripts/migrations/001_add_service_level.py
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
              f"It will be created fresh with the correct schema on next app run.")
        return

    con = duckdb.connect(DB_PATH)

    if column_exists(con, 'forecast_runs', 'service_level'):
        print("service_level column already present — nothing to do.")
        con.close()
        return

    before_count = con.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]

    con.execute("ALTER TABLE forecast_runs ADD COLUMN service_level DOUBLE DEFAULT 0.95")
    con.execute("UPDATE forecast_runs SET service_level = 0.95 WHERE service_level IS NULL")

    con.close()
    print(f"Migration complete: added service_level to forecast_runs, "
          f"backfilled {before_count} existing row(s) with 0.95.")


if __name__ == "__main__":
    main()
