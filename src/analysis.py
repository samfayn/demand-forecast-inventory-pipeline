"""
Runs the analytical SQL in sql/ against DuckDB.

The queries live in sql/*.sql as plain files rather than as strings embedded
in Python, so they can be read, run in a DuckDB shell, or edited without
touching application code. This module is a thin loader: it reads a file,
substitutes any {placeholders}, executes it, and hands back a DataFrame.

Several queries read the cleaned Parquet directly via read_parquet(). DuckDB
scans it in place, so a 46M-row analysis needs no import step and no copy of
the data in the database.

Usage:
    # run everything
    python src/analysis.py

    # one query
    python src/analysis.py --query abc_classification

    # list what's available
    python src/analysis.py --list
"""
import argparse
import os
import sys

import duckdb
import pandas as pd

SQL_DIR = os.path.join(os.path.dirname(__file__), '..', 'sql')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'inventory.duckdb')
PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sales_clean.parquet')


class AnalysisError(Exception):
    """Raised when an analysis query can't be loaded or run."""


# Which queries need what. Queries touching the Parquet are useless without
# it; queries touching forecast_runs need a populated database.
QUERIES = {
    'abc_classification': {
        'needs_parquet': True,
        'needs_runs': False,
        'title': 'ABC classification by revenue contribution',
    },
    'weekly_seasonality_by_category': {
        'needs_parquet': True,
        'needs_runs': False,
        'title': 'Day-of-week demand pattern by category',
    },
    'when_model_beats_naive': {
        'needs_parquet': False,
        'needs_runs': True,
        'title': 'Forecast accuracy by demand volume quartile',
    },
    'store_forecast_quality': {
        'needs_parquet': False,
        'needs_runs': True,
        'title': 'Best and worst forecast items per store',
        'params': {'n_per_store': 2},
    },
    'accuracy_vs_demand_pattern': {
        'needs_parquet': True,
        'needs_runs': True,
        'title': 'Forecast accuracy vs demand intermittency',
    },
    'mase_outlier_diagnosis': {
        'needs_parquet': False,
        'needs_runs': True,
        'title': 'MASE outliers: genuine failures or degenerate scale factors?',
    },
}


def load_query(name):
    """Read a .sql file from sql/ and return its text."""
    path = os.path.join(SQL_DIR, f'{name}.sql')
    if not os.path.exists(path):
        raise AnalysisError(f"No such query: {name}. Run with --list to see options.")
    with open(path) as f:
        return f.read()


def run_query(name, db_path=None, parquet_path=None, **params):
    """
    Execute a named query and return the result as a DataFrame.

    Placeholders in the SQL ({parquet_path}, and any query-specific params)
    are substituted before execution. These are file paths and integers under
    this module's control, not user input, so formatting them in is safe here
    — anything reaching SQL from an untrusted source should still go through
    bind parameters, as it does in pipeline.query_runs().
    """
    if name not in QUERIES:
        raise AnalysisError(f"Unknown query: {name}. Run with --list to see options.")

    spec = QUERIES[name]
    db_path = db_path or DB_PATH
    parquet_path = parquet_path or PARQUET_PATH

    if spec['needs_parquet'] and not os.path.exists(parquet_path):
        raise AnalysisError(
            f"{name} reads the cleaned Parquet, but {parquet_path} doesn't exist. "
            f"Run: python src/data_prep.py")

    if spec['needs_runs'] and not os.path.exists(db_path):
        raise AnalysisError(
            f"{name} reads saved forecast runs, but {db_path} doesn't exist. "
            f"Run: python scripts/run_batch_forecasts.py --top-n 5")

    sql = load_query(name)
    substitutions = {'parquet_path': parquet_path}
    substitutions.update(spec.get('params', {}))
    substitutions.update(params)
    sql = sql.format(**substitutions)

    try:
        if spec['needs_runs']:
            with duckdb.connect(db_path, read_only=True) as con:
                return con.execute(sql).df()
        # Parquet-only queries don't need the runs database at all
        with duckdb.connect(':memory:') as con:
            return con.execute(sql).df()
    except duckdb.Error as e:
        raise AnalysisError(f"Query {name} failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--query', help="Run a single query by name")
    parser.add_argument('--list', action='store_true', help="List available queries")
    parser.add_argument('--db', default=None, help="Path to inventory.duckdb")
    parser.add_argument('--parquet', default=None, help="Path to sales_clean.parquet")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable queries:\n")
        for name, spec in QUERIES.items():
            sources = []
            if spec['needs_parquet']:
                sources.append('parquet')
            if spec['needs_runs']:
                sources.append('forecast_runs')
            print(f"  {name}")
            print(f"      {spec['title']}")
            print(f"      reads: {', '.join(sources)}\n")
        return

    names = [args.query] if args.query else list(QUERIES)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)

    failures = 0
    for name in names:
        try:
            df = run_query(name, db_path=args.db, parquet_path=args.parquet)
        except AnalysisError as e:
            print(f"\n[skipped] {name}: {e}\n")
            failures += 1
            continue

        print("\n" + "=" * 78)
        print(QUERIES[name]['title'])
        print(f"sql/{name}.sql")
        print("=" * 78)
        print(df.to_string(index=False))

    print()
    if failures and args.query:
        sys.exit(1)


if __name__ == '__main__':
    main()