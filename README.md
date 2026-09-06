# Demand Forecasting & Inventory Optimization

[![Tests](https://github.com/samfayn/demand-forecast-inventory-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/samfayn/demand-forecast-inventory-pipeline/actions/workflows/tests.yml)
[![Live app](https://img.shields.io/badge/demo-live-brightgreen)](https://demand-forecaster.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Try the live app →](https://demand-forecaster.streamlit.app/)**

A batch pipeline and forecasting app that turns 46 million rows of raw Walmart sales into
per-item inventory policies. Every forecast is backtested against held-out data and scored
against a seasonal naive baseline, and every run is written to a queryable database.

The inventory side applies standard Industrial & Systems Engineering theory (Economic Order
Quantity, safety stock, reorder point) on top of a Prophet demand forecast, using the M5
Forecasting dataset.

## What the data says

The interesting results here came from measuring the model against a baseline rather than
reporting its error in isolation. Three things came out of 825 backtested forecasts, and the
first two contradicted what I expected before running them.

### Prophet barely beats a seasonal naive forecast

Median MASE across all backtested runs is **0.964**, and 460 of 825 (56%) score below 1.0.
MASE divides the model's mean absolute error by the error of a seasonal naive forecast
(repeat the same weekday from last week) computed on the training window, so 1.0 means the
model matched that baseline and below 1.0 means it beat it.

A median of 0.964 is a 4% improvement over doing nothing clever. Three independent sampling
designs (200 random items, 30 highest-demand items, 596 stratified across demand deciles)
all landed between 0.93 and 0.97 with 53–60% beating the baseline. The result is consistent,
and it is not a strong endorsement of the model.

This is the number most worth knowing about the project, and it only exists because there is
a baseline to compare against. Reported alone, the median MAPE of 69% would have said
nothing about whether Prophet was earning its complexity.

### Demand volume does not predict forecast quality. Intermittency does.

The obvious hypothesis is that high-volume items forecast better. Split by demand quartile,
they do not:

| Volume quartile | Demand (units/day) | Runs | Median MASE | Beat naive |
|---|---|---|---|---|
| 1 | 0.00 – 0.22 | 206 | 0.905 | 60% |
| 2 | 0.22 – 0.59 | 206 | 1.002 | 50% |
| 3 | 0.59 – 1.55 | 205 | 0.947 | 58% |
| 4 | 1.55 – 91.13 | 205 | 0.974 | 55% |

No trend, and the lowest quartile scores best. Split instead by the share of days with zero
sales, a pattern appears immediately:

| Demand pattern | Runs | Zero-day % | Median MASE | Beat naive |
|---|---|---|---|---|
| Continuous (<10% zero days) | 31 | 3.2 | 0.787 | 68% |
| Occasional gaps (10–30%) | 100 | 21.6 | 0.919 | 60% |
| Intermittent (30–60%) | 244 | 45.7 | 0.878 | 68% |
| Highly intermittent (60%+) | 449 | 79.0 | 1.026 | 47% |

Prophet beats the baseline everywhere except the highly intermittent tail, where it loses.
Average demand and zero-day share are two ways of measuring how small an item is, and only
one of them predicts whether a smooth trend-and-seasonality model will work. The operational
reading is that Prophet is worth running down to roughly 60% zero-days, and past that a
simpler baseline is as good and much cheaper.

### MAPE is actively misleading on this data

Median MAPE is 69% against a median RMSE of 0.98 units. Both describe the same forecasts.
The typical forecast misses by less than one unit per day, which for an item selling two or
three units is reasonable; as a percentage of a very small number, that same miss reads as
69%. Being wrong by one unit on a day with one sale is a 100% error no matter how small the
operational consequence. The M5 competition avoided MAPE for this reason and scored on
WRMSSE instead.

The dashboard leads with MASE, shows RMSE beside it, and reports MAPE last with a note that
it is unreliable at low volume.

### The catalog is more concentrated than the classic 80/20

ABC classification over all 30,490 item/store pairs:

| Class | Pairs | % of catalog | % of revenue | Avg units/day |
|---|---|---|---|---|
| A | 11,170 | 36.6% | 80.0% | 2.68 |
| B | 9,406 | 30.8% | 15.0% | 0.80 |
| C | 9,914 | 32.5% | 5.0% | 0.36 |

It takes 36.6% of the catalog to reach 80% of revenue, so the concentration is real but
looser than the textbook figure. The bottom third contributes 5%.

Safety stock correlates with demand standard deviation at 1.000 and with mean demand at
0.718. The 1.000 is arithmetic rather than a finding: safety stock is `z × σ × √L`, and with
service level and lead time held fixed across the sweep it reduces to a constant times σ, so
a perfect correlation is what a correct implementation must produce. The 0.718 is the useful
number. Higher-volume items do tend to be more variable, but loosely enough that ranking a
catalog by units sold would misallocate safety stock against the items whose variability
diverges from their volume.

### Where the extreme values come from

One run scored a MASE of 368.5. It is not a near-dead series: `HOUSEHOLD_1_032` at CA_3
averages 1.85 units/day with an RMSE of 3.25. Working backwards, the denominator must have
been near zero, which means the training window was essentially flat and the item became
active during the holdout period.

That is a limitation of MASE rather than a model failure. The scale factor comes from the
training window while the error comes from the test window, so an item that changes regime
between the two produces a meaningless ratio. `sql/mase_outlier_diagnosis.sql` exists to
separate these cases from genuine failures, and it is why this project reports the median
rather than the mean (0.964 against 1.47).

## Live demo

**[demand-forecaster.streamlit.app](https://demand-forecaster.streamlit.app/)**

The deployment runs on a subset: 576 products across all 10 stores, 8.7M rows, with 1,066
saved forecast runs preloaded so the Saved Results tab has real content on arrival. Streamlit
Community Cloud caps an app near 1 GB of memory, which the full 46M-row dataset cannot fit.
The pipeline, forecasting, and inventory logic are identical; only the catalog is smaller.

Saving a new run fails on the deployed app, by design. Streamlit's filesystem is ephemeral
and the demo database ships read-only, so a forecast computes and displays normally but
reports that it was not persisted rather than crashing.

## Architecture

```
raw M5 CSVs → src/data_prep.py → sales_clean.parquet ─┬→ app.py (Streamlit)      ─┐
                                                       ├→ scripts/run_batch_...   ─┼→ DuckDB
                                                       └→ src/analysis.py ─────────┘
                                                              (reads both)
```

The preparation stage is a batch job that runs once. Everything downstream reads its Parquet
output. The dashboard and the batch runner call the same functions in `src/pipeline.py`, so
an interactive forecast and a swept one cannot disagree.

## Data preparation

`src/data_prep.py` is an importable module with a CLI entry point, covered by 11 tests.

| Step | Detail |
|---|---|
| Ingest | 3 raw CSVs: sales (30,490 × 1,913 wide), calendar (1,969 rows), prices (6.8M rows) |
| Reshape | Melts wide to long → **58.3M rows** |
| Join | Calendar on `d`, then prices on `(store_id, item_id, wm_yr_wk)` |
| Filter | Drops 12.3M rows with no price |
| Persist | Writes **46.0M rows** to Parquet |

Final dataset: 3,049 products across 10 stores, 2011-01-29 to 2016-04-24.

The filter is the step that matters for correctness. Rows with no matching price are periods
before an item was stocked at that store. They carry a sales value of 0, but the item did not
exist there yet, so they are structurally absent rather than real zero-demand observations.
Keeping them would pull every downstream mean toward zero and inflate safety stock across the
catalog. `test_filter_drops_only_prelaunch_rows_not_real_zero_sales` pins the distinction
down: pre-launch rows go, real zero-sales days for stocked items stay.

A side effect worth knowing: about a fifth of product/store combinations do not exist at all,
because those items were never stocked at those stores. The dashboard's store dropdown is
filtered by the selected product for this reason.

## Forecasting and accuracy

Prophet models are fit per product/store selection rather than pre-trained across the
catalog, and memoized so reselecting a combination reuses the fit. Yearly and weekly
seasonality on, daily off, `changepoint_prior_scale=0.05`, 90-day horizon.

Weekly seasonality is switched on because the data supports it. Every category peaks on
Saturday and Sunday and troughs midweek, with a peak-to-trough spread of 32 to 41 index
points (`sql/weekly_seasonality_by_category.sql`).

Every forecast is backtested before its inventory policy is shown. The model trains on all
data except the final 90 days, predicts that window, and is scored on MASE, RMSE, and MAPE.
MAPE is computed only over days with nonzero actual sales and comes back empty when there
are none; RMSE and MASE are always reported, so a well-defined metric is never discarded
because another one is undefined. Negative predictions are clipped to zero first.

## Inventory model

| Metric | Formula |
|---|---|
| Safety stock | Z × σ(demand) × √(lead time) |
| Reorder point | (Avg daily demand × lead time) + safety stock |
| EOQ | √(2 × annual demand × ordering cost / (holding cost × price)) |

Service level, lead time, ordering cost, and holding cost are adjustable in the dashboard. Z
is derived from the selected service level through the inverse normal CDF.

## Persistence

Runs are written to DuckDB using a header-detail schema rather than one wide table:

| Table | Grain | Contents |
|---|---|---|
| `forecast_runs` | One row per run | Timestamp, item, store, parameters, inventory metrics, MASE, RMSE, MAPE |
| `forecast_daily` | One row per forecast day | `yhat` and its bounds, linked by `run_id` |

Splitting by grain keeps run-level metrics from repeating across 90 forecast days and lets
the app load a run summary without touching its daily rows.

Summary statistics in the app are aggregated in SQL across the whole table rather than
computed from the page of rows loaded for display. An earlier version computed them from a
200-row page and reported "1,065 runs" as "200".

Schema changes go through numbered migration scripts in `scripts/migrations/` rather than
recreating the database. Each is idempotent and backfills existing rows.

## Analytical SQL

`sql/` holds six queries as plain files, run through `src/analysis.py`. They are files rather
than strings embedded in Python so they can be read, edited, or pasted into a DuckDB shell
directly.

```bash
python src/analysis.py --list          # what's available
python src/analysis.py                 # run everything
python src/analysis.py --query abc_classification
```

| Query | Reads | Answers |
|---|---|---|
| `abc_classification` | Parquet | Which items carry the revenue |
| `weekly_seasonality_by_category` | Parquet | Is there a weekly pattern worth modeling |
| `when_model_beats_naive` | DuckDB | Does forecast quality track demand volume |
| `accuracy_vs_demand_pattern` | both | Does intermittency predict forecast quality |
| `store_forecast_quality` | DuckDB | Best and worst forecast items per store |
| `mase_outlier_diagnosis` | DuckDB | Are extreme scores real failures or metric artifacts |

Two of them read the 46M-row Parquet in place through DuckDB's `read_parquet()`, so a
full-catalog aggregation needs no import step and no second copy of the data.

## Batch runner

`scripts/run_batch_forecasts.py` sweeps many combinations headlessly through the same code
path as the dashboard.

```bash
python scripts/run_batch_forecasts.py --top-n 5              # highest-demand per store
python scripts/run_batch_forecasts.py --sample 20 --seed 42  # random per store
python scripts/run_batch_forecasts.py --stratified 60        # 60 per demand decile
```

`--stratified` exists because random sampling follows the catalog's own skew. A third of
item/store pairs are C-class, so a random draw is dominated by near-zero-demand series and
tells you little about the items that carry the revenue. Stratifying by demand decile is what
made the intermittency result above legible.

A Prophet failure on one item logs a warning and the sweep continues, so a long run does not
die on combination 400 of 600. Combinations already saved are skipped by default, making an
interrupted sweep resumable. The run is sequential by design; see Known Limitations.

Lookups use a sorted MultiIndex built once per sweep rather than scanning the full frame per
combination, which took per-run time from 7.3s to 1.9s on the full dataset.

## Testing

91 tests, 100% line coverage on `src/pipeline.py`. CI runs them on Python 3.10, 3.11, and
3.12 on every push, plus a lint job.

```bash
pip install -r requirements-dev.txt
pytest                      # everything
pytest -m "not slow"        # skips the real Prophet fits
pytest --cov=pipeline --cov=data_prep --cov=analysis --cov-report=term-missing
```

| File | Tests | Covers |
|---|---|---|
| `test_persistence.py` | 20 | DB round-trips, run-ID sequencing, SQL injection regression, summary aggregation, `DatabaseError` paths |
| `test_analysis.py` | 20 | Each SQL query's shape and relationships, CLI as a subprocess |
| `test_forecasting.py` | 18 | MASE and MAPE arithmetic, `ForecastingError` propagation, real Prophet integration |
| `test_data_prep.py` | 15 | Filtering, indexed lookup, memory-optimized dtypes, Parquet round-trip |
| `test_data_prep_pipeline.py` | 11 | Melt, joins, the pre-launch filter, CLI as a subprocess |
| `test_inventory_math.py` | 7 | EOQ, safety stock, ROP against closed-form formulas |

Tests run against temporary databases and synthetic files, never against real data, so the
suite cannot corrupt saved runs. Prophet's failure paths are exercised with a stub because
real Prophet cannot be made to fail on demand; its success paths use the real library.

## Setup

```bash
git clone https://github.com/samfayn/demand-forecast-inventory-pipeline.git
cd demand-forecast-inventory-pipeline
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

Download `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` from the
[M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into
`data/`, then:

```bash
python src/data_prep.py --data-dir data --output data/sales_clean.parquet
streamlit run app.py
```

Without the raw CSVs the app falls back to the committed demo subset in `demo_data/`, so it
runs immediately after cloning.

## Project structure

```
demand-forecast-inventory-pipeline/
│
├── .github/workflows/tests.yml   # CI: pytest on 3.10/3.11/3.12, lint, requirements validation
├── .streamlit/config.toml
├── data/                         # Not tracked: full 46M-row dataset and run database
├── demo_data/                    # Tracked: 8.7M-row subset + 1,066 runs for the deployment
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Calls src/data_prep.py, keeps exploratory analysis
│   ├── 02_forecasting.ipynb         # Prophet exploration and tuning
│   └── 03_inventory_optimization.ipynb
├── scripts/
│   ├── migrations/               # Numbered, idempotent schema migrations
│   ├── build_demo_data.py        # Builds the deployable subset
│   └── run_batch_forecasts.py    # Headless multi-combination sweep
├── sql/                          # Six analytical queries
├── src/
│   ├── data_prep.py              # Batch pipeline: ingest, reshape, join, filter, persist
│   ├── pipeline.py               # Forecasting, evaluation, inventory math, DB I/O
│   └── analysis.py               # SQL loader and runner
├── tests/                        # 91 tests
├── app.py                        # Streamlit dashboard
├── pytest.ini
├── requirements.txt              # Runtime only
└── requirements-dev.txt          # Adds pytest, coverage, linting, Jupyter
```

Notebooks 02 and 03 document how the forecasting and inventory logic were developed. The
production path is `data_prep.py` → Parquet → `app.py` or the batch runner → DuckDB.

## Known limitations

**Prophet treats every series independently.** No hierarchical or cross-series structure is
modeled, so a store-wide promotion has to be learned separately by every affected item. The
M5 winners all exploited that structure. Given that the model only edges out a naive
baseline, this is the most likely place to find real improvement.

**The model should probably not be used on the intermittent tail.** The data above shows it
losing to seasonal naive above 60% zero-days, which is 449 of 825 backtested runs. Croston's
method or a negative-binomial demand model would fit that segment better. The app runs
Prophet on those items anyway and flags the result rather than switching methods.

**The inventory math assumes normally distributed demand and constant lead time.** Neither
holds well for low-volume intermittent items, which is most of this catalog.

**MASE breaks down when a series changes regime between training and holdout.** The scale
factor is computed on the training window, so an item that was dormant during training and
active afterward produces an enormous ratio that says nothing about the model. One run in 825
hit this. Reporting the median rather than the mean keeps it from distorting the summary.

**`run_id` is assigned via `MAX(run_id) + 1`, which is not concurrency-safe.** Two processes
writing at once can collide. The batch runner is sequential specifically because of this.
A DuckDB sequence or a UUID key would fix it properly.

**The insert into `forecast_runs` was originally positional and broke silently under schema
migration.** `ALTER TABLE` appends a column at the end while `CREATE TABLE` placed it
mid-schema, so a migrated database and a fresh one had different column orders and positional
inserts wrote values into the wrong fields without raising. It is now a named-column insert.

**There is no orchestrator.** `data_prep.py` runs as a single process with no retry or
partial-failure recovery. Splitting it into tasks under Airflow or Prefect would be the next
step for anything running unattended.

**The deployed app is memory-constrained.** `load_data` reads only the eight columns used
downstream and stores identifiers as categoricals, which takes the demo subset from 1.4 GB to
191 MB in memory. The full dataset still cannot be deployed on free hosting.

## Dataset

[M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy): five years of
daily sales across 3,049 products in 10 Walmart stores in California, Texas, and Wisconsin.