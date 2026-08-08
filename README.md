# Demand Forecasting & Inventory Optimization

A data preparation pipeline and interactive forecasting app that turns raw M5 retail
data into inventory policies, with forecast accuracy validated by holdout backtesting
and every run persisted to a queryable SQL database.

## Overview

This project applies Industrial & Systems Engineering inventory theory — Economic Order
Quantity, Safety Stock, and Reorder Point — on top of a demand forecasting model, using
the M5 Forecasting dataset of Walmart retail sales.

It has two parts:

1. **A batch data preparation pipeline** that reshapes and joins three raw CSV sources
   into a single analysis-ready Parquet dataset.
2. **An interactive Streamlit app** that forecasts demand for a selected product/store
   combination, derives an inventory policy from the forecast, validates accuracy
   against held-out data, and saves every run to DuckDB for later comparison.

## Data Preparation Pipeline

`notebooks/01_data_preparation.ipynb` performs the batch transformation:

| Step | Detail |
|---|---|
| Ingest | 3 raw CSVs — daily sales (30,490 × 1,913 wide), calendar (1,969 rows), sell prices (6.8M rows) |
| Reshape | Melts sales from wide to long format → **58.3M rows** |
| Join | Merges calendar on `d`; merges prices on the composite key `(store_id, item_id, wm_yr_wk)` |
| Filter | Drops 12.3M rows with no price — periods before an item was stocked at that store, which are structurally absent rather than zero-demand |
| Persist | Writes **46.0M rows** to `sales_clean.parquet` |

Final dataset: 3,049 products across 10 stores, 2011-01-29 to 2016-04-24.

The pre-launch filter matters for correctness. Those rows carry a sales value of 0, but
the item did not exist at that store yet. Treating them as real zero-demand observations
would bias every downstream forecast and inflate safety stock requirements.

## Forecasting

Prophet models are trained **on demand, one per product/store selection**, not
pre-trained across the catalog. Training is memoized with `@st.cache_data`, so
re-selecting a combination reuses the fitted model instead of refitting it.

Model configuration: yearly and weekly seasonality enabled, daily seasonality disabled,
`changepoint_prior_scale=0.05`, 90-day forecast horizon.

### Accuracy Validation

Every forecast is backtested before its inventory policy is trusted:

- Train on all data except the final 90 days
- Predict that held-out window and compare against actuals
- Report **MAPE** and **RMSE**

MAPE is computed only over days with nonzero actual sales, since percentage error is
undefined at zero and intermittent retail demand contains many zero days. Negative
predictions are clipped to zero before scoring — negative demand is not physically
meaningful.

## Inventory Model

| Metric | Formula |
|---|---|
| Safety Stock | Z × σ(demand) × √(lead time) |
| Reorder Point | (Avg daily demand × lead time) + Safety Stock |
| EOQ | √(2 × annual demand × ordering cost / (holding cost × price)) |

Default service level is 95% (Z = 1.645), adjustable in the dashboard along with lead
time, ordering cost, and holding cost.

## Persistence Layer

Results are written to a local DuckDB database (`data/inventory.duckdb`) using a
header-detail schema rather than a single wide table:

| Table | Grain | Contents |
|---|---|---|
| `forecast_runs` | One row per run | Run timestamp, item, store, input parameters, computed inventory metrics, MAPE, RMSE |
| `forecast_daily` | One row per forecast day | `yhat` and its lower/upper bounds, linked to a run via `run_id` |

Splitting by grain keeps run-level metrics from being duplicated across every forecast
day, and lets the app load a saved run's summary without reading its daily rows. The
**Saved Results** tab queries these tables directly, so any prior run can be revisited
or compared without retraining.

## Tech Stack

Python, Pandas, NumPy, Facebook Prophet, Streamlit, Matplotlib, DuckDB, Parquet/PyArrow

## Project Structure

```
demand-forecast-inventory-pipeline/
│
├── data/                                # Not tracked in git
│   ├── sales_clean.parquet              # Output of notebook 01
│   └── inventory.duckdb                 # Saved forecast runs
├── notebooks/
│   ├── 01_data_preparation.ipynb        # Batch pipeline: ingest, reshape, join, filter, persist
│   ├── 02_forecasting.ipynb             # Prophet exploration and tuning
│   └── 03_inventory_optimization.ipynb  # Inventory formula development
├── src/
│   └── pipeline.py                      # Core functions: forecasting, evaluation, inventory math, DB I/O
├── app.py                               # Streamlit dashboard
└── requirements.txt
```

Notebooks 02 and 03 are development artifacts documenting how the forecasting and
inventory logic were built. The production path is notebook 01 → `sales_clean.parquet`
→ `app.py` → DuckDB.

## Setup

**1. Clone**
```bash
git clone https://github.com/samfayn/demand-forecast-inventory-pipeline.git
cd demand-forecast-inventory-pipeline
```

**2. Virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux
```

**3. Dependencies**
```bash
pip install -r requirements.txt
```

**4. Data**

Download from the [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
into `data/`:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

**5. Run the preparation pipeline**
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```
Run all cells to produce `data/sales_clean.parquet`.

**6. Launch the dashboard**
```bash
streamlit run app.py
```

## Known Limitations

- The preparation stage runs as a notebook rather than under an orchestrator. Productionizing
  would mean splitting it into discrete tasks with retry and failure handling.
- Prophet treats each product/store series independently; no cross-series or hierarchical
  effects are modeled.
- Inventory calculations assume constant lead time and normally distributed demand, which
  holds poorly for intermittent low-volume items.
- `run_id` is assigned via `MAX(run_id) + 1`, which is not safe under concurrent writes.

## Dataset

[M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — 5 years of
daily sales across 3,049 products in 10 Walmart stores in California, Texas, and Wisconsin.
