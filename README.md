# Demand Forecasting & Inventory Optimization Pipeline

An end-to-end data pipeline that combines demand forecasting with ISE inventory 
optimization principles to generate actionable inventory policies from raw retail data.

## Overview

This project was built to demonstrate how Industrial & Systems Engineering concepts 
like Economic Order Quantity (EOQ), Safety Stock, and Reorder Point calculations can 
be powered by modern data science tools. It uses the M5 Forecasting dataset — a 
real-world Walmart retail dataset — to forecast product demand and generate optimal 
inventory policies through an interactive dashboard.

## Features

- **Data Pipeline** — Ingests and transforms 3 raw datasets totaling 58M+ rows into 
  a clean, analysis-ready format using Pandas
- **Demand Forecasting** — Trains a Facebook Prophet model per product to generate 
  90-day demand forecasts with uncertainty intervals
- **Inventory Optimization** — Calculates Safety Stock, Reorder Point, and EOQ from 
  forecast outputs using ISE inventory theory
- **Interactive Dashboard** — Streamlit app that allows users to select any 
  product/store combination and adjust inventory parameters in real time

## Tech Stack

- Python, Pandas, NumPy
- Facebook Prophet
- Streamlit
- Matplotlib
- Parquet / PyArrow

## Project Structure
```
demand-forecast-inventory-pipeline/
│
├── data/                          # Raw and processed data (not tracked in git)
├── notebooks/
│   ├── 01_exploration.ipynb       # Data ingestion, cleaning, and transformation
│   ├── 02_forecasting.ipynb       # Prophet model training and evaluation
│   └── 03_inventory_optimization.ipynb  # Inventory policy calculations
├── src/
│   └── pipeline.py                # Core reusable pipeline functions
├── app.py                         # Streamlit dashboard
└── requirements.txt               # Project dependencies
```

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/samfayn/demand-forecast-inventory-pipeline.git
cd demand-forecast-inventory-pipeline
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the data**

Download the following files from the 
[M5 Forecasting Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) 
and place them in the `data/` folder:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

**5. Run the data pipeline**

Run the notebooks in order:
- `notebooks/01_exploration.ipynb` — processes raw data and saves `data/sales_clean.parquet`
- `notebooks/02_forecasting.ipynb` — explores the forecasting model
- `notebooks/03_inventory_optimization.ipynb` — explores inventory calculations

**6. Launch the dashboard**
```bash
streamlit run app.py
```

## Inventory Model

The inventory policy is calculated using classic ISE formulas:

| Metric | Formula |
|---|---|
| Safety Stock | Z × σ(demand) × √(lead time) |
| Reorder Point | (Avg daily demand × lead time) + Safety Stock |
| EOQ | √(2 × annual demand × ordering cost / holding cost × price) |

A 95% service level (Z = 1.645) is used by default, adjustable in the dashboard.

## Dataset

This project uses the 
[M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) 
from Kaggle, which contains 5 years of daily sales data across 3,049 products in 
10 Walmart stores across California, Texas, and Wisconsin.
```