# 🌾 Crop Yield Prediction System (Python GUI)

A beginner-friendly machine learning project that predicts crop yield
(tons/hectare) from environmental factors, wrapped in a desktop GUI
with pie charts, bar charts, scatter plots, and histograms.

## Features

- **Predict Yield tab** — sliders/dropdowns for Rainfall, Temperature,
  Humidity, Fertilizer usage, Soil type, Crop type → predicted yield,
  with a live bar chart comparing your prediction to dataset benchmarks
  and a pie chart of factor influence.
- **Data Dashboard tab** — 6 charts in one view: soil-type pie chart,
  average yield per crop (bar), rainfall vs yield (scatter),
  temperature vs yield (scatter), average yield per soil type (bar),
  and yield distribution (histogram).
- **Model Insights tab** — feature importance chart and a
  predicted-vs-actual scatter plot with test-set R², MAE, RMSE.

## Tech Stack

- **ML**: scikit-learn `RandomForestRegressor`
- **Data**: pandas, numpy (synthetic dataset — see below)
- **GUI**: tkinter (built into Python)
- **Charts**: matplotlib, embedded via `FigureCanvasTkAgg`

## Project Structure

```
crop_yield_project/
├── generate_data.py   # creates crop_data.csv (1500 synthetic samples)
├── model.py            # CropYieldModel: trains + predicts
├── app.py               # Tkinter GUI application (run this)
├── crop_data.csv       # generated dataset
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Run

```bash
python3 generate_data.py   # only needed once, app.py auto-generates if missing
python3 app.py
```

A window opens with three tabs: Predict Yield, Data Dashboard, Model Insights.

## Using Your Own Real Data

Replace `crop_data.csv` with your own file, keeping these exact columns:

| Column | Type | Notes |
|---|---|---|
| Rainfall_mm | float | mm |
| Temperature_C | float | °C |
| Humidity_pct | float | % |
| Soil_Type | string | e.g. Sandy, Clay, Loamy |
| Fertilizer_kg_per_ha | float | kg/hectare |
| Crop_Type | string | e.g. Wheat, Rice, Maize |
| Yield_tons_per_ha | float | target variable |

Then just re-run `python3 app.py` — it retrains automatically on startup.

## Notes

- The bundled dataset is **synthetic** (generated with realistic but
  made-up relationships) so the project runs out of the box. Swap in
  real agricultural data (e.g. from Kaggle or government agri datasets)
  for a genuine analysis.
- Model achieves **R² ≈ 0.81** on the synthetic test set.
- To improve: try `GradientBoostingRegressor`, hyperparameter tuning
  (`GridSearchCV`), or add features like soil pH, sunlight hours, or
  pesticide usage.
