# ⚡ PowerSight — Renewable Energy Forecasting with Temporal Fusion Transformers

> Predicting solar and wind energy generation for Germany and the United Kingdom using state-of-the-art deep learning — one hour at a time.

---

## What is this?

Energy grids are fragile. Too much or too little renewable generation at the wrong moment destabilizes the entire network. **PowerSight** tackles this by forecasting hourly electricity load using a **Temporal Fusion Transformer (TFT)** — a model that doesn't just predict, it *explains* which variables drove each forecast.

Trained on 5+ years of real ENTSO-E grid data (2015–2020) fused with live weather observations, PowerSight achieves a **validation SMAPE of 3.1%** across two of Europe's largest energy markets.

---

## Pipeline Overview

```
Raw OPSD Grid Data (300 columns, 50K rows)
        ↓
   Filter → DE + GB only
        ↓
   Merge with Open-Meteo Weather API
        ↓
   Clean + Impute missing values
        ↓
   StandardScale + Sliding Window sequences
        ↓
   Train TFT (1.4M params) via PyTorch Lightning
        ↓
   Evaluate + Interpret attention weights
```

---

## Dataset

| Source | Description |
|--------|-------------|
| [OPSD Time Series](https://open-power-system-data.org/) | Hourly solar/wind generation, load, capacity — 32 European countries |
| [Open-Meteo Archive API](https://open-meteo.com/) | Hourly temperature, shortwave radiation, wind speed |

**Countries:** Germany (DE) · United Kingdom (GB)  
**Period:** January 2015 – September 2020  
**Granularity:** 1 hour

---

## Models

| Model | Purpose | Val SMAPE |
|-------|---------|-----------|
| Temporal Fusion Transformer | Primary forecaster | **3.1%** |
| LSTM | Baseline comparison | — |

TFT advantages over vanilla LSTM:
- Multi-horizon probabilistic forecasts (10th / 50th / 90th percentile)
- Learns per-country static context
- Interpretable attention over past time steps
- Variable importance scores via gating mechanisms

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Data_EDA.ipynb` | Exploratory analysis — distributions, correlations, seasonal patterns |
| `Data_preprocess.ipynb` | Filtering, weather merging, imputation, export |
| `transformer_train.ipynb` | TFT + LSTM training, evaluation, attention interpretation |

---

## Results

- **Validation SMAPE: 3.1%** on 24-step ahead load forecasting
- Attention heatmaps reveal model focuses on recent 6–12 hours for load prediction
- Variable selection gates confirm `windspeed_10m` and `shortwave_radiation` as top drivers

---

## Setup

```bash
git clone https://github.com/yourusername/PowerSight.git
cd PowerSight

conda create -n powersight python=3.12
conda activate powersight

pip install pandas numpy scikit-learn torch pytorch-lightning \
            pytorch-forecasting lightning xgboost optuna \
            statsmodels captum shap matplotlib joblib
```

Download the OPSD dataset and place it at:

```
Dataset/time_series_60min_singleindex.csv
```

---

## Tech Stack

`Python` · `PyTorch` · `PyTorch Lightning` · `PyTorch Forecasting` · `pandas` · `scikit-learn` · `XGBoost` · `Optuna` · `SHAP` · `Open-Meteo API`

---

## Author

**Shivansh Gupta**
