import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PowerSight — Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ PowerSight")
st.sidebar.markdown("Renewable Energy Load Forecasting")
st.sidebar.divider()

country = st.sidebar.selectbox("Country", ["Germany", "UK"])
anomaly_sigma = st.sidebar.slider("Anomaly threshold (σ)", 1.0, 3.0, 2.0, 0.1)
checkpoint_path = st.sidebar.text_input(
    "Model checkpoint path",
    placeholder="/path/to/best-tft-*.ckpt"
)
scaler_path = st.sidebar.text_input("Scaler path", value="scaler.pkl")
data_path = st.sidebar.text_input(
    "Cleaned data path",
    value="cleaned_germany_energy_weather.csv" if country == "Germany" else "cleaned_uk_energy_weather.csv"
)

run = st.sidebar.button("Run Forecast", type="primary", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("⚡ PowerSight — Renewable Energy Forecast")
st.markdown("24-hour probabilistic load forecasting with anomaly detection")
st.divider()

@st.cache_resource
def load_model(ckpt_path):
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_data
def load_data(path, country_name):
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df["country"] = country_name
    df["is_Germany"] = 1 if country_name == "Germany" else 0
    df["is_UK"] = 1 if country_name == "UK" else 0

    col_map_de = {
        "DE_load_actual_entsoe_transparency": "load_actual",
        "DE_solar_capacity": "solar_capacity",
        "DE_solar_generation_actual": "solar_generation",
        "DE_wind_generation_actual": "wind_generation",
        "DE_wind_onshore_generation_actual": "wind_onshore",
        "DE_wind_offshore_generation_actual": "wind_offshore",
    }
    col_map_uk = {
        "GB_GBN_load_actual_entsoe_transparency": "load_actual",
        "GB_GBN_solar_capacity": "solar_capacity",
        "GB_GBN_solar_generation_actual": "solar_generation",
        "GB_GBN_wind_generation_actual": "wind_generation",
        "GB_GBN_wind_onshore_generation_actual": "wind_onshore",
        "GB_GBN_wind_offshore_generation_actual": "wind_offshore",
    }
    col_map = col_map_de if country_name == "Germany" else col_map_uk
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing)
    return df

def build_dataset(df):
    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    df["time_id"] = (
        (df["utc_timestamp"] - df["utc_timestamp"].min())
        .dt.total_seconds() // 3600
    ).astype(int)

    max_time_id = df["time_id"].max()
    val_cutoff = max_time_id - 336
    train_df = df[df["time_id"] <= val_cutoff]
    val_df = df[df["time_id"] > val_cutoff]

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_id",
        target="load_actual",
        group_ids=["country"],
        min_encoder_length=168,
        max_encoder_length=168,
        min_prediction_length=24,
        max_prediction_length=24,
        static_categoricals=["country"],
        static_reals=["is_Germany", "is_UK"],
        time_varying_known_reals=["time_id", "temperature_2m", "shortwave_radiation", "windspeed_10m"],
        time_varying_unknown_reals=[
            "load_actual", "solar_capacity", "solar_generation",
            "wind_generation", "wind_onshore", "wind_offshore"
        ],
        target_normalizer=GroupNormalizer(groups=["country"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )
    return val_dataset, val_df

def detect_anomalies(actuals, p50, sigma_thresh):
    residuals = actuals - p50
    mu = residuals.mean()
    sigma = residuals.std()
    anomaly_mask = np.abs(residuals - mu) > sigma_thresh * sigma
    return anomaly_mask, mu, sigma

# ── Forecast ──────────────────────────────────────────────────────────────────
if run:
    if not checkpoint_path:
        st.error("Provide model checkpoint path in sidebar.")
        st.stop()

    with st.spinner("Loading model and data..."):
        try:
            model = load_model(checkpoint_path)
            df = load_data(data_path, country)
        except Exception as e:
            st.error(f"Load failed: {e}")
            st.stop()

    with st.spinner("Building dataset and running inference..."):
        try:
            val_dataset, val_df = build_dataset(df)
            val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

            with torch.no_grad():
                raw_preds = model.predict(val_loader, mode="quantiles", return_x=False)

            preds_np = raw_preds.cpu().numpy()

            actuals_series = (
                val_df.sort_values("time_id")["load_actual"]
                .values[-preds_np.shape[0] * 24:]
                .reshape(-1, 24)
            )

        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    n_windows = preds_np.shape[0]
    timestamps = val_df.sort_values("time_id")["utc_timestamp"].values[-n_windows * 24:]

    p10 = preds_np[:, :, 0].flatten()
    p50 = preds_np[:, :, 1].flatten()
    p90 = preds_np[:, :, 2].flatten()
    actuals_flat = actuals_series.flatten()
    ts_flat = timestamps

    anomaly_mask, res_mu, res_sigma = detect_anomalies(actuals_flat, p50, anomaly_sigma)
    anomaly_ts = ts_flat[anomaly_mask]
    anomaly_actual = actuals_flat[anomaly_mask]
    anomaly_pred = p50[anomaly_mask]

    # ── Metrics ───────────────────────────────────────────────────────────────
    smape = np.mean(2 * np.abs(actuals_flat - p50) / (np.abs(actuals_flat) + np.abs(p50) + 1e-8)) * 100
    mae = np.mean(np.abs(actuals_flat - p50))
    n_anomalies = anomaly_mask.sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SMAPE", f"{smape:.2f}%")
    col2.metric("MAE (scaled)", f"{mae:.2f}")
    col3.metric("Anomalies detected", int(n_anomalies))
    col4.metric("Coverage (p10–p90)", f"{np.mean((actuals_flat >= p10) & (actuals_flat <= p90))*100:.1f}%")

    st.divider()

    # ── Forecast plot ─────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["24-Hour Load Forecast with Confidence Bands", "Residuals + Anomalies"],
        vertical_spacing=0.08
    )

    fig.add_trace(go.Scatter(
        x=np.concatenate([ts_flat, ts_flat[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself",
        fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="p10–p90 band"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts_flat, y=p10,
        line=dict(color="rgba(99,110,250,0.4)", dash="dot", width=1),
        name="p10"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts_flat, y=p90,
        line=dict(color="rgba(99,110,250,0.4)", dash="dot", width=1),
        name="p90"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts_flat, y=p50,
        line=dict(color="#636EFA", width=2),
        name="p50 (forecast)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts_flat, y=actuals_flat,
        line=dict(color="#EF553B", width=1.5),
        name="Actual"
    ), row=1, col=1)

    if n_anomalies > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_ts, y=anomaly_actual,
            mode="markers",
            marker=dict(color="#FF6B00", size=8, symbol="x", line=dict(width=2)),
            name=f"Anomaly (>{anomaly_sigma}σ)"
        ), row=1, col=1)

    residuals = actuals_flat - p50
    fig.add_trace(go.Scatter(
        x=ts_flat, y=residuals,
        line=dict(color="#888", width=1),
        name="Residual",
        showlegend=False
    ), row=2, col=1)

    fig.add_hline(y=res_mu + anomaly_sigma * res_sigma, line_dash="dash",
                  line_color="#FF6B00", opacity=0.6, row=2, col=1)
    fig.add_hline(y=res_mu - anomaly_sigma * res_sigma, line_dash="dash",
                  line_color="#FF6B00", opacity=0.6, row=2, col=1)
    fig.add_hline(y=0, line_color="white", opacity=0.3, row=2, col=1)

    if n_anomalies > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_ts, y=residuals[anomaly_mask],
            mode="markers",
            marker=dict(color="#FF6B00", size=8, symbol="x", line=dict(width=2)),
            name="Anomaly",
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=650,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Load (scaled)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly table ─────────────────────────────────────────────────────────
    if n_anomalies > 0:
        st.subheader(f"🚨 {int(n_anomalies)} Anomalous Hours Detected")
        anomaly_df = pd.DataFrame({
            "Timestamp": anomaly_ts,
            "Actual Load": anomaly_actual.round(3),
            "Forecast (p50)": anomaly_pred.round(3),
            "Deviation": (anomaly_actual - anomaly_pred).round(3),
            "Deviation (σ)": ((anomaly_actual - anomaly_pred - res_mu) / res_sigma).round(2)
        })
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in forecast window.")

else:
    st.info("Configure settings in sidebar and click **Run Forecast** to begin.")
    st.markdown("""
    ### How to use
    1. Train TFT model in `transformer_train.ipynb`
    2. Save scaler: `joblib.dump(standar_scaler, 'scaler.pkl')`
    3. Note checkpoint path from `checkpoint_callback.best_model_path`
    4. Paste both paths in sidebar
    5. Select country → Run Forecast

    ### What you'll see
    - Actual vs predicted load with p10/p50/p90 confidence bands
    - Anomaly markers — hours deviating more than Nσ from forecast
    - Residual panel with threshold lines
    - Anomaly table with exact timestamps and deviation magnitudes
    """)