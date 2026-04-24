import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.kalman import KalmanFilter
from src.utils.data_generation import generate_kalman_dataset
from src.utils.data_validation import validate_kalman_data

st.set_page_config(page_title="Kalman Filter", layout="wide")

st.title("Kalman filter")

st.markdown(
    """
    Fit a Kalman filter to time-series observations of a latent, drifting state.
    The model assumes the underlying state performs a Gaussian random walk and
    that observations are noisy measurements of the current state. It estimates
    the process noise (how much the state drifts per step) and the observation
    noise (how much measurement uncertainty there is).

    This is the natural temporal extension of the Bayesian integration model:
    at every time step it combines a prior (from the previous estimate) with
    a likelihood (the current observation) using the same inverse-variance
    weighting rule.
    """
)

st.header("1. Load data")

data_source = st.radio(
    "Data source",
    options=["Use example dataset", "Upload my own CSV"],
    horizontal=True,
)

data = None

if data_source == "Use example dataset":
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        q_true = st.number_input("True process noise", 0.05, 3.0, 0.5, 0.05)
    with col_b:
        r_true = st.number_input("True observation noise", 0.05, 5.0, 1.2, 0.05)
    with col_c:
        n_steps = st.number_input("Number of steps", 50, 1000, 200, 50)
    seed_input = st.text_input("Random seed", value="42")
    try:
        seed = int(seed_input)
        if seed < 0:
            st.error("Seed must be a non-negative integer.")
            seed = None
    except ValueError:
        st.error("Seed must be a whole number.")
        seed = None
    if seed is not None:
        data = generate_kalman_dataset(
            process_noise=q_true, observation_noise=r_true,
            n_steps=int(n_steps), seed=seed,
        )
else:
    uploaded = st.file_uploader(
        "Upload a CSV with an 'observation' column (one row per time step)",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if data is None:
    st.info("Load a dataset to continue.")
    st.stop()

try:
    validate_kalman_data(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.header("2. Data preview")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data.head(20), width="stretch")
with col2:
    st.markdown("**Summary**")
    st.markdown(f"- Time steps: {len(data)}")
    st.markdown(f"- Observation mean: {data['observation'].mean():.3f}")
    st.markdown(f"- Observation std: {data['observation'].std():.3f}")

st.header("3. Fit model")

if st.button("Fit Kalman filter", type="primary"):
    with st.spinner("Fitting..."):
        try:
            model = KalmanFilter().fit(data)
            filtered_means, filtered_vars = model.filter(data["observation"].values)
            st.session_state["kalman_model"] = model
            st.session_state["kalman_data"] = data
            st.session_state["kalman_filtered"] = (filtered_means, filtered_vars)
            st.success("Model fitted.")
        except Exception as exc:
            st.error(f"Fitting failed: {exc}")
            st.stop()

if "kalman_model" not in st.session_state:
    st.info("Fit the model to see results.")
    st.stop()

model = st.session_state["kalman_model"]
data = st.session_state["kalman_data"]
filtered_means, filtered_vars = st.session_state["kalman_filtered"]

st.header("4. Fitted parameters")

summary = model.summary()
c1, c2 = st.columns(2)
c1.metric("Process noise", f"{summary['process_noise']:.3f}")
c2.metric("Observation noise", f"{summary['observation_noise']:.3f}")

ll = model.log_likelihood(data)
bic = model.bic(data)
st.markdown(f"**Log-likelihood:** {ll:.2f}    |    **BIC:** {bic:.2f}")

st.header("5. Visualisations")

tab1, tab2 = st.tabs(["Filtered trajectory", "Residuals"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"] if "time" in data.columns else np.arange(len(data)),
        y=data["observation"], mode="markers", name="Observations",
        marker=dict(size=5, opacity=0.6),
    ))
    if "true_state" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["true_state"], mode="lines",
            name="True state", line=dict(dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=data["time"] if "time" in data.columns else np.arange(len(data)),
        y=filtered_means, mode="lines", name="Filtered estimate",
        line=dict(width=3),
    ))
    upper = filtered_means + 1.96 * np.sqrt(filtered_vars)
    lower = filtered_means - 1.96 * np.sqrt(filtered_vars)
    fig.add_trace(go.Scatter(
        x=np.concatenate([np.arange(len(data)), np.arange(len(data))[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(100,100,200,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% credible interval",
        showlegend=True,
    ))
    fig.update_layout(
        title="Observations, filtered estimate and uncertainty over time",
        xaxis_title="Time step",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    residuals = data["observation"].values - filtered_means
    fig = px.histogram(
        x=residuals, nbins=30,
        title="Innovation residuals (observation minus filtered estimate)",
    )
    fig.update_layout(xaxis_title="Residual", yaxis_title="Count")
    st.plotly_chart(fig, width="stretch")

st.header("6. Export")

summary_df = pd.DataFrame([summary])
csv_buffer = io.StringIO()
summary_df.to_csv(csv_buffer, index=False)
st.download_button(
    "Download fitted parameters (CSV)",
    csv_buffer.getvalue(),
    file_name="kalman_fit.csv",
    mime="text/csv",
)