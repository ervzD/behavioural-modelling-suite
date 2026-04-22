import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.bayesian import BayesianIntegration
from src.utils.data_generation import generate_bayesian_dataset
from src.utils.data_validation import validate_bayesian_data


st.set_page_config(page_title="Bayesian Integration", layout="wide")

st.title("Bayesian cue integration")

st.markdown(
    """
    Fit an optimal Bayesian cue combination model to multisensory perception
    data. The model estimates the noise (standard deviation) of each unimodal
    channel from unimodal trials and predicts combined-modality responses as
    a reliability-weighted average.
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
    seed_input = st.text_input("Random seed for example data", value="42")
    try:
        seed = int(seed_input)
        if seed < 0:
            st.error("Seed must be a non-negative integer.")
            seed = None
    except ValueError:
        st.error("Seed must be a whole number.")
        seed = None
    sigma_v_input = st.text_input("True visual noise (sigma)", value="1.5")
    try:
        sigma_v = float(sigma_v_input)
        if sigma_v <= 0:
            st.error("Visual noise must be a positive number.")
            sigma_v = None
    except ValueError:
        st.error("Visual noise must be a number.")
        sigma_v = None

    sigma_a_input = st.text_input("True auditory noise (sigma)", value="3.0")
    try:
        sigma_a = float(sigma_a_input)
        if sigma_a <= 0:
            st.error("Auditory noise must be a positive number.")
            sigma_a = None
    except ValueError:
        st.error("Auditory noise must be a number.")
        sigma_a = None

    n_input = st.text_input("Trials per condition", value="200")
    try:
        n = int(n_input)
        if n < 1:
            st.error("Trials per condition must be a positive integer.")
            n = None
    except ValueError:
        st.error("Trials per condition must be a whole number.")
        n = None
    if all(v is not None for v in (seed, sigma_v, sigma_a, n)):
        data = generate_bayesian_dataset(
            n_trials_per_condition=n,
            sigma_visual=sigma_v,
            sigma_auditory=sigma_a,
            seed=seed,
        )
else:
    uploaded = st.file_uploader(
        "Upload a CSV with columns: modality, stimulus, response",
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
    validate_bayesian_data(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.header("2. Data preview")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data.head(20), use_container_width=True)
with col2:
    counts = data["modality"].value_counts()
    st.markdown("**Trials per modality**")
    for mod, n in counts.items():
        st.markdown(f"- {mod}: {n}")

st.header("3. Fit model")

if st.button("Fit Bayesian model", type="primary"):
    try:
        model = BayesianIntegration().fit(data)
        st.session_state["bayesian_model"] = model
        st.session_state["bayesian_data"] = data
        st.success("Model fitted.")
    except Exception as exc:
        st.error(f"Fitting failed: {exc}")
        st.stop()

if "bayesian_model" not in st.session_state:
    st.info("Fit the model to see results.")
    st.stop()

model = st.session_state["bayesian_model"]
data = st.session_state["bayesian_data"]

st.header("4. Fitted parameters")

summary = model.summary()

param_col1, param_col2, param_col3, param_col4 = st.columns(4)
param_col1.metric("Visual noise (sigma)", f"{summary['sigma_visual']:.3f}")
param_col2.metric("Auditory noise (sigma)", f"{summary['sigma_auditory']:.3f}")
param_col3.metric("Visual weight", f"{summary['weight_visual']:.3f}")
param_col4.metric("Auditory weight", f"{summary['weight_auditory']:.3f}")

if len(data[data["modality"] == "combined"]) > 0:
    ll = model.log_likelihood(data)
    bic = model.bic(data)
    st.markdown(f"**Log-likelihood:** {ll:.2f}    |    **BIC:** {bic:.2f}")

st.header("5. Visualisations")

tab1, tab2, tab3 = st.tabs(["Response distributions", "Weight prediction", "Residuals"])

with tab1:
    fig = px.histogram(
        data,
        x="response",
        color="modality",
        nbins=40,
        barmode="overlay",
        opacity=0.6,
        title="Response distribution by modality",
    )
    fig.update_layout(xaxis_title="Response", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    sigma_v_range = np.linspace(0.1, max(summary["sigma_visual"] * 3, 6), 100)
    w_v_predicted = (summary["sigma_auditory"] ** 2) / (
        sigma_v_range ** 2 + summary["sigma_auditory"] ** 2
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sigma_v_range, y=w_v_predicted, mode="lines",
        name="Optimal visual weight",
    ))
    fig.add_trace(go.Scatter(
        x=[summary["sigma_visual"]], y=[summary["weight_visual"]],
        mode="markers", marker=dict(size=12), name="Fitted value",
    ))
    fig.update_layout(
        title="Visual weight as a function of visual noise",
        xaxis_title="Visual noise (sigma)",
        yaxis_title="Visual weight",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    res_frames = []
    for mod in ("visual", "auditory"):
        sub = data[data["modality"] == mod]
        if len(sub) > 0:
            res_frames.append(pd.DataFrame({
                "modality": mod,
                "residual": sub["response"] - sub["stimulus"],
            }))
    if res_frames:
        residuals = pd.concat(res_frames, ignore_index=True)
        fig = px.box(residuals, x="modality", y="residual",
                     title="Residuals by modality (response minus stimulus)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No unimodal trials available to compute residuals.")

st.header("6. Export")

summary_df = pd.DataFrame([summary])
csv_buffer = io.StringIO()
summary_df.to_csv(csv_buffer, index=False)
st.download_button(
    "Download fitted parameters (CSV)",
    csv_buffer.getvalue(),
    file_name="bayesian_fit.csv",
    mime="text/csv",
)