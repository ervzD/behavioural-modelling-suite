import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.validation import validate_bayesian, validate_ddm, validate_kalman


st.set_page_config(page_title="Validation", layout="wide")

st.title("Model validation")

st.markdown(
    """
    The models in this suite are validated in two ways: recovery of known
    generating parameters from synthetic data, and reproduction of canonical
    paradigms from the published literature. The results below are generated
    live from the current implementation, so any future change to the model
    code will be reflected here.
    """
)

st.header("Bayesian integration: reproducing Alais & Burr (2004)")

st.markdown(
    """
    Alais and Burr manipulated the reliability of the visual component of an
    audio-visual localisation task and showed that combined-modality weights
    track the inverse-variance optimum. We reproduce their paradigm by
    generating data across a range of visual noise levels and fitting the
    model at each level.
    """
)

with st.spinner("Running Bayesian validation..."):
    bayes_results = validate_bayesian()

st.dataframe(bayes_results.round(3), width="stretch")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bayes_results["true_sigma_visual"],
    y=bayes_results["optimal_weight_visual"],
    mode="lines+markers", name="Optimal (theoretical)",
))
fig.add_trace(go.Scatter(
    x=bayes_results["true_sigma_visual"],
    y=bayes_results["fitted_weight_visual"],
    mode="lines+markers", name="Fitted (from data)",
    line=dict(dash="dash"),
))
fig.update_layout(
    title="Visual weight: theoretical optimum vs. fitted estimate",
    xaxis_title="True visual noise (sigma)",
    yaxis_title="Visual weight",
)
st.plotly_chart(fig, width="stretch")

st.header("Drift diffusion: parameter recovery")

st.markdown(
    """
    We generate synthetic data from four sets of known drift rate, boundary
    separation, and non-decision time, then fit the model to each and compare
    recovered parameters against ground truth. Accurate recovery demonstrates
    that the likelihood and optimisation procedure are correctly implemented.
    """
)

with st.spinner("Fitting DDM (may take a minute)..."):
    ddm_results = validate_ddm()

st.dataframe(ddm_results.round(3), width="stretch")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ddm_results["true_v"], y=ddm_results["fitted_v"],
    mode="markers", name="Drift rate (v)", marker=dict(size=10),
))
fig.add_trace(go.Scatter(
    x=ddm_results["true_a"], y=ddm_results["fitted_a"],
    mode="markers", name="Boundary (a)", marker=dict(size=10),
))
fig.add_trace(go.Scatter(
    x=ddm_results["true_t0"], y=ddm_results["fitted_t0"],
    mode="markers", name="Non-decision time (t0)", marker=dict(size=10),
))
min_val = min(ddm_results["true_v"].min(), ddm_results["true_a"].min(), ddm_results["true_t0"].min())
max_val = max(ddm_results["true_v"].max(), ddm_results["true_a"].max(), ddm_results["true_t0"].max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", name="Identity", line=dict(dash="dot", color="grey"),
))
fig.update_layout(
    title="DDM parameter recovery",
    xaxis_title="True value", yaxis_title="Fitted value",
)
st.plotly_chart(fig, width="stretch")

st.header("Kalman filter: parameter recovery")

st.markdown(
    """
    We simulate random-walk state dynamics with known process and observation
    noise, then fit the model and compare recovered noise parameters against
    their true values.
    """
)

with st.spinner("Fitting Kalman filter..."):
    kalman_results = validate_kalman()

st.dataframe(kalman_results.round(3), width="stretch")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=kalman_results["true_process_noise"],
    y=kalman_results["fitted_process_noise"],
    mode="markers", name="Process noise", marker=dict(size=10),
))
fig.add_trace(go.Scatter(
    x=kalman_results["true_observation_noise"],
    y=kalman_results["fitted_observation_noise"],
    mode="markers", name="Observation noise", marker=dict(size=10),
))
min_val = min(kalman_results["true_process_noise"].min(), kalman_results["true_observation_noise"].min())
max_val = max(kalman_results["true_process_noise"].max(), kalman_results["true_observation_noise"].max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", name="Identity", line=dict(dash="dot", color="grey"),
))
fig.update_layout(
    title="Kalman filter parameter recovery",
    xaxis_title="True value", yaxis_title="Fitted value",
)
st.plotly_chart(fig, width="stretch")