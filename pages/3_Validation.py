import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.validation import validate_bayesian, validate_ddm


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
    Alais and Burr manipulated the reliability of the visual component of
    an audio-visual localisation task and showed that combined-modality
    weights track the inverse-variance optimum. We reproduce their paradigm
    by generating data across a range of visual noise levels and fitting
    the model at each level.
    """
)

with st.spinner("Running Bayesian validation across noise levels..."):
    bayes_results = validate_bayesian()

st.dataframe(bayes_results.round(3), use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bayes_results["true_sigma_visual"],
    y=bayes_results["optimal_weight_visual"],
    mode="lines+markers",
    name="Optimal (theoretical)",
))
fig.add_trace(go.Scatter(
    x=bayes_results["true_sigma_visual"],
    y=bayes_results["fitted_weight_visual"],
    mode="lines+markers",
    name="Fitted (from data)",
    line=dict(dash="dash"),
))
fig.update_layout(
    title="Visual weight: theoretical optimum vs. fitted estimate",
    xaxis_title="True visual noise (sigma)",
    yaxis_title="Visual weight",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    The fitted visual weights track the theoretical optimum almost exactly,
    confirming that the implementation recovers the Bayesian combination
    rule across a 12-fold range of visual reliability.
    """
)

st.header("Drift diffusion: parameter recovery (Ratcliff & McKoon, 2008)")

st.markdown(
    """
    We generate synthetic data from four sets of known drift rate, boundary
    separation, and non-decision time, then fit the model to each and compare
    recovered parameters against ground truth. Accurate recovery demonstrates
    that the likelihood and optimisation procedure are correctly implemented.
    """
)

with st.spinner("Fitting DDM across parameter sets (may take a minute)..."):
    ddm_results = validate_ddm()

st.dataframe(ddm_results.round(3), use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ddm_results["true_v"], y=ddm_results["fitted_v"],
    mode="markers", name="Drift rate (v)",
    marker=dict(size=10),
))
fig.add_trace(go.Scatter(
    x=ddm_results["true_a"], y=ddm_results["fitted_a"],
    mode="markers", name="Boundary (a)",
    marker=dict(size=10),
))
fig.add_trace(go.Scatter(
    x=ddm_results["true_t0"], y=ddm_results["fitted_t0"],
    mode="markers", name="Non-decision time (t0)",
    marker=dict(size=10),
))
min_val = min(
    ddm_results["true_v"].min(), ddm_results["true_a"].min(), ddm_results["true_t0"].min(),
)
max_val = max(
    ddm_results["true_v"].max(), ddm_results["true_a"].max(), ddm_results["true_t0"].max(),
)
fig.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", name="Identity",
    line=dict(dash="dot", color="grey"),
))
fig.update_layout(
    title="DDM parameter recovery",
    xaxis_title="True value",
    yaxis_title="Fitted value",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    Points lying close to the identity line indicate accurate recovery.
    Small deviations are expected due to finite-sample variability.
    """
)