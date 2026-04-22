import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.ddm import DriftDiffusionModel
from src.utils.data_generation import generate_ddm_dataset
from src.utils.data_validation import validate_ddm_data

st.set_page_config(page_title="Drift Diffusion", layout="wide")

st.title("Drift diffusion model")

st.markdown(
    """
    Fit a drift diffusion model to two-alternative forced choice data.
    The model describes decisions as noisy accumulation of evidence toward
    one of two boundaries, producing joint distributions of choice and
    reaction time.
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
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        v_true = st.number_input("True drift rate (v)", -3.0, 3.0, 1.0, 0.1)
    with col_b:
        a_true = st.number_input("True boundary (a)", 0.5, 3.0, 1.2, 0.1)
    with col_c:
        t0_true = st.number_input("True non-decision time (t0)", 0.05, 1.0, 0.25, 0.05)
    with col_d:
        n_trials = st.number_input("Number of trials", 100, 2000, 500, 100)
    seed = st.slider("Random seed", 1, 100, 42)
    with st.spinner("Generating synthetic trials..."):
        data = generate_ddm_dataset(
            v=v_true, a=a_true, t0=t0_true,
            n_trials=int(n_trials), seed=seed,
        )
else:
    uploaded = st.file_uploader(
        "Upload a CSV with columns: choice (0 or 1), rt (seconds)",
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
    validate_ddm_data(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.header("2. Data preview")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data.head(20), use_container_width=True)
with col2:
    st.markdown("**Summary**")
    st.markdown(f"- Total trials: {len(data)}")
    st.markdown(f"- Proportion choice = 1: {data['choice'].mean():.3f}")
    st.markdown(f"- Mean RT: {data['rt'].mean():.3f} s")
    st.markdown(f"- Median RT: {data['rt'].median():.3f} s")

st.header("3. Fit model")

st.caption(
    "Fitting uses maximum likelihood with the Navarro-Fuss (2009) "
    "first-passage time density. For 500 trials expect roughly 10-30 seconds."
)

if st.button("Fit DDM", type="primary"):
    with st.spinner("Fitting the model..."):
        try:
            model = DriftDiffusionModel().fit(data)
            st.session_state["ddm_model"] = model
            st.session_state["ddm_data"] = data
            st.success("Model fitted.")
        except Exception as exc:
            st.error(f"Fitting failed: {exc}")
            st.stop()

if "ddm_model" not in st.session_state:
    st.info("Fit the model to see results.")
    st.stop()

model = st.session_state["ddm_model"]
data = st.session_state["ddm_data"]

st.header("4. Fitted parameters")

summary = model.summary()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Drift rate (v)", f"{summary['drift_rate']:.3f}")
c2.metric("Boundary (a)", f"{summary['boundary_separation']:.3f}")
c3.metric("Non-decision time (t0)", f"{summary['non_decision_time']:.3f}")
c4.metric("Start point (z)", f"{summary['starting_point']:.3f}")

ll = model.log_likelihood(data)
bic = model.bic(data)
st.markdown(f"**Log-likelihood:** {ll:.2f}    |    **BIC:** {bic:.2f}")

st.header("5. Visualisations")

tab1, tab2, tab3 = st.tabs(["RT distributions", "Choice proportions", "Posterior predictive"])

with tab1:
    fig = px.histogram(
        data, x="rt", color=data["choice"].astype(str),
        nbins=40, barmode="overlay", opacity=0.6,
        title="Reaction time distribution by choice",
        labels={"color": "choice"},
    )
    fig.update_layout(xaxis_title="Reaction time (s)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    props = pd.DataFrame({
        "choice": ["0 (lower)", "1 (upper)"],
        "proportion": [1 - data["choice"].mean(), data["choice"].mean()],
    })
    fig = px.bar(props, x="choice", y="proportion",
                 title="Empirical choice proportions", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.caption(
        "Blue: observed data. Orange: data simulated from the fitted "
        "model parameters. Overlap indicates good fit."
    )
    simulated = DriftDiffusionModel.simulate(
        v=model.v, a=model.a, t0=model.t0,
        n_trials=len(data), seed=999,
    )
    combined_rt = pd.concat([
        data.assign(source="observed"),
        simulated.assign(source="simulated"),
    ], ignore_index=True)

    fig = px.histogram(
        combined_rt, x="rt", color="source",
        nbins=40, barmode="overlay", opacity=0.55,
        title="Observed vs. model-predicted RT distribution",
    )
    fig.update_layout(xaxis_title="Reaction time (s)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

st.header("6. Export")

summary_df = pd.DataFrame([summary])
csv_buffer = io.StringIO()
summary_df.to_csv(csv_buffer, index=False)
st.download_button(
    "Download fitted parameters (CSV)",
    csv_buffer.getvalue(),
    file_name="ddm_fit.csv",
    mime="text/csv",
)