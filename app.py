import streamlit as st

st.set_page_config(
    page_title="Behavioural Modelling Suite",
    page_icon=None,
    layout="wide",
)

st.title("Behavioural Modelling Suite")

st.markdown(
    """
    A web-based platform for fitting and visualising computational models
    of human behaviour. Upload your experimental data, select a model, and
    examine whether simple theoretical accounts describe your observations.

    Use the sidebar on the left to navigate between models.
    """
)

st.header("Available models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bayesian cue integration")
    st.markdown(
        """
        Fits an optimal Bayesian model to multisensory (e.g. audio-visual)
        perception data. Estimates the noise of each unimodal channel and
        predicts how the two are combined.

        Suitable for datasets containing unimodal and bimodal trials where
        participants report a perceived stimulus value.

        Reference: Alais & Burr (2004), *Current Biology*.
        """
    )

with col2:
    st.subheader("Drift diffusion model")
    st.markdown(
        """
        Fits a two-alternative forced choice decision process to reaction
        time and accuracy data. Estimates drift rate, boundary separation,
        and non-decision time.

        Suitable for datasets of binary choices with reaction times.

        Reference: Ratcliff & McKoon (2008), *Neural Computation*.
        """
    )

st.header("Getting started")
st.markdown(
    """
    1. Select a model from the sidebar.
    2. Either upload your own CSV file or use the built-in example dataset.
    3. Fit the model and inspect the parameter estimates and diagnostic plots.
    4. Download the fitted results as a CSV for further analysis.

    Data is processed in-memory only and is not stored on any server.
    """
)

st.header("Data format requirements")
st.markdown(
    """
    Each model page describes the exact columns it expects. In short:

    - **Bayesian integration** expects `modality`, `stimulus`, `response`.
    - **Drift diffusion** expects `choice` (0 or 1) and `rt` (seconds).

    Files must be CSV, with a header row.
    """
)