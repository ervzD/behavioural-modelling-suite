import streamlit as st


st.set_page_config(page_title="About", layout="wide")

st.title("About the Behavioural Modelling Suite")

st.header("Purpose")
st.markdown(
    """
    The Behavioural Modelling Suite is a web-based tool for fitting simple
    computational models of human behaviour to experimental data. It is
    intended for researchers and students in behavioural science who want
    to test whether established theoretical accounts describe their data,
    but who may not have the programming background to implement the
    models themselves.

    The core philosophy is that many widely used behavioural models are
    mathematically simple, yet remain inaccessible because implementations
    are typically scattered across bespoke scripts in proprietary
    environments. By providing a standard interface, open-source code,
    and validation against published paradigms, the suite aims to make
    these analyses more reproducible and accessible.
    """
)

st.header("Implemented models")

st.subheader("Bayesian cue integration")
st.markdown(
    """
    Optimal combination of two noisy sensory estimates (e.g. visual and
    auditory) using inverse-variance weighting. Predicts the mean and
    variance of combined-modality responses given unimodal noise levels.

    Canonical reference: Alais, D. & Burr, D. (2004). The ventriloquist
    effect results from near-optimal bimodal integration. *Current
    Biology*, 14(3), 257-262.
    """
)

st.subheader("Drift diffusion model")
st.markdown(
    """
    Two-alternative forced choice decision-making modelled as noisy
    accumulation of evidence toward one of two decision boundaries.
    Fitted parameters include drift rate, boundary separation, and
    non-decision time. The likelihood uses the first-passage time density
    approximation of Navarro and Fuss (2009).

    Canonical reference: Ratcliff, R. & McKoon, G. (2008). The diffusion
    decision model: theory and data for two-choice decision tasks.
    *Neural Computation*, 20(4), 873-922.
    """
)

st.subheader("Kalman filter")
st.markdown(
    """
    One-dimensional linear-Gaussian state-space model for tracking a
    latent, drifting quantity from noisy observations. Estimates process
    noise (state drift) and observation noise by maximum likelihood.

    Canonical reference: Shadmehr, R. & Mussa-Ivaldi, F. (2012).
    *Biological Learning and Control*. MIT Press.
    """
)

st.header("Methodology")
st.markdown(
    """
    Parameter estimation uses maximum likelihood throughout. For the
    Bayesian model, closed-form estimators are used. For the drift
    diffusion and Kalman models, numerical optimisation is performed
    using SciPy (L-BFGS-B and Nelder-Mead respectively).

    Every implemented model is accompanied by automated tests that
    verify mathematical correctness and parameter recovery from
    synthetic data. See the Validation page for live recovery results.
    """
)

st.header("Data format")
st.markdown(
    """
    All data must be supplied as CSV files with a header row. Each model
    expects specific columns:

    - **Bayesian integration:** `modality` (values: visual, auditory,
      combined), `stimulus`, `response`
    - **Drift diffusion:** `choice` (0 or 1), `rt` (seconds, positive)
    - **Kalman filter:** `observation` (one row per time step)

    Example datasets are available on each model page by toggling the
    data source to "Use example dataset".
    """
)

st.header("Privacy")
st.markdown(
    """
    Uploaded files are processed in-memory only for the duration of the
    browser session. No data is persisted on any server. Users remain
    responsible for ensuring that any data they upload is suitably
    anonymised and complies with their institutional data governance
    policies, including GDPR where applicable.
    """
)

st.header("Source code and licence")
st.markdown(
    """
    The full source code, including the models, tests, and this web
    interface, is available on GitHub. The suite is released under the
    MIT Licence. Issues, pull requests, and extensions are welcome.
    """
)

st.header("References")
st.markdown(
    """
    - Alais, D. & Burr, D. (2004). The ventriloquist effect results from
      near-optimal bimodal integration. *Current Biology*, 14(3), 257-262.
    - Ernst, M. O. & Banks, M. S. (2002). Humans integrate visual and
      haptic information in a statistically optimal fashion. *Nature*,
      415(6870), 429-433.
    - Kalman, R. E. (1960). A new approach to linear filtering and
      prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
    - Knill, D. C. & Pouget, A. (2004). The Bayesian brain: the role of
      uncertainty in neural coding and computation. *Trends in
      Neurosciences*, 27(12), 712-719.
    - Navarro, D. J. & Fuss, I. G. (2009). Fast and accurate calculations
      for first-passage times in Wiener diffusion models. *Journal of
      Mathematical Psychology*, 53(4), 222-230.
    - Ratcliff, R. & McKoon, G. (2008). The diffusion decision model:
      theory and data for two-choice decision tasks. *Neural
      Computation*, 20(4), 873-922.
    - Shadmehr, R. & Mussa-Ivaldi, F. (2012). *Biological Learning and
      Control*. MIT Press.
    """
)