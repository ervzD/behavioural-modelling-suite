# Behavioural Modelling Suite

A web-based platform for fitting and visualising computational models of human behaviour.

The suite provides an accessible interface for researchers and students to apply established behavioural models to their own experimental data, without requiring specialist programming skills. The goal is to support open, reproducible behavioural science by making standard modelling analyses more transparent and widely usable.

## Implemented models

- **Bayesian cue integration** for multisensory perception (Alais & Burr, 2004)
- **Drift diffusion model** for two-alternative choice tasks with reaction times (Ratcliff & McKoon, 2008)
- **Kalman filter** for latent-state tracking in time-series data

Every model is accompanied by automated tests and live parameter-recovery validation against the published literature.

## Quick start

```bash
git clone https://github.com/ervzD/behavioural-modelling-suite.git
cd behavioural-modelling-suite
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Running the tests

```bash
pytest -v
```

## Project structure

```
behavioural-modelling-suite/
    app.py                    Streamlit landing page
    pages/                    Individual model pages
    src/
        models/               Model implementations
        utils/                Data generation and validation
    tests/                    Pytest suite
    data/examples/            Example datasets
```

## Data format

Each model page documents the columns it expects. In summary:

| Model | Required columns |
|-------|------------------|
| Bayesian integration | `modality`, `stimulus`, `response` |
| Drift diffusion | `choice` (0 or 1), `rt` (seconds) |
| Kalman filter | `observation` |

CSV files must include a header row.

## Licence

Released under the MIT Licence. See `LICENSE` for the full text.