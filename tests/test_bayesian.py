import numpy as np
import pandas as pd
import pytest

from src.models.bayesian import BayesianIntegration


def test_equal_reliability_gives_equal_weights():
    w_v, w_a = BayesianIntegration.optimal_weights(1.0, 1.0)
    assert np.isclose(w_v, 0.5)
    assert np.isclose(w_a, 0.5)


def test_more_reliable_cue_dominates():
    w_v, w_a = BayesianIntegration.optimal_weights(0.5, 5.0)
    assert w_v > w_a
    assert np.isclose(w_v + w_a, 1.0)


def test_combined_variance_smaller_than_each():
    _, sigma_combined = BayesianIntegration.predict(0, 0, 2.0, 3.0)
    assert sigma_combined < 2.0
    assert sigma_combined < 3.0


def test_fit_recovers_sigmas():
    rng = np.random.default_rng(42)
    n = 500
    true_sigma_v = 2.0
    true_sigma_a = 4.0

    stimulus = rng.uniform(-10, 10, size=n)
    visual_response = stimulus + rng.normal(0, true_sigma_v, size=n)
    auditory_response = stimulus + rng.normal(0, true_sigma_a, size=n)

    df = pd.DataFrame({
        'modality': ['visual'] * n + ['auditory'] * n,
        'stimulus': np.concatenate([stimulus, stimulus]),
        'response': np.concatenate([visual_response, auditory_response]),
    })

    model = BayesianIntegration().fit(df)
    assert abs(model.sigma_visual - true_sigma_v) < 0.3
    assert abs(model.sigma_auditory - true_sigma_a) < 0.5


def test_missing_columns_raises():
    df = pd.DataFrame({'modality': ['visual'], 'response': [1.0]})
    with pytest.raises(ValueError):
        BayesianIntegration().fit(df)