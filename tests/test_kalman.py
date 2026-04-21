import numpy as np
import pandas as pd
import pytest

from src.models.kalman import KalmanFilter


def test_simulate_produces_correct_shape():
    data = KalmanFilter.simulate(0.5, 1.0, n_steps=50, seed=1)
    assert len(data) == 50
    assert set(data.columns) == {"time", "true_state", "observation"}


def test_filter_reduces_noise():
    data = KalmanFilter.simulate(0.2, 2.0, n_steps=200, seed=2)
    kf = KalmanFilter(process_noise=0.2, observation_noise=2.0)
    filtered_means, _ = kf.filter(data["observation"].values)
    obs_error = np.std(data["observation"] - data["true_state"])
    filt_error = np.std(filtered_means - data["true_state"])
    assert filt_error < obs_error


def test_fit_recovers_parameters():
    rng_seed = 99
    data = KalmanFilter.simulate(0.5, 1.2, n_steps=500, seed=rng_seed)
    model = KalmanFilter().fit(data)
    assert abs(model.process_noise - 0.5) < 0.3
    assert abs(model.observation_noise - 1.2) < 0.3


def test_fit_requires_enough_data():
    with pytest.raises(ValueError):
        KalmanFilter().fit(pd.DataFrame({"observation": [1.0, 2.0]}))


def test_filter_without_params_raises():
    kf = KalmanFilter()
    with pytest.raises(RuntimeError):
        kf.filter([1.0, 2.0, 3.0])