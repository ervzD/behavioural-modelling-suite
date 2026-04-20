import numpy as np
import pandas as pd
import pytest

from src.models.ddm import DriftDiffusionModel


def test_simulate_returns_correct_shape():
    data = DriftDiffusionModel.simulate(v=1.0, a=1.5, t0=0.2, n_trials=100, seed=1)
    assert len(data) == 100
    assert set(data.columns) == {'choice', 'rt'}
    assert data['choice'].isin([0, 1]).all()
    assert (data['rt'] > 0).all()


def test_positive_drift_prefers_upper_boundary():
    data = DriftDiffusionModel.simulate(v=2.0, a=1.5, t0=0.2, n_trials=500, seed=2)
    assert data['choice'].mean() > 0.75


def test_negative_drift_prefers_lower_boundary():
    data = DriftDiffusionModel.simulate(v=-2.0, a=1.5, t0=0.2, n_trials=500, seed=3)
    assert data['choice'].mean() < 0.25


def test_larger_boundary_slower_rt():
    small = DriftDiffusionModel.simulate(v=1.0, a=0.8, t0=0.2, n_trials=300, seed=4)
    large = DriftDiffusionModel.simulate(v=1.0, a=2.0, t0=0.2, n_trials=300, seed=5)
    assert large['rt'].mean() > small['rt'].mean()


def test_fit_recovers_parameters():
    true_v, true_a, true_t0 = 1.0, 1.2, 0.25
    data = DriftDiffusionModel.simulate(
        v=true_v, a=true_a, t0=true_t0, n_trials=800, seed=42
    )
    model = DriftDiffusionModel().fit(data)
    assert abs(model.v - true_v) < 0.3
    assert abs(model.a - true_a) < 0.3
    assert abs(model.t0 - true_t0) < 0.1


def test_missing_columns_raises():
    df = pd.DataFrame({'rt': [0.5, 0.6]})
    with pytest.raises(ValueError):
        DriftDiffusionModel().fit(df)