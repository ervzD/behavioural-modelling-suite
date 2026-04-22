import pandas as pd
import pytest

from src.utils.data_validation import (
    validate_bayesian_data,
    validate_ddm_data,
    validate_kalman_data,
)


def test_bayesian_accepts_valid_data():
    df = pd.DataFrame({
        "modality": ["visual"] * 3 + ["auditory"] * 3,
        "stimulus": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "response": [1.1, 2.1, 2.9, 0.9, 2.0, 3.2],
    })
    validate_bayesian_data(df)


def test_bayesian_rejects_missing_columns():
    df = pd.DataFrame({"modality": ["visual"], "response": [1.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_bayesian_data(df)


def test_bayesian_rejects_unknown_modality():
    df = pd.DataFrame({
        "modality": ["visual", "visual", "tactile", "tactile"],
        "stimulus": [1.0, 2.0, 1.0, 2.0],
        "response": [1.1, 2.1, 1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Unrecognised modality"):
        validate_bayesian_data(df)


def test_ddm_accepts_valid_data():
    df = pd.DataFrame({
        "choice": [0, 1] * 10,
        "rt": [0.5, 0.6] * 10,
    })
    validate_ddm_data(df)


def test_ddm_rejects_ms_scale_rts():
    df = pd.DataFrame({
        "choice": [0, 1] * 10,
        "rt": [500, 600] * 10,
    })
    with pytest.raises(ValueError, match="seconds"):
        validate_ddm_data(df)


def test_ddm_rejects_non_binary_choice():
    df = pd.DataFrame({
        "choice": [0, 1, 2] * 5,
        "rt": [0.5, 0.6, 0.7] * 5,
    })
    with pytest.raises(ValueError, match="0 or 1"):
        validate_ddm_data(df)


def test_kalman_accepts_valid_data():
    df = pd.DataFrame({"observation": [1.0, 2.0, 3.0, 4.0]})
    validate_kalman_data(df)


def test_kalman_rejects_missing_column():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="observation"):
        validate_kalman_data(df)


def test_kalman_rejects_too_few_observations():
    df = pd.DataFrame({"observation": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 3"):
        validate_kalman_data(df)