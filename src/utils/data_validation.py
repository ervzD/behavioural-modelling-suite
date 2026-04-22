"""
Shared validation routines for user-uploaded data. Each function raises a
ValueError with a clear, user-facing message if the data fails a check.
The Streamlit pages catch these and display the message to the user.
"""

import pandas as pd


def _require_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Uploaded data must be a table (CSV).")


def _require_columns(data, required):
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected: {', '.join(sorted(required))}."
        )


def _require_numeric(data, columns):
    for col in columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' must contain numeric values.")
        if data[col].isna().any():
            raise ValueError(f"Column '{col}' must not contain missing values.")


def validate_bayesian_data(data):
    """
    Check that uploaded data is suitable for the Bayesian integration model.
    """
    _require_dataframe(data)

    _require_columns(data, {"modality", "stimulus", "response"})

    if len(data) == 0:
        raise ValueError("The uploaded file contains no rows.")

    valid_modalities = {"visual", "auditory", "combined"}
    found = set(data["modality"].dropna().unique())
    unknown = found - valid_modalities
    if unknown:
        raise ValueError(
            f"Unrecognised modality labels: {sorted(unknown)}. "
            "Expected only: visual, auditory, combined."
        )

    for mod in ("visual", "auditory"):
        sub = data[data["modality"] == mod]
        if len(sub) < 2:
            raise ValueError(
                f"Need at least 2 '{mod}' trials to estimate its noise; "
                f"got {len(sub)}."
            )

    _require_numeric(data, ("stimulus", "response"))


def validate_ddm_data(data):
    """
    Check that uploaded data is suitable for the drift diffusion model.
    """
    _require_dataframe(data)

    _require_columns(data, {"choice", "rt"})

    if len(data) < 10:
        raise ValueError(
            f"Need at least 10 trials to fit reliably; got {len(data)}."
        )

    unique_choices = set(data["choice"].dropna().unique())
    if not unique_choices.issubset({0, 1}):
        raise ValueError(
            f"The 'choice' column must contain only 0 or 1; "
            f"found {sorted(unique_choices)}."
        )

    _require_numeric(data, ("choice", "rt"))

    if (data["rt"] <= 0).any():
        raise ValueError("All reaction times must be positive.")

    if data["rt"].max() > 60:
        raise ValueError(
            "Some reaction times exceed 60 seconds. "
            "Check that RT is in seconds, not milliseconds."
        )


def validate_kalman_data(data):
    """
    Check that uploaded data is suitable for the Kalman filter model.
    """
    _require_dataframe(data)

    _require_columns(data, {"observation"})

    if len(data) < 3:
        raise ValueError(
            f"Need at least 3 observations; got {len(data)}."
        )

    _require_numeric(data, ("observation",))