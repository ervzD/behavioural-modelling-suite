import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class KalmanFilter:
    """
    One-dimensional Kalman filter for tracking a latent state through time.

    The state evolves as
        x_{t+1} = x_t + w,    w ~ N(0, process_noise^2)
    and is observed as
        y_t = x_t + v,        v ~ N(0, observation_noise^2)

    Parameters
    ----------
    process_noise : standard deviation of the random walk on the latent state
    observation_noise : standard deviation of the observation noise
    initial_mean : prior mean on the latent state at t = 0
    initial_variance : prior variance on the latent state at t = 0
    """

    def __init__(self, process_noise=None, observation_noise=None,
                 initial_mean=0.0, initial_variance=1.0):
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_mean = initial_mean
        self.initial_variance = initial_variance
        self.fitted = False

    def filter(self, observations):
        """
        Run the forward pass. Returns filtered means and variances at each step.
        """
        if self.process_noise is None or self.observation_noise is None:
            raise RuntimeError("Parameters must be set before filtering.")

        n = len(observations)
        means = np.zeros(n)
        variances = np.zeros(n)

        mean = self.initial_mean
        variance = self.initial_variance
        q = self.process_noise ** 2
        r = self.observation_noise ** 2

        for t, y in enumerate(observations):
            predicted_mean = mean
            predicted_variance = variance + q

            kalman_gain = predicted_variance / (predicted_variance + r)
            mean = predicted_mean + kalman_gain * (y - predicted_mean)
            variance = (1 - kalman_gain) * predicted_variance

            means[t] = mean
            variances[t] = variance

        return means, variances

    @staticmethod
    def _neg_log_likelihood(params, observations, initial_mean, initial_variance):
        q_std, r_std = params
        if q_std <= 0 or r_std <= 0:
            return 1e10

        n = len(observations)
        mean = initial_mean
        variance = initial_variance
        q = q_std ** 2
        r = r_std ** 2
        total = 0.0

        for y in observations:
            predicted_mean = mean
            predicted_variance = variance + q
            innovation_variance = predicted_variance + r
            innovation = y - predicted_mean
            total += 0.5 * (
                np.log(2 * np.pi * innovation_variance)
                + (innovation ** 2) / innovation_variance
            )
            kalman_gain = predicted_variance / innovation_variance
            mean = predicted_mean + kalman_gain * innovation
            variance = (1 - kalman_gain) * predicted_variance

        return total

    def fit(self, data):
        """
        Fit process_noise and observation_noise by maximum likelihood.

        Expects a DataFrame with a single column 'observation' (one row per
        time step) or a 1-D array-like of observations.
        """
        if isinstance(data, pd.DataFrame):
            if "observation" not in data.columns:
                raise ValueError("Data must contain an 'observation' column.")
            observations = data["observation"].values
        else:
            observations = np.asarray(data)

        if len(observations) < 3:
            raise ValueError("Need at least 3 observations to fit.")

        x0 = [np.std(np.diff(observations)) + 0.1, np.std(observations) + 0.1]
        result = minimize(
            self._neg_log_likelihood, x0,
            args=(observations, self.initial_mean, self.initial_variance),
            method="Nelder-Mead",
        )
        self.process_noise, self.observation_noise = result.x
        self._neg_ll = result.fun
        self.fitted = True
        return self

    def log_likelihood(self, data):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before computing likelihood.")
        if isinstance(data, pd.DataFrame):
            observations = data["observation"].values
        else:
            observations = np.asarray(data)
        return -self._neg_log_likelihood(
            [self.process_noise, self.observation_noise],
            observations, self.initial_mean, self.initial_variance,
        )

    def bic(self, data):
        if isinstance(data, pd.DataFrame):
            n = len(data)
        else:
            n = len(data)
        k = 2
        return k * np.log(n) - 2 * self.log_likelihood(data)

    def summary(self):
        if not self.fitted:
            return "Model not yet fitted."
        return {
            "process_noise": self.process_noise,
            "observation_noise": self.observation_noise,
            "initial_mean": self.initial_mean,
            "initial_variance": self.initial_variance,
        }

    @staticmethod
    def simulate(process_noise, observation_noise, n_steps=100,
                 initial_state=0.0, seed=None):
        """
        Simulate a random-walk state plus observation noise. Returns a
        DataFrame with columns 'time', 'true_state', and 'observation'.
        """
        rng = np.random.default_rng(seed)
        states = np.zeros(n_steps)
        observations = np.zeros(n_steps)

        x = initial_state
        for t in range(n_steps):
            x += rng.normal(0, process_noise)
            states[t] = x
            observations[t] = x + rng.normal(0, observation_noise)

        return pd.DataFrame({
            "time": np.arange(n_steps),
            "true_state": states,
            "observation": observations,
        })