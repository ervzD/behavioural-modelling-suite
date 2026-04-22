import numpy as np
import pandas as pd
from scipy.optimize import minimize


class DriftDiffusionModel:
    """
    Simple drift diffusion model for two-alternative forced choice tasks.

    Evidence is assumed to accumulate according to a Wiener process with drift
    v starting at z*a until it hits one of two absorbing boundaries at 0 or a.
    Reaction time is the first-passage time plus a non-decision component t0.

    Parameters
    ----------
    v : drift rate
    a : boundary separation
    t0 : non-decision time
    z : relative starting point bias (0 to 1, default 0.5 for unbiased)
    """

    def __init__(self, v=None, a=None, t0=None, z=0.5):
        self.v = v
        self.a = a
        self.t0 = t0
        self.z = z
        self.fitted = False

    @staticmethod
    def simulate(v, a, t0, z=0.5, n_trials=1000, dt=0.001, max_t=10.0, seed=None):
        """
        Simulate trials using Euler-Maruyama integration of the Wiener process.
        Returns a DataFrame with columns: choice (0 or 1) and rt (seconds).
        """
        rng = np.random.default_rng(seed)
        choices = np.zeros(n_trials, dtype=int)
        rts = np.zeros(n_trials)

        sqrt_dt = np.sqrt(dt)
        max_steps = int(max_t / dt)

        for i in range(n_trials):
            x = z * a
            for step in range(max_steps):
                x += v * dt + rng.normal(0, sqrt_dt)
                if x >= a:
                    choices[i] = 1
                    rts[i] = (step + 1) * dt + t0
                    break
                if x <= 0:
                    choices[i] = 0
                    rts[i] = (step + 1) * dt + t0
                    break
            else:
                choices[i] = 1 if x > z * a else 0
                rts[i] = max_t + t0

        return pd.DataFrame({'choice': choices, 'rt': rts})

    @staticmethod
    def _wiener_pdf(rt, v, a, z, err=1e-7):
        """
        First-passage time density at the lower boundary for a Wiener process,
        using the infinite-series approximation from Navarro & Fuss (2009).
        """
        if rt <= 0:
            return 1e-12

        tt = rt / (a ** 2)
        w = z

        if np.pi * tt * err < 1:
            kl = np.sqrt(-2 * np.log(np.pi * tt * err) / (np.pi ** 2 * tt))
            kl = max(kl, 1 / (np.pi * np.sqrt(tt)))
        else:
            kl = 1 / (np.pi * np.sqrt(tt))

        if 2 * np.sqrt(2 * np.pi * tt) * err < 1:
            ks = 2 + np.sqrt(-2 * tt * np.log(2 * np.sqrt(2 * np.pi * tt) * err))
            ks = max(ks, np.sqrt(tt) + 1)
        else:
            ks = 2

        if ks < kl:
            K = int(np.ceil(ks))
            lower = -int((K - 1) / 2)
            upper = int(K / 2)
            p = 0.0
            for k in range(lower, upper + 1):
                p += (w + 2 * k) * np.exp(-((w + 2 * k) ** 2) / (2 * tt))
            p /= np.sqrt(2 * np.pi * tt ** 3)
        else:
            K = int(np.ceil(kl))
            p = 0.0
            for k in range(1, K + 1):
                p += k * np.exp(-(k ** 2) * (np.pi ** 2) * tt / 2) * np.sin(k * np.pi * w)
            p *= np.pi

        p *= np.exp(-v * a * w - (v ** 2) * rt / 2) / (a ** 2)
        return max(p, 1e-12)

    @classmethod
    def _trial_likelihood(cls, rt, choice, v, a, t0, z):
        decision_time = rt - t0
        if decision_time <= 0:
            return 1e-12
        if choice == 1:
            return cls._wiener_pdf(decision_time, -v, a, 1 - z)
        else:
            return cls._wiener_pdf(decision_time, v, a, z)

    @classmethod
    def _neg_log_likelihood(cls, params, rts, choices):
        v, a, t0 = params
        if a <= 0.01 or t0 <= 0 or t0 > rts.min():
            return 1e10
        total = 0.0
        for rt, choice in zip(rts, choices):
            p = cls._trial_likelihood(rt, choice, v, a, t0, 0.5)
            total -= np.log(p)
        return total

    def fit(self, data):
        """
        Fit v, a, and t0 by maximum likelihood. Starting point bias z is fixed
        at 0.5 (unbiased). Expects a DataFrame with columns 'choice' (0 or 1)
        and 'rt' (in seconds).
        """
        required = {'choice', 'rt'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")

        rts = data['rt'].values
        choices = data['choice'].values

        if rts.min() <= 0:
            raise ValueError("All reaction times must be positive.")

        t0_init = max(0.1, rts.min() * 0.5)
        x0 = [0.5, 1.0, t0_init]
        bounds = [(-10, 10), (0.1, 5.0), (0.01, rts.min() * 0.95)]

        result = minimize(
            self._neg_log_likelihood, x0, args=(rts, choices),
            method='L-BFGS-B', bounds=bounds,
        )

        self.v, self.a, self.t0 = result.x
        self.z = 0.5
        self.fitted = True
        return self

    def log_likelihood(self, data):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before computing likelihood.")
        return -self._neg_log_likelihood(
            [self.v, self.a, self.t0], data['rt'].values, data['choice'].values
        )

    def bic(self, data):
        n = len(data)
        k = 3
        return k * np.log(n) - 2 * self.log_likelihood(data)

    def summary(self):
        if not self.fitted:
            return "Model not yet fitted."
        return {
            'drift_rate': self.v,
            'boundary_separation': self.a,
            'non_decision_time': self.t0,
            'starting_point': self.z,
        }