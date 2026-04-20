import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class BayesianIntegration:
    """
    Optimal Bayesian cue combination for two sensory modalities.

    Given two noisy estimates of the same quantity (e.g. visual and auditory
    position), the optimal combined estimate is a weighted average where
    weights are proportional to the reliability (inverse variance) of each cue.

    Based on the framework used in Alais & Burr (2004) and Ernst & Banks (2002).
    """

    def __init__(self, sigma_visual=None, sigma_auditory=None):
        self.sigma_visual = sigma_visual
        self.sigma_auditory = sigma_auditory
        self.fitted = False

    @staticmethod
    def predict(s_visual, s_auditory, sigma_visual, sigma_auditory):
        """
        Predict the combined estimate from two unimodal estimates.

        Returns the combined estimate and its standard deviation.
        """
        var_v = sigma_visual ** 2
        var_a = sigma_auditory ** 2

        weight_v = var_a / (var_v + var_a)
        weight_a = var_v / (var_v + var_a)

        combined = weight_v * s_visual + weight_a * s_auditory
        sigma_combined = np.sqrt((var_v * var_a) / (var_v + var_a))

        return combined, sigma_combined

    @staticmethod
    def optimal_weights(sigma_visual, sigma_auditory):
        var_v = sigma_visual ** 2
        var_a = sigma_auditory ** 2
        w_v = var_a / (var_v + var_a)
        w_a = var_v / (var_v + var_a)
        return w_v, w_a

    def fit(self, data):
        """
        Fit sigma_visual and sigma_auditory from unimodal trial data.

        Expects a DataFrame with columns:
            modality:  'visual', 'auditory', or 'combined'
            stimulus:  true stimulus value on that trial
            response:  participant response on that trial
        """
        required = {'modality', 'stimulus', 'response'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")

        visual_trials = data[data['modality'] == 'visual']
        auditory_trials = data[data['modality'] == 'auditory']

        if len(visual_trials) < 2 or len(auditory_trials) < 2:
            raise ValueError(
                "Need at least 2 visual and 2 auditory trials to estimate noise."
            )

        visual_errors = visual_trials['response'] - visual_trials['stimulus']
        auditory_errors = auditory_trials['response'] - auditory_trials['stimulus']

        self.sigma_visual = float(np.std(visual_errors, ddof=1))
        self.sigma_auditory = float(np.std(auditory_errors, ddof=1))
        self.fitted = True

        return self

    def log_likelihood(self, data):
        """
        Log-likelihood of combined-modality responses under the fitted model.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before computing likelihood.")

        combined_trials = data[data['modality'] == 'combined'].copy()
        if len(combined_trials) == 0:
            return 0.0

        if 'stimulus_visual' in data.columns and 'stimulus_auditory' in data.columns:
            s_v = combined_trials['stimulus_visual'].values
            s_a = combined_trials['stimulus_auditory'].values
        else:
            s_v = combined_trials['stimulus'].values
            s_a = combined_trials['stimulus'].values

        predicted, sigma_pred = self.predict(
            s_v, s_a, self.sigma_visual, self.sigma_auditory
        )

        responses = combined_trials['response'].values
        ll = norm.logpdf(responses, loc=predicted, scale=sigma_pred).sum()
        return float(ll)

    def bic(self, data):
        """Bayesian Information Criterion. Lower is better."""
        n = len(data[data['modality'] == 'combined'])
        k = 2
        ll = self.log_likelihood(data)
        return k * np.log(n) - 2 * ll

    def summary(self):
        if not self.fitted:
            return "Model not yet fitted."
        w_v, w_a = self.optimal_weights(self.sigma_visual, self.sigma_auditory)
        return {
            'sigma_visual': self.sigma_visual,
            'sigma_auditory': self.sigma_auditory,
            'weight_visual': w_v,
            'weight_auditory': w_a,
        }