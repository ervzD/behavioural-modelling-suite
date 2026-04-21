import numpy as np
import pandas as pd

from src.models.ddm import DriftDiffusionModel
from src.models.kalman import KalmanFilter


def generate_bayesian_dataset(
    n_trials_per_condition=200,
    sigma_visual=1.5,
    sigma_auditory=3.0,
    stimulus_range=(-10, 10),
    seed=None,
):
    """
    Generate a synthetic dataset suitable for fitting the Bayesian integration model.

    Produces trials in three conditions: visual-only, auditory-only, and combined.
    Responses are drawn from Gaussian distributions centred on the true stimulus
    with the specified noise levels. Combined-trial responses follow the optimal
    Bayesian combination rule.
    """
    rng = np.random.default_rng(seed)

    low, high = stimulus_range
    stimuli = rng.uniform(low, high, size=n_trials_per_condition)

    visual_response = stimuli + rng.normal(0, sigma_visual, size=n_trials_per_condition)
    auditory_response = stimuli + rng.normal(0, sigma_auditory, size=n_trials_per_condition)

    var_v = sigma_visual ** 2
    var_a = sigma_auditory ** 2
    w_v = var_a / (var_v + var_a)
    w_a = var_v / (var_v + var_a)
    sigma_combined = np.sqrt((var_v * var_a) / (var_v + var_a))

    s_v_combined = stimuli + rng.normal(0, sigma_visual, size=n_trials_per_condition)
    s_a_combined = stimuli + rng.normal(0, sigma_auditory, size=n_trials_per_condition)
    combined_mean = w_v * s_v_combined + w_a * s_a_combined
    combined_response = combined_mean + rng.normal(0, sigma_combined * 0.1, size=n_trials_per_condition)

    visual_df = pd.DataFrame({
        'modality': 'visual',
        'stimulus': stimuli,
        'response': visual_response,
    })
    auditory_df = pd.DataFrame({
        'modality': 'auditory',
        'stimulus': stimuli,
        'response': auditory_response,
    })
    combined_df = pd.DataFrame({
        'modality': 'combined',
        'stimulus': stimuli,
        'response': combined_response,
    })

    return pd.concat([visual_df, auditory_df, combined_df], ignore_index=True)


def generate_ddm_dataset(v=1.0, a=1.2, t0=0.25, n_trials=500, seed=None):
    """
    Generate synthetic two-alternative forced choice data following a DDM.
    """
    return DriftDiffusionModel.simulate(v=v, a=a, t0=t0, n_trials=n_trials, seed=seed)


def generate_kalman_dataset(process_noise=0.5, observation_noise=1.2,
                             n_steps=100, seed=None):
    """
    Generate synthetic tracking data following a random walk with noisy observations.
    """
    return KalmanFilter.simulate(
        process_noise=process_noise,
        observation_noise=observation_noise,
        n_steps=n_steps,
        seed=seed,
    )


if __name__ == '__main__':
    bayes_df = generate_bayesian_dataset(seed=42)
    bayes_df.to_csv('data/examples/bayesian_example.csv', index=False)
    print(f"Saved {len(bayes_df)} trials to data/examples/bayesian_example.csv")

    ddm_df = generate_ddm_dataset(seed=42)
    ddm_df.to_csv('data/examples/ddm_example.csv', index=False)
    print(f"Saved {len(ddm_df)} trials to data/examples/ddm_example.csv")

    kalman_df = generate_kalman_dataset(seed=42)
    kalman_df.to_csv('data/examples/kalman_example.csv', index=False)
    print(f"Saved {len(kalman_df)} observations to data/examples/kalman_example.csv")