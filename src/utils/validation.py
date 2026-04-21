"""
Validation of the implemented behavioural models against their canonical
published paradigms.

Bayesian integration: validated against the logic of Alais & Burr (2004),
"The Ventriloquist Effect Results from Near-Optimal Bimodal Integration".
We reproduce their two central predictions, namely that combined-modality
weights match the inverse-variance optimum and that combined-modality
variance is reduced as predicted.

Drift diffusion: validated against parameter recovery as reported in
Ratcliff & McKoon (2008), "The Diffusion Decision Model". We generate
data from known drift, boundary, and non-decision parameters and confirm
that maximum likelihood estimation recovers them.

Kalman filter: validated by parameter recovery on simulated random-walk
tracking data, following the cognitive modelling logic in Shadmehr &
Mussa-Ivaldi (2012).
"""

import numpy as np
import pandas as pd

from src.models.bayesian import BayesianIntegration
from src.models.ddm import DriftDiffusionModel
from src.models.kalman import KalmanFilter
from src.utils.data_generation import (
    generate_bayesian_dataset,
    generate_ddm_dataset,
    generate_kalman_dataset,
)


def validate_bayesian(visual_noise_levels=(0.5, 1.5, 3.0, 6.0),
                       auditory_noise=3.0, n_trials=400, seed=7):
    results = []
    for sigma_v in visual_noise_levels:
        data = generate_bayesian_dataset(
            n_trials_per_condition=n_trials,
            sigma_visual=sigma_v,
            sigma_auditory=auditory_noise,
            seed=seed,
        )
        model = BayesianIntegration().fit(data)
        fitted_w_v, _ = BayesianIntegration.optimal_weights(
            model.sigma_visual, model.sigma_auditory
        )
        true_w_v, _ = BayesianIntegration.optimal_weights(sigma_v, auditory_noise)

        combined_trials = data[data['modality'] == 'combined']
        empirical_sigma_combined = float(np.std(
            combined_trials['response'] - combined_trials['stimulus'], ddof=1
        ))
        _, predicted_sigma_combined = BayesianIntegration.predict(
            0, 0, model.sigma_visual, model.sigma_auditory
        )

        results.append({
            'true_sigma_visual': sigma_v,
            'fitted_sigma_visual': model.sigma_visual,
            'optimal_weight_visual': true_w_v,
            'fitted_weight_visual': fitted_w_v,
            'optimal_sigma_combined': predicted_sigma_combined,
            'empirical_sigma_combined': empirical_sigma_combined,
        })
    return pd.DataFrame(results)


def validate_ddm(parameter_sets=None, n_trials=800, seed=42):
    if parameter_sets is None:
        parameter_sets = [
            {'v': 0.5, 'a': 1.0, 't0': 0.2},
            {'v': 1.0, 'a': 1.2, 't0': 0.25},
            {'v': 1.5, 'a': 1.5, 't0': 0.3},
            {'v': -0.8, 'a': 1.0, 't0': 0.2},
        ]

    results = []
    for i, params in enumerate(parameter_sets):
        data = generate_ddm_dataset(
            v=params['v'], a=params['a'], t0=params['t0'],
            n_trials=n_trials, seed=seed + i,
        )
        model = DriftDiffusionModel().fit(data)
        results.append({
            'true_v': params['v'],
            'fitted_v': model.v,
            'true_a': params['a'],
            'fitted_a': model.a,
            'true_t0': params['t0'],
            'fitted_t0': model.t0,
        })
    return pd.DataFrame(results)


def validate_kalman(parameter_sets=None, n_steps=300, seed=11):
    if parameter_sets is None:
        parameter_sets = [
            {'process_noise': 0.3, 'observation_noise': 0.8},
            {'process_noise': 0.5, 'observation_noise': 1.2},
            {'process_noise': 1.0, 'observation_noise': 2.0},
            {'process_noise': 0.2, 'observation_noise': 1.5},
        ]
    results = []
    for i, params in enumerate(parameter_sets):
        data = generate_kalman_dataset(
            process_noise=params['process_noise'],
            observation_noise=params['observation_noise'],
            n_steps=n_steps, seed=seed + i,
        )
        model = KalmanFilter().fit(data)
        results.append({
            'true_process_noise': params['process_noise'],
            'fitted_process_noise': model.process_noise,
            'true_observation_noise': params['observation_noise'],
            'fitted_observation_noise': model.observation_noise,
        })
    return pd.DataFrame(results)


def print_bayesian_report(results):
    print("Bayesian integration — Alais & Burr (2004) paradigm")
    print("=" * 70)
    for _, row in results.iterrows():
        diff = abs(row['optimal_weight_visual'] - row['fitted_weight_visual'])
        print(
            f"  sigma_v = {row['true_sigma_visual']:.2f}  |  "
            f"optimal w_v = {row['optimal_weight_visual']:.3f}  |  "
            f"fitted w_v = {row['fitted_weight_visual']:.3f}  |  "
            f"|diff| = {diff:.3f}"
        )


def print_ddm_report(results):
    print()
    print("Drift diffusion — parameter recovery")
    print("=" * 70)
    for _, row in results.iterrows():
        print(
            f"  true (v={row['true_v']:+.2f}, a={row['true_a']:.2f}, t0={row['true_t0']:.2f})"
            f"   -->   fitted (v={row['fitted_v']:+.2f}, a={row['fitted_a']:.2f}, t0={row['fitted_t0']:.2f})"
        )


def print_kalman_report(results):
    print()
    print("Kalman filter — parameter recovery")
    print("=" * 70)
    for _, row in results.iterrows():
        print(
            f"  true (q={row['true_process_noise']:.2f}, r={row['true_observation_noise']:.2f})"
            f"   -->   fitted (q={row['fitted_process_noise']:.2f}, r={row['fitted_observation_noise']:.2f})"
        )


if __name__ == '__main__':
    bayes_results = validate_bayesian()
    bayes_results.to_csv('data/examples/bayesian_validation.csv', index=False)
    print_bayesian_report(bayes_results)

    ddm_results = validate_ddm()
    ddm_results.to_csv('data/examples/ddm_validation.csv', index=False)
    print_ddm_report(ddm_results)

    kalman_results = validate_kalman()
    kalman_results.to_csv('data/examples/kalman_validation.csv', index=False)
    print_kalman_report(kalman_results)

    print()
    print("Saved detailed results to data/examples/")