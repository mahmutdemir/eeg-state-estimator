"""Tests for the consistency diagnostics, against known distributions.

These deliberately do not involve the Kalman filter. A diagnostic verified only by
pointing it at the estimator it is meant to check can agree with a broken estimator and
silently bless it; the two have to be verified independently before being used together.

So Ljung-Box is calibrated on iid Gaussian noise, the chi-square bounds are checked
against `scipy.stats.chi2` directly, and coverage is checked on synthetic draws whose
coverage is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from eeg_state_estimator.statespace.diagnostics import (
    chi_square_consistency_bounds,
    credible_interval_coverage,
    ljung_box,
    nees_across_runs,
    nis_over_time,
    normalized_estimation_error_squared,
    normalized_innovation_squared,
)


# --------------------------------------------------------------------------- bounds


def test_chi_square_bounds_match_scipy() -> None:
    dof = 400
    lower, upper = chi_square_consistency_bounds(dof, confidence=0.95)
    assert np.isclose(lower, stats.chi2.ppf(0.025, dof) / dof)
    assert np.isclose(upper, stats.chi2.ppf(0.975, dof) / dof)
    assert lower < 1.0 < upper


def test_chi_square_bounds_tighten_as_degrees_of_freedom_grow() -> None:
    narrow = chi_square_consistency_bounds(1000)
    wide = chi_square_consistency_bounds(10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------- Ljung-Box


def test_ljung_box_does_not_reject_white_noise_more_often_than_its_alpha() -> None:
    """Calibration: on iid Gaussian input the rejection rate must sit near alpha = 0.05."""
    rng = np.random.default_rng(0)
    rejections = 0
    trials = 400
    for _ in range(trials):
        result = ljung_box(rng.normal(0.0, 1.0, 1000), n_lags=20)
        rejections += int(result.rejects_whiteness(alpha=0.05))
    rate = rejections / trials
    assert 0.02 < rate < 0.10, f"rejection rate {rate:.3f} is not near the nominal 0.05"


def test_ljung_box_rejects_a_strongly_autocorrelated_series() -> None:
    """Power: an AR(1) with phi = 0.6 is obviously not white and must be detected."""
    rng = np.random.default_rng(1)
    n = 1000
    innovations = rng.normal(0.0, 1.0, n)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = 0.6 * series[t - 1] + innovations[t]
    assert ljung_box(series, n_lags=20).rejects_whiteness(alpha=0.05)


def test_ljung_box_consumes_degrees_of_freedom_for_fitted_parameters() -> None:
    rng = np.random.default_rng(2)
    residuals = rng.normal(0.0, 1.0, 500)
    plain = ljung_box(residuals, n_lags=20, n_fitted_parameters=0)
    fitted = ljung_box(residuals, n_lags=20, n_fitted_parameters=2)
    assert plain.dof == 20
    assert fitted.dof == 18
    assert fitted.p_value < plain.p_value, "fewer dof must make the same statistic less likely"


def test_ljung_box_rejects_more_lags_than_samples() -> None:
    with pytest.raises(ValueError, match="n_lags"):
        ljung_box(np.zeros(10), n_lags=20)


# --------------------------------------------------------------------------- NIS / NEES


def test_normalized_squares_are_elementwise_ratios() -> None:
    innovation = np.array([1.0, 2.0, 3.0])
    variance = np.array([1.0, 4.0, 9.0])
    assert np.allclose(normalized_innovation_squared(innovation, variance), [1.0, 1.0, 1.0])

    truth = np.array([0.0, 0.0])
    mean = np.array([1.0, 2.0])
    posterior = np.array([1.0, 4.0])
    assert np.allclose(normalized_estimation_error_squared(truth, mean, posterior), [1.0, 1.0])


def test_nis_accepts_correctly_scaled_input() -> None:
    """Innovations drawn with exactly the variance the filter claims must be consistent."""
    rng = np.random.default_rng(3)
    n = 5000
    variance = np.full(n, 2.0)
    innovation = rng.normal(0.0, np.sqrt(2.0), n)
    assert nis_over_time(innovation, variance).is_consistent


def test_nis_rejects_input_whose_variance_is_understated() -> None:
    rng = np.random.default_rng(4)
    n = 5000
    claimed = np.full(n, 1.0)
    actual = rng.normal(0.0, np.sqrt(4.0), n)
    assert not nis_over_time(actual, claimed).is_consistent


def test_nees_across_runs_requires_a_time_index() -> None:
    truth = np.zeros((10, 5))
    with pytest.raises(TypeError):
        nees_across_runs(truth, truth, np.ones((10, 5)))  # type: ignore[call-arg]


def test_nees_across_runs_rejects_one_dimensional_input() -> None:
    """The 2-D signature is what prevents a time-average being computed by accident."""
    with pytest.raises(ValueError, match="two-dimensional"):
        nees_across_runs(np.zeros(5), np.zeros(5), np.ones(5), time_index=1)


def test_nees_accepts_correctly_scaled_draws() -> None:
    rng = np.random.default_rng(5)
    n_runs, n_samples = 400, 10
    variance = np.full((n_runs, n_samples), 3.0)
    truth = np.zeros((n_runs, n_samples))
    mean = rng.normal(0.0, np.sqrt(3.0), (n_runs, n_samples))
    assert nees_across_runs(truth, mean, variance, time_index=5).is_consistent


# --------------------------------------------------------------------------- coverage


def test_coverage_of_correctly_scaled_draws_is_near_nominal() -> None:
    rng = np.random.default_rng(6)
    n_runs, n_samples = 4000, 3
    variance = np.full((n_runs, n_samples), 1.0)
    truth = np.zeros((n_runs, n_samples))
    mean = rng.normal(0.0, 1.0, (n_runs, n_samples))
    result = credible_interval_coverage(truth, mean, variance, time_index=1)
    assert abs(result.fraction - 0.95) < 0.02
    assert result.n_runs == n_runs


def test_coverage_standard_error_is_the_binomial_one() -> None:
    rng = np.random.default_rng(7)
    n_runs = 500
    variance = np.ones((n_runs, 2))
    truth = np.zeros((n_runs, 2))
    mean = rng.normal(0.0, 1.0, (n_runs, 2))
    result = credible_interval_coverage(truth, mean, variance, time_index=1)
    expected = np.sqrt(0.95 * 0.05 / n_runs)
    assert np.isclose(result.standard_error, expected, rtol=0.4)
