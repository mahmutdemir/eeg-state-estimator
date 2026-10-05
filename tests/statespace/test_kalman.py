"""Consistency tests for the scalar Kalman filter.

An estimator that is accurate on average but reports the wrong uncertainty is not usable
by anything that has to act on its own confidence. So accuracy is only the first of these
tests; the rest check that the posterior variance means what it claims.

Each diagnostic is also pointed at a deliberately broken filter and asserted to **fail**
(REQ-037). A diagnostic suite that only ever reports PASS is indistinguishable from a
suite of `assert True`.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_state_estimator.statespace.diagnostics import (
    credible_interval_coverage,
    ljung_box,
    nees_across_runs,
    nis_over_time,
)
from eeg_state_estimator.statespace.kalman import (
    RandomWalkModel,
    ScalarKalmanFilter,
    filter_series,
)

Q_TRUE, R_TRUE = 0.05, 1.0


def _simulate(
    *, n_runs: int, n_samples: int, seed: int, process_variance: float = Q_TRUE
) -> tuple[np.ndarray, np.ndarray]:
    """Independent random-walk trajectories and their noisy observations."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, np.sqrt(process_variance), (n_runs, n_samples))
    truth = np.cumsum(steps, axis=1)
    observations = truth + rng.normal(0.0, np.sqrt(R_TRUE), (n_runs, n_samples))
    return truth, observations


def _filter_runs(
    observations: np.ndarray, model: RandomWalkModel, *, initial_variance: float = 1.0e6
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty_like(observations)
    variances = np.empty_like(observations)
    for index, row in enumerate(observations):
        track = filter_series(row, model, initial_variance=initial_variance)
        means[index] = track.mean
        variances[index] = track.variance
    return means, variances


# --------------------------------------------------------------------------- REQ-030


@pytest.mark.req("REQ-030")
def test_posterior_variance_is_reported_and_strictly_positive() -> None:
    _, observations = _simulate(n_runs=1, n_samples=500, seed=0)
    track = filter_series(observations[0], RandomWalkModel(Q_TRUE, R_TRUE))
    assert track.variance.shape == observations[0].shape
    assert np.all(track.variance > 0.0)


@pytest.mark.req("REQ-030")
def test_zero_process_noise_reduces_to_the_recursive_weighted_mean() -> None:
    """With Q=0 the state cannot move, so the filter must become inverse-variance weighting.

    Closed-form oracle: P_k = 1 / (1/P0 + k/R). This is the test that would catch a sign
    error or a misplaced factor anywhere in the predict/update pair.
    """
    rng = np.random.default_rng(0)
    n, r, p0 = 40, 2.0, 1.0e6
    observations = 5.0 + rng.normal(0.0, np.sqrt(r), n)
    track = filter_series(
        observations, RandomWalkModel(0.0, r), initial_mean=0.0, initial_variance=p0
    )

    k = np.arange(1, n + 1)
    expected_variance = 1.0 / (1.0 / p0 + k / r)
    expected_mean = np.cumsum(observations) / r * expected_variance
    assert np.allclose(track.variance, expected_variance, rtol=1e-8)
    assert np.allclose(track.mean, expected_mean, rtol=1e-8)


@pytest.mark.req("REQ-030")
def test_posterior_variance_stays_positive_under_a_diffuse_prior() -> None:
    """The naive update P <- (1 - K*C)*P cancels catastrophically as K approaches 1.

    At P0 = 1e16 it returns 2.22 where the true answer is 2.00 -- an 11% error from pure
    floating-point cancellation, and at larger P0 it can go negative. The implementation
    uses P * R / S, which is the same algebra with no subtraction in it.
    """
    r = 2.0
    model = RandomWalkModel(0.0, r)
    for initial_variance in (1.0e6, 1.0e12, 1.0e16, 1.0e20):
        filt = ScalarKalmanFilter(model, initial_mean=0.0, initial_variance=initial_variance)
        step = filt.step(1.0)
        # Exact closed form. Note this is not quite R: it approaches R only as P0 -> inf,
        # so comparing against the limit would conflate a real difference with an error.
        exact = initial_variance * r / (initial_variance + r)
        assert step.variance > 0.0
        assert np.isclose(step.variance, exact, rtol=1e-12), f"P0={initial_variance:g}"


@pytest.mark.req("REQ-030")
def test_steady_state_variance_matches_the_riccati_fixed_point() -> None:
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    _, observations = _simulate(n_runs=1, n_samples=4000, seed=1)
    track = filter_series(observations[0], model, initial_variance=1.0)
    assert np.isclose(track.variance[-1], model.steady_state_posterior_variance(), rtol=1e-6)


# --------------------------------------------------------------------------- REQ-031


@pytest.mark.req("REQ-031")
def test_innovation_and_its_variance_are_exposed() -> None:
    """They are not debug output -- they are the only observables on which correctness can
    be assessed without ground truth, which is the situation on any real recording."""
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    filt = ScalarKalmanFilter(model, initial_mean=0.0, initial_variance=1.0)
    step = filt.step(0.7)

    expected_innovation = 0.7 - 0.0
    expected_s = 1.0 * (1.0 + Q_TRUE) + R_TRUE
    assert np.isclose(step.innovation, expected_innovation)
    assert np.isclose(step.innovation_variance, expected_s)
    assert np.isclose(step.normalized_innovation, expected_innovation / np.sqrt(expected_s))


# --------------------------------------------------------------------------- REQ-032


@pytest.mark.req("REQ-032")
def test_self_simulation_recovery_beats_the_raw_observations() -> None:
    truth, observations = _simulate(n_runs=1, n_samples=4000, seed=2)
    track = filter_series(observations[0], RandomWalkModel(Q_TRUE, R_TRUE))
    burn = 100
    filtered_rmse = float(np.sqrt(np.mean((track.mean[burn:] - truth[0][burn:]) ** 2)))
    raw_rmse = float(np.sqrt(np.mean((observations[0][burn:] - truth[0][burn:]) ** 2)))
    assert filtered_rmse < raw_rmse / 2.0


# --------------------------------------------------------------------------- REQ-033


@pytest.mark.req("REQ-033")
def test_credible_intervals_are_calibrated() -> None:
    """Across 500 runs, nominal 95% intervals must contain the truth 95% +/- 3%.

    Coverage is measured across independent runs at a fixed time index, not pooled over
    time within a run: the coverage indicators along one run are autocorrelated for the
    same reason NEES is, so pooling would understate the uncertainty and the bound would
    be mis-calibrated.
    """
    truth, observations = _simulate(n_runs=500, n_samples=320, seed=3)
    means, variances = _filter_runs(observations, RandomWalkModel(Q_TRUE, R_TRUE))
    result = credible_interval_coverage(truth, means, variances, time_index=300)
    assert abs(result.fraction - 0.95) <= 0.03, (
        f"coverage {result.fraction:.4f}, "
        f"{abs(result.fraction - 0.95) / result.standard_error:.1f} standard errors from nominal"
    )


# --------------------------------------------------------------------------- REQ-034


@pytest.mark.req("REQ-034")
def test_innovations_are_white_under_a_correct_model() -> None:
    """Ljung-Box with p = 0 fitted parameters: Q and R are given here, not estimated from
    the data, so no degrees of freedom are consumed. Getting that wrong is the most common
    error in applying this test."""
    _, observations = _simulate(n_runs=1, n_samples=4000, seed=4)
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    track = filter_series(
        observations[0], model, initial_variance=model.steady_state_posterior_variance()
    )
    result = ljung_box(track.normalized_innovation[200:], n_lags=20)
    assert not result.rejects_whiteness(alpha=0.05), f"p = {result.p_value:.4f}"


@pytest.mark.req("REQ-034")
def test_nis_is_consistent_over_time() -> None:
    """NIS may be averaged over time: innovations are white by construction under a
    correct model, so the samples are independent. NEES may not -- see REQ-035."""
    _, observations = _simulate(n_runs=1, n_samples=5000, seed=5)
    track = filter_series(observations[0], RandomWalkModel(Q_TRUE, R_TRUE))
    result = nis_over_time(track.innovation, track.innovation_variance, burn_in=200)
    assert result.is_consistent, f"NIS = {result.statistic:.4f}, bounds {result.bounds}"


# --------------------------------------------------------------------------- REQ-035


@pytest.mark.req("REQ-035")
def test_nees_is_consistent_across_runs_at_a_fixed_time() -> None:
    """NEES must be averaged across independent runs, not over time within one run.

    The estimation-error sequence is strongly autocorrelated -- the filter carries error
    forward through the state -- so time-averaged NEES has an effective sample size far
    below its sample count, and a chi-square test against the nominal bounds passes only
    about 65% of the time on a *correct* filter. That is a test which passes on the seed
    it was written with and flakes afterwards.
    """
    truth, observations = _simulate(n_runs=400, n_samples=320, seed=6)
    means, variances = _filter_runs(observations, RandomWalkModel(Q_TRUE, R_TRUE))
    result = nees_across_runs(truth, means, variances, time_index=300)
    assert result.is_consistent, f"NEES = {result.statistic:.4f}, bounds {result.bounds}"


@pytest.mark.req("REQ-035")
def test_estimation_error_is_autocorrelated_but_innovations_are_not() -> None:
    """The measurement that justifies the aggregation rule above, pinned as a test."""
    truth, observations = _simulate(n_runs=1, n_samples=5000, seed=7)
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    track = filter_series(observations[0], model)

    error = (truth[0] - track.mean)[200:]
    innovation = track.normalized_innovation[200:]
    error_lag1 = float(np.corrcoef(error[:-1], error[1:])[0, 1])
    innovation_lag1 = float(np.corrcoef(innovation[:-1], innovation[1:])[0, 1])

    assert error_lag1 > 0.5, "estimation error must be strongly autocorrelated"
    assert abs(innovation_lag1) < 0.05, "innovations must be white"


# --------------------------------------------------------------------------- REQ-036


@pytest.mark.req("REQ-036")
def test_converges_from_a_deliberately_poor_initialization() -> None:
    truth, observations = _simulate(n_runs=1, n_samples=600, seed=8)
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    track = filter_series(observations[0], model, initial_mean=100.0, initial_variance=1.0)

    within = np.abs(track.mean - truth[0]) <= 3.0 * np.sqrt(track.variance)
    assert np.all(within[100:]), "must be inside 3 posterior sd by 100 steps and stay there"


@pytest.mark.req("REQ-036")
@pytest.mark.parametrize("initial_variance", [0.01, 1.0, 1.0e6])
def test_steady_state_variance_does_not_depend_on_its_initial_value(
    initial_variance: float,
) -> None:
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    _, observations = _simulate(n_runs=1, n_samples=4000, seed=9)
    track = filter_series(observations[0], model, initial_variance=initial_variance)
    assert np.isclose(track.variance[-1], model.steady_state_posterior_variance(), rtol=1e-6)


# --------------------------------------------------------------------------- REQ-037


@pytest.mark.req("REQ-037")
@pytest.mark.parametrize(
    ("factor", "label"),
    [(0.1, "process variance 10x too small"), (10.0, "process variance 10x too large")],
)
def test_ljung_box_rejects_a_misspecified_process_variance(factor: float, label: str) -> None:
    """Negative control. The filter is knowably wrong; the diagnostic must say so."""
    _, observations = _simulate(n_runs=1, n_samples=4000, seed=10)
    wrong = RandomWalkModel(Q_TRUE * factor, R_TRUE)
    track = filter_series(observations[0], wrong)
    result = ljung_box(track.normalized_innovation[200:], n_lags=20)
    assert result.rejects_whiteness(alpha=0.05), f"{label}: p = {result.p_value:.4f}"


@pytest.mark.req("REQ-037")
def test_nis_rejects_a_misscaled_observation_variance() -> None:
    """Negative control. R 10x too large makes the filter over-confident about its
    innovations, which NIS detects even though the trajectory still looks plausible."""
    _, observations = _simulate(n_runs=1, n_samples=5000, seed=11)
    wrong = RandomWalkModel(Q_TRUE, R_TRUE * 10.0)
    track = filter_series(observations[0], wrong)
    result = nis_over_time(track.innovation, track.innovation_variance, burn_in=200)
    assert not result.is_consistent, f"NIS = {result.statistic:.4f} should be outside the bounds"


@pytest.mark.req("REQ-037")
def test_coverage_detects_an_overconfident_filter() -> None:
    """Negative control. Q 10x too small shrinks the credible intervals, so the truth
    falls outside them far more often than 5% of the time."""
    truth, observations = _simulate(n_runs=400, n_samples=320, seed=12)
    means, variances = _filter_runs(observations, RandomWalkModel(Q_TRUE * 0.1, R_TRUE))
    result = credible_interval_coverage(truth, means, variances, time_index=300)
    assert result.fraction < 0.90, f"coverage {result.fraction:.4f} should collapse"


# --------------------------------------------------------------------------- API shape


@pytest.mark.req("REQ-030")
def test_state_size_is_constant_in_record_length() -> None:
    """Bounded memory. The state is three scalars however long the recording runs."""
    filt = ScalarKalmanFilter(RandomWalkModel(Q_TRUE, R_TRUE))
    before = filt.state_nbytes
    filt.update(np.zeros(100_000))
    assert filt.state_nbytes == before


@pytest.mark.req("REQ-002")
def test_batch_and_streaming_paths_agree_exactly() -> None:
    """filter_series must contain no filtering arithmetic of its own, so there is only one
    implementation and the two paths cannot drift apart."""
    _, observations = _simulate(n_runs=1, n_samples=500, seed=13)
    model = RandomWalkModel(Q_TRUE, R_TRUE)

    batch = filter_series(observations[0], model)
    filt = ScalarKalmanFilter(model)
    stepwise = np.array([filt.step(y).mean for y in observations[0]])
    assert np.array_equal(batch.mean, stepwise)


@pytest.mark.req("REQ-002")
def test_reset_returns_the_filter_to_its_initial_state() -> None:
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    filt = ScalarKalmanFilter(model, initial_mean=0.5, initial_variance=2.0)
    first = filt.step(1.0)
    filt.update(np.arange(50, dtype=float))
    filt.reset()
    assert filt.n_updates == 0
    again = filt.step(1.0)
    assert again == first


@pytest.mark.req("REQ-001")
@pytest.mark.parametrize(
    ("process_variance", "observation_variance", "match"),
    [(-1.0, 1.0, "process_variance"), (1.0, 0.0, "observation_variance")],
)
def test_model_rejects_invalid_variances(
    process_variance: float, observation_variance: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RandomWalkModel(process_variance, observation_variance)
