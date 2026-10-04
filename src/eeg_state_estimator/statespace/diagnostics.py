"""Consistency diagnostics for a recursive estimator.

Implements REQ-033, REQ-034, REQ-035, REQ-037.

An estimator can track accurately while reporting a posterior variance that is wrong.
These check the second thing. Each one fails on a different kind of error, which is why
there are four rather than one:

| diagnostic | needs the true state | averaged over | usable on a real recording |
|---|---|---|---|
| NIS        | no  | time | **yes** |
| Ljung-Box  | no  | time | **yes** |
| NEES       | YES | runs | no — simulation only |
| coverage   | YES | runs | no — simulation only |

**The aggregation column is not a stylistic choice.** Innovations are white by
construction under a correctly specified model, so NIS values along one run are
independent and may be averaged over time. The estimation *error* sequence is not — the
filter carries error forward through the state, giving a lag-1 autocorrelation around
0.68 — so NEES values along one run are strongly dependent, the effective sample size is
far below the sample count, and a chi-square test against the nominal bounds is
mis-calibrated. Measured, a time-averaged NEES test sits inside its nominal 95% bounds
only about 65% of the time on a *correct* filter: it passes on the seed it was written
with and flakes afterwards.

The simulation-only functions take two-dimensional `(n_runs, n_samples)` arrays and a
required keyword-only `time_index`, so it is not possible to call them with a single run
or to average over time by accident. `truth` is their first positional parameter and has
no default, so neither can be reached without ground truth in hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from eeg_state_estimator.types import FloatArray

# Grouped by whether the diagnostic needs ground truth, not alphabetically. That split
# is the thing a reader has to get right, so it is worth more here than sort order.
__all__ = [  # noqa: RUF022
    # Field-computable: need only what the filter itself produces.
    "ChiSquareConsistency",
    "CoverageResult",
    "LjungBoxResult",
    "chi_square_consistency_bounds",
    "ljung_box",
    "nis_over_time",
    "normalized_innovation_squared",
    # Simulation only: require the true state.
    "credible_interval_coverage",
    "nees_across_runs",
    "normalized_estimation_error_squared",
]


@dataclass(frozen=True, slots=True)
class ChiSquareConsistency:
    """A normalised mean-square statistic against its chi-square acceptance bounds."""

    statistic: float
    dof: int
    lower: float
    upper: float
    confidence: float

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    @property
    def is_consistent(self) -> bool:
        return self.lower <= self.statistic <= self.upper


def chi_square_consistency_bounds(dof: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance bounds for a mean of `dof` independent chi-square(1) values.

    The mean of `n` iid chi-square(1) variables is chi-square(n)/n, so the bounds are the
    chi-square quantiles divided by `n`. For a `d`-dimensional state the per-sample
    degrees of freedom would be `d` rather than 1; the scalar case here is that rule with
    `d = 1`, not a coincidence.
    """
    if dof < 1:
        message = f"dof must be at least 1, got {dof}"
        raise ValueError(message)
    if not 0.0 < confidence < 1.0:
        message = f"confidence must be strictly between 0 and 1, got {confidence}"
        raise ValueError(message)
    tail = (1.0 - confidence) / 2.0
    lower = float(stats.chi2.ppf(tail, dof)) / dof
    upper = float(stats.chi2.ppf(1.0 - tail, dof)) / dof
    return lower, upper


# --------------------------------------------------------------------------------------
# Field-computable diagnostics
# --------------------------------------------------------------------------------------


def normalized_innovation_squared(innovation: object, innovation_variance: object) -> FloatArray:
    """`nu**2 / S`, elementwise. Each value is chi-square(1) under a correct model."""
    nu = np.asarray(innovation, dtype=np.float64)
    s = np.asarray(innovation_variance, dtype=np.float64)
    if nu.shape != s.shape:
        message = f"shape mismatch: innovation {nu.shape} vs variance {s.shape}"
        raise ValueError(message)
    if np.any(s <= 0.0):
        message = "innovation_variance must be strictly positive"
        raise ValueError(message)
    return np.asarray(nu**2 / s, dtype=np.float64)


def nis_over_time(
    innovation: object,
    innovation_variance: object,
    *,
    burn_in: int = 0,
    confidence: float = 0.95,
) -> ChiSquareConsistency:
    """Mean NIS over time within one run, against its chi-square bounds.

    Averaging over time is legitimate here because innovations are white under a correct
    model. This is the consistency check that works on a real recording, where the true
    state is unavailable.
    """
    values = normalized_innovation_squared(innovation, innovation_variance)[burn_in:]
    if values.size < 1:
        message = f"burn_in={burn_in} leaves no samples"
        raise ValueError(message)
    dof = int(values.size)
    lower, upper = chi_square_consistency_bounds(dof, confidence)
    return ChiSquareConsistency(
        statistic=float(np.mean(values)),
        dof=dof,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


@dataclass(frozen=True, slots=True)
class LjungBoxResult:
    """Ljung-Box portmanteau test for autocorrelation in a residual series."""

    statistic: float
    dof: int
    p_value: float
    n_lags: int

    def rejects_whiteness(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def ljung_box(residuals: object, *, n_lags: int, n_fitted_parameters: int = 0) -> LjungBoxResult:
    """Ljung-Box test, implemented directly rather than via a statistics dependency.

        r_k = sum_t (z_t - z_bar)(z_{t+k} - z_bar) / sum_t (z_t - z_bar)**2
        Q(h) = n (n + 2) * sum_{k=1..h} r_k**2 / (n - k)
        Q(h) ~ chi-square(h - p)

    The `n - k` denominator is the Ljung-Box refinement over Box-Pierce; it corrects the
    small-sample bias and is why the test is well calibrated at h = 20.

    `n_fitted_parameters` is `p`, the number of parameters estimated *from the data*. For
    this package it is 0: Q and R are given, not fitted, so no degrees of freedom are
    consumed. Passing a non-zero `p` when nothing was fitted is the most common error in
    applying this test, and it makes the test conservative in the wrong direction.
    """
    z = np.asarray(residuals, dtype=np.float64)
    if z.ndim != 1:
        message = f"residuals must be one-dimensional, got shape {z.shape}"
        raise ValueError(message)
    n = int(z.size)
    if n_lags < 1:
        message = f"n_lags must be at least 1, got {n_lags}"
        raise ValueError(message)
    if n_lags >= n:
        message = f"n_lags={n_lags} must be fewer than the {n} samples available"
        raise ValueError(message)
    dof = n_lags - n_fitted_parameters
    if dof < 1:
        message = (
            f"n_lags={n_lags} minus n_fitted_parameters={n_fitted_parameters} "
            f"leaves {dof} degrees of freedom"
        )
        raise ValueError(message)

    centred = z - z.mean()
    denominator = float(np.sum(centred**2))
    if denominator <= 0.0:
        message = "residuals have zero variance; the test is undefined"
        raise ValueError(message)

    total = 0.0
    for lag in range(1, n_lags + 1):
        autocorrelation = float(np.sum(centred[:-lag] * centred[lag:])) / denominator
        total += autocorrelation**2 / (n - lag)

    statistic = n * (n + 2) * total
    p_value = float(stats.chi2.sf(statistic, dof))
    return LjungBoxResult(statistic=statistic, dof=dof, p_value=p_value, n_lags=n_lags)


# --------------------------------------------------------------------------------------
# Simulation-only diagnostics -- these require the true state
# --------------------------------------------------------------------------------------


def normalized_estimation_error_squared(
    truth: object, mean: object, variance: object
) -> FloatArray:
    """`(x - x_hat)**2 / P`, elementwise. Chi-square(1) under a correct model.

    Requires the true state, so it is computable only in simulation. The real-data
    analogue is `normalized_innovation_squared`.
    """
    x = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    p = np.asarray(variance, dtype=np.float64)
    if not (x.shape == m.shape == p.shape):
        message = f"shape mismatch: truth {x.shape}, mean {m.shape}, variance {p.shape}"
        raise ValueError(message)
    if np.any(p <= 0.0):
        message = "variance must be strictly positive"
        raise ValueError(message)
    return np.asarray((x - m) ** 2 / p, dtype=np.float64)


def _require_runs_by_samples(name: str, values: FloatArray) -> None:
    if values.ndim != 2:
        message = (
            f"{name} must be two-dimensional (n_runs, n_samples), got shape {values.shape}. "
            f"NEES and coverage are averaged across independent runs at one time index, "
            f"not over time within a run -- see the module docstring."
        )
        raise ValueError(message)


def nees_across_runs(
    truth: object,
    mean: object,
    variance: object,
    *,
    time_index: int,
    confidence: float = 0.95,
) -> ChiSquareConsistency:
    """Mean NEES across independent runs at one time index, against chi-square bounds.

    Simulation only. The two-dimensional signature and the required `time_index` exist to
    make the correct aggregation the only one that is expressible — see the module
    docstring for why a time average is mis-calibrated.
    """
    x = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    p = np.asarray(variance, dtype=np.float64)
    for name, values in (("truth", x), ("mean", m), ("variance", p)):
        _require_runs_by_samples(name, values)

    values = normalized_estimation_error_squared(
        x[:, time_index], m[:, time_index], p[:, time_index]
    )
    dof = int(values.size)
    lower, upper = chi_square_consistency_bounds(dof, confidence)
    return ChiSquareConsistency(
        statistic=float(np.mean(values)),
        dof=dof,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Empirical credible-interval coverage across independent runs."""

    n_covered: int
    n_runs: int
    nominal: float

    @property
    def fraction(self) -> float:
        return self.n_covered / self.n_runs

    @property
    def standard_error(self) -> float:
        """Binomial standard error at the nominal rate, for an auditable margin."""
        return math.sqrt(self.nominal * (1.0 - self.nominal) / self.n_runs)


def credible_interval_coverage(
    truth: object,
    mean: object,
    variance: object,
    *,
    time_index: int,
    confidence: float = 0.95,
) -> CoverageResult:
    """Fraction of runs whose credible interval contains the true state at `time_index`.

    Simulation only. Coverage is measured across runs rather than pooled over time within
    a run: the indicators along one run are autocorrelated for the same reason NEES is, so
    pooling would understate the uncertainty and the binomial standard error would be
    wrong — tempting, because pooling makes the measured coverage look reassuringly tight.
    """
    x = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    p = np.asarray(variance, dtype=np.float64)
    for name, values in (("truth", x), ("mean", m), ("variance", p)):
        _require_runs_by_samples(name, values)
    if not 0.0 < confidence < 1.0:
        message = f"confidence must be strictly between 0 and 1, got {confidence}"
        raise ValueError(message)

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    half_width = z * np.sqrt(p[:, time_index])
    covered = np.abs(x[:, time_index] - m[:, time_index]) <= half_width
    return CoverageResult(
        n_covered=int(np.sum(covered)), n_runs=int(covered.size), nominal=confidence
    )
