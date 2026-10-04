"""Tests for the synthetic generators.

These run before anything that uses them. The generators are the analytic oracles the
rest of the suite is checked against, so if they are wrong every downstream failure has
two possible causes and the suite stops being diagnostic.

Note what is being tested here: the *generator*, against closed-form algebra, with no
spectral estimator involved. `test_power_law_amplitude_spectrum_is_exact` inspects the
FFT magnitudes directly rather than estimating a PSD, so a failure cannot be blamed on
the estimator that has not been written yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_state_estimator.types import FloatArray
from synthetic.generators import (
    DEFAULT_SAMPLE_RATE,
    ar_process,
    ar_theoretical_variance,
    burst_suppression,
    power_law_noise,
    sinusoid,
    white_noise,
)

FS = DEFAULT_SAMPLE_RATE


# --------------------------------------------------------------------------- sinusoid


def test_sinusoid_has_the_requested_amplitude() -> None:
    x = sinusoid(n_samples=1024, sample_rate=FS, frequency=10.0, amplitude=3.0)
    assert np.isclose(x.max(), 3.0, rtol=1e-3)
    assert np.isclose(x.min(), -3.0, rtol=1e-3)


def test_sinusoid_power_is_half_amplitude_squared() -> None:
    """The oracle REQ-012 is checked against. An integer number of cycles makes it exact."""
    # 8 s at 10 Hz is exactly 80 cycles, so the mean square is exactly A^2/2.
    x = sinusoid(n_samples=int(8 * FS), sample_rate=FS, frequency=10.0, amplitude=2.0)
    assert np.isclose(np.mean(x**2), 2.0, rtol=1e-12)


def test_sinusoid_is_deterministic_and_takes_no_seed() -> None:
    kwargs = {"n_samples": 256, "sample_rate": FS, "frequency": 7.0}
    assert np.array_equal(sinusoid(**kwargs), sinusoid(**kwargs))


# --------------------------------------------------------------------------- white noise


def test_white_noise_variance_matches_within_its_sampling_bound() -> None:
    """Variance of a variance estimate is 2*sigma^4/(n-1); allow 4 standard errors."""
    n = 8192
    x = white_noise(n_samples=n, sample_rate=FS, variance=3.0, seed=0)
    standard_error = 3.0 * np.sqrt(2.0 / (n - 1))
    assert abs(np.var(x) - 3.0) < 4.0 * standard_error


@pytest.mark.req("REQ-002")
def test_white_noise_is_reproducible_from_its_seed() -> None:
    a = white_noise(n_samples=512, sample_rate=FS, variance=1.0, seed=42)
    b = white_noise(n_samples=512, sample_rate=FS, variance=1.0, seed=42)
    c = white_noise(n_samples=512, sample_rate=FS, variance=1.0, seed=43)
    assert np.array_equal(a, b), "same seed must give bitwise-identical output"
    assert not np.array_equal(a, c), "different seeds must differ"


# --------------------------------------------------------------------------- power law


def test_power_law_amplitude_spectrum_is_exact() -> None:
    """The generator's realised spectrum follows f**-a exactly, not merely in expectation.

    This is what makes it usable as a slope oracle: there is no per-bin chi-square scatter
    to average away, so the expected slope and the realised slope are the same number.

    Checked directly on the FFT magnitudes, with no PSD estimator involved.
    """
    n, exponent, knee = 4096, 1.5, 1.0
    x = power_law_noise(
        n_samples=n, sample_rate=FS, exponent=exponent, knee=knee, variance=1.0, seed=0
    )
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    magnitude = np.abs(np.fft.rfft(x))

    above_knee = freqs > knee
    expected_shape = freqs[above_knee] ** (-exponent / 2.0)
    ratio = magnitude[above_knee] / expected_shape

    # A constant ratio means the shape is exactly right; only the overall scale differs.
    assert np.allclose(ratio, ratio[0], rtol=1e-9)


def test_power_law_is_flat_below_the_knee() -> None:
    """The knee bounds the dynamic range, which keeps taper leakage from biasing the slope."""
    n, knee = 4096, 2.0
    x = power_law_noise(n_samples=n, sample_rate=FS, exponent=3.0, knee=knee, seed=1)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    magnitude = np.abs(np.fft.rfft(x))

    below = (freqs > 0) & (freqs < knee)
    assert np.allclose(magnitude[below], magnitude[below][0], rtol=1e-9)


def test_power_law_has_zero_mean() -> None:
    """The DC bin is set to zero at synthesis, which removes the dominant leakage source."""
    x = power_law_noise(n_samples=2048, sample_rate=FS, exponent=1.0, seed=2)
    assert abs(np.mean(x)) < 1e-12


def test_power_law_scales_to_the_requested_variance() -> None:
    x = power_law_noise(n_samples=4096, sample_rate=FS, exponent=1.0, variance=5.0, seed=3)
    assert np.isclose(np.var(x), 5.0, rtol=1e-9)


# --------------------------------------------------------------------------- AR process


def test_ar_process_variance_matches_yule_walker() -> None:
    """Proves the burn-in was long enough: a short burn-in leaves the variance low."""
    coefficients = [0.6, -0.3]
    x = ar_process(
        n_samples=200_000, sample_rate=FS, coefficients=coefficients, innovation_std=1.0, seed=0
    )
    theoretical = ar_theoretical_variance(coefficients=coefficients, innovation_std=1.0)
    assert np.isclose(np.var(x), theoretical, rtol=0.05)


def test_ar_process_rejects_an_unstable_specification() -> None:
    with pytest.raises(ValueError, match="stationary"):
        ar_process(n_samples=100, sample_rate=FS, coefficients=[1.5], seed=0)


# --------------------------------------------------------------------- burst suppression


@pytest.mark.parametrize(
    ("n_samples", "fraction"),
    [(1280, 0.3), (1001, 0.3), (5000, 0.457), (2048, 0.0), (2048, 1.0)],
)
def test_burst_suppression_fraction_is_exact_in_sample_counts(
    n_samples: int, fraction: float
) -> None:
    """'Exact' means exact in integer samples -- you cannot have 30% of 1001 samples.

    The record reports what it actually produced, so a test never has to guess.
    """
    record = burst_suppression(
        n_samples=n_samples, sample_rate=FS, suppression_fraction=fraction, seed=0
    )
    assert record.suppression_mask.sum() == record.n_suppressed_samples
    assert record.realized_fraction == record.n_suppressed_samples / n_samples
    assert abs(record.realized_fraction - fraction) <= 0.5 / n_samples
    assert record.signal.size == n_samples
    assert record.suppression_mask.size == n_samples


def test_burst_suppression_episodes_cover_the_record_exactly() -> None:
    record = burst_suppression(
        n_samples=4096, sample_rate=FS, suppression_fraction=0.35, n_episodes=5, seed=0
    )
    assert sum(record.episode_lengths) == record.n_suppressed_samples


def test_suppressed_segments_are_low_amplitude_not_zero() -> None:
    """Zeros would make the Tier 3 lead-off/flatline test vacuous -- a disconnected
    electrode and a suppressed brain must remain distinguishable."""
    record = burst_suppression(
        n_samples=4096,
        sample_rate=FS,
        suppression_fraction=0.4,
        burst_amplitude=30.0,
        suppression_amplitude=2.0,
        seed=0,
    )
    suppressed: FloatArray = record.signal[record.suppression_mask]
    bursting: FloatArray = record.signal[~record.suppression_mask]
    assert np.std(suppressed) > 0.0, "suppressed segments must not be identically zero"
    assert np.std(suppressed) < np.std(bursting) / 3.0


def test_burst_suppression_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="suppression_fraction"):
        burst_suppression(n_samples=1024, sample_rate=FS, suppression_fraction=1.5, seed=0)
