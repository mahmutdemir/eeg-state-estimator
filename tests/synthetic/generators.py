"""Synthetic signals with analytically known answers.

These are the oracles. Every spectral requirement is checked against a signal whose
correct answer is known in closed form rather than against a reference implementation,
because comparing two implementations proves they agree while comparing against a
derivation proves the implementation is right.

Every function is keyword-only and takes an explicit `seed`. There is no module-level
random state and no use of the legacy `numpy.random` global API — ruff's NPY002 rejects
it — so determinism (REQ-002) is a property the linter enforces rather than a convention
somebody has to remember.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from eeg_state_estimator.types import FloatArray

#: Sample rate of the frontal EEG recordings this package is aimed at.
DEFAULT_SAMPLE_RATE: float = 128.0


def sinusoid(
    *,
    n_samples: int,
    sample_rate: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> FloatArray:
    """A pure sinusoid. Mean square is exactly `amplitude**2 / 2` over whole cycles.

    No randomness and no seed: this is a deterministic oracle for peak frequency
    (REQ-011) and peak power (REQ-012).
    """
    times = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.asarray(amplitude * np.sin(2.0 * np.pi * frequency * times + phase), dtype=np.float64)


def white_noise(
    *, n_samples: int, sample_rate: float, variance: float = 1.0, seed: int
) -> FloatArray:
    """Gaussian white noise of known variance.

    `sample_rate` is unused in the construction but is part of every generator's
    signature, so a caller never has to remember which ones need it.
    """
    del sample_rate
    if variance < 0.0:
        message = f"variance must be non-negative, got {variance}"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(0.0, math.sqrt(variance), n_samples), dtype=np.float64)


def power_law_noise(
    *,
    n_samples: int,
    sample_rate: float,
    exponent: float,
    knee: float = 1.0,
    variance: float = 1.0,
    seed: int,
    randomize_amplitude: bool = False,
) -> FloatArray:
    """Noise whose power spectrum follows `f**-exponent` above `knee`.

    Synthesised in the frequency domain with a **deterministic amplitude spectrum and
    random phase only**. Because `|X[m]|` is set rather than drawn, the realised
    spectrum follows the power law exactly instead of only in expectation — there is no
    per-bin chi-square scatter for a slope fit to average away. That is what makes this
    an oracle for REQ-016 rather than merely a sample from the right family.

    The `knee` is not cosmetic. An unbounded `f**-a` has unbounded power as `a` grows,
    and the resulting dynamic range exceeds the broadband leakage floor of a multitaper
    average: leaked low-frequency power then forms a pedestal under the high-frequency
    bins and flattens the measured slope. Bounding the spectrum below `knee` fixes that,
    and is also the physically honest choice — real `1/f` spectra have a low-frequency
    knee. Keep `knee` below the slope-fitting band.

    `randomize_amplitude=True` draws Rayleigh amplitudes instead, giving a genuinely
    stochastic realisation. Use it for property tests (scale invariance, determinism),
    not for slope recovery: it injects the estimator variance this oracle exists to avoid.
    """
    if knee <= 0.0:
        message = f"knee must be positive, got {knee}"
        raise ValueError(message)

    rng = np.random.default_rng(seed)
    freqs = np.asarray(np.fft.rfftfreq(n_samples, 1.0 / sample_rate), dtype=np.float64)

    # Amplitude ~ sqrt(power), hence the exponent halves. Flat below the knee.
    shape = np.maximum(freqs, knee) ** (-exponent / 2.0)
    if randomize_amplitude:
        shape = shape * rng.rayleigh(scale=1.0, size=shape.size)

    phases = rng.uniform(0.0, 2.0 * np.pi, freqs.size)
    spectrum = shape * np.exp(1j * phases)

    # A zero DC bin gives a zero-mean record, removing the dominant leakage source
    # before any taper sees it. The Nyquist bin of an even-length record must be real
    # for the inverse transform to be real-valued.
    spectrum[0] = 0.0
    if n_samples % 2 == 0:
        spectrum[-1] = np.abs(spectrum[-1])

    signal = np.fft.irfft(spectrum, n=n_samples)
    signal = signal - signal.mean()
    current = signal.std()
    if current > 0.0:
        signal = signal * (math.sqrt(variance) / current)
    return np.asarray(signal, dtype=np.float64)


def ar_theoretical_variance(*, coefficients: Sequence[float], innovation_std: float = 1.0) -> float:
    """Stationary variance of an AR(p) process, from the Yule-Walker equations.

    Used to prove the burn-in in `ar_process` was long enough: too short a burn-in
    leaves the realised variance below this value.
    """
    order = len(coefficients)
    phi = np.asarray(coefficients, dtype=np.float64)

    # Solve for autocovariances gamma[0..p] jointly. The Yule-Walker system is
    #   gamma[k] = sum_j phi[j] * gamma[|k - j - 1|]   for k >= 1
    #   gamma[0] = sum_j phi[j] * gamma[j]   + sigma^2
    design = np.zeros((order + 1, order + 1), dtype=np.float64)
    design[0, 0] = 1.0
    for j in range(order):
        design[0, j + 1] -= phi[j]
    for k in range(1, order + 1):
        design[k, k] = 1.0
        for j in range(order):
            design[k, abs(k - j - 1)] -= phi[j]

    rhs = np.zeros(order + 1, dtype=np.float64)
    rhs[0] = innovation_std**2
    gamma = np.linalg.solve(design, rhs)
    return float(gamma[0])


def ar_process(
    *,
    n_samples: int,
    sample_rate: float,
    coefficients: Sequence[float],
    innovation_std: float = 1.0,
    seed: int,
    burn_in: int | None = None,
) -> FloatArray:
    """A stationary autoregressive process, `x[t] = sum_k a_k x[t-k] + e[t]`.

    Generated with an explicit discarded burn-in rather than with `scipy.signal.lfilter_zi`,
    which returns the *step-response* steady state — not the stationary stochastic
    distribution — and would leave a subtle and wrong transient at the start of the record.
    """
    del sample_rate
    phi = np.asarray(coefficients, dtype=np.float64)

    # Stationary iff every root of the characteristic polynomial lies inside the unit
    # circle. Checked here so a caller gets a clear message instead of an array of inf.
    companion_roots = np.roots(np.concatenate(([1.0], -phi)))
    if companion_roots.size and np.max(np.abs(companion_roots)) >= 1.0:
        message = (
            f"coefficients {list(coefficients)} do not define a stationary process: "
            f"largest characteristic root has modulus {np.max(np.abs(companion_roots)):.4f}"
        )
        raise ValueError(message)

    order = phi.size
    if burn_in is None:
        burn_in = max(1000, 10 * order)

    rng = np.random.default_rng(seed)
    total = n_samples + burn_in
    innovations = rng.normal(0.0, innovation_std, total)
    series = np.zeros(total, dtype=np.float64)
    for t in range(order, total):
        series[t] = float(phi @ series[t - order : t][::-1]) + innovations[t]
    return np.asarray(series[burn_in:], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BurstSuppressionRecord:
    """A burst-suppression record and the ground truth that generated it.

    `realized_fraction` is reported rather than assumed, so a test never has to guess
    what the generator actually produced. See `burst_suppression` for what "exact" means.
    """

    signal: FloatArray
    suppression_mask: np.ndarray
    requested_fraction: float
    realized_fraction: float
    n_suppressed_samples: int
    episode_lengths: tuple[int, ...]


def _integer_partition(total: int, parts: int) -> list[int]:
    """Split `total` into `parts` integers that sum to it exactly."""
    base, remainder = divmod(total, parts)
    return [base + 1] * remainder + [base] * (parts - remainder)


def burst_suppression(
    *,
    n_samples: int,
    sample_rate: float,
    suppression_fraction: float,
    n_episodes: int = 5,
    burst_amplitude: float = 30.0,
    suppression_amplitude: float = 2.0,
    seed: int,
) -> BurstSuppressionRecord:
    """Alternating burst and suppression segments with an exactly known fraction.

    "Exact" means exact in **integer sample counts**: you cannot have exactly 30% of 1001
    samples. The guarantee is a single rounding at the start, so
    `abs(realized_fraction - suppression_fraction) <= 0.5 / n_samples`, with the realised
    value reported on the record.

    Segment lengths come from integer partitions and sum to `n_samples` by construction —
    there is no floating-point accumulation and no renewal process to drift.

    Suppressed segments are low-amplitude noise, **not zeros**. That is physiologically
    right, and it keeps burst suppression distinguishable from a flatline lead-off
    artifact; a generator emitting exact zeros would make that safety test vacuous.
    """
    if not 0.0 <= suppression_fraction <= 1.0:
        message = f"suppression_fraction must be in [0, 1], got {suppression_fraction}"
        raise ValueError(message)
    if n_episodes < 1:
        message = f"n_episodes must be at least 1, got {n_episodes}"
        raise ValueError(message)
    del sample_rate

    rng = np.random.default_rng(seed)
    n_suppressed = round(suppression_fraction * n_samples)
    n_bursting = n_samples - n_suppressed

    suppressed_lengths = _integer_partition(n_suppressed, n_episodes)
    burst_lengths = _integer_partition(n_bursting, n_episodes + 1)
    rng.shuffle(suppressed_lengths)
    rng.shuffle(burst_lengths)

    mask = np.zeros(n_samples, dtype=bool)
    cursor = 0
    for index, suppressed_length in enumerate(suppressed_lengths):
        cursor += burst_lengths[index]
        mask[cursor : cursor + suppressed_length] = True
        cursor += suppressed_length

    signal = rng.normal(0.0, burst_amplitude, n_samples)
    signal[mask] = rng.normal(0.0, suppression_amplitude, int(mask.sum()))

    return BurstSuppressionRecord(
        signal=np.asarray(signal, dtype=np.float64),
        suppression_mask=mask,
        requested_fraction=suppression_fraction,
        realized_fraction=n_suppressed / n_samples,
        n_suppressed_samples=n_suppressed,
        episode_lengths=tuple(suppressed_lengths),
    )
