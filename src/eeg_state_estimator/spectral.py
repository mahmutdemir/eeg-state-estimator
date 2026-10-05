"""Multitaper spectral estimation on DPSS tapers.

Implements REQ-001, REQ-002, REQ-010 … REQ-020.

Built directly on `scipy.signal.windows.dpss` rather than on a high-level spectrogram
helper, because the normalisation, the taper count and the one-sided doubling rule are
the parts that have to be right and the parts a reader needs to be able to check.

The analysis functions take a `Spectrum`, not a signal. Only `multitaper_psd` consumes a
signal. That boundary is deliberate: it lets `spectral_edge_frequency` and `band_power`
be tested against hand-built PSD arrays whose answers are exact, instead of against
estimated ones carrying the estimator's own variance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import dpss

from eeg_state_estimator.types import FloatArray

#: Conventional EEG analysis bands, in hertz.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "slow_delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


@dataclass(frozen=True, slots=True)
class MultitaperConfig:
    """Parameters of the multitaper estimate.

    `time_bandwidth` (NW) sets the half-bandwidth `W = NW / T` in hertz, so the
    resolution bandwidth is `2W`: two spectral lines closer than that are not resolved.
    `n_tapers` (K) sets the variance: the relative standard deviation of each PSD bin is
    roughly `1 / sqrt(K)`.

    The default NW = 4 with K = 7 gives 2 Hz resolution on a 4-second record, which
    separates an alpha peak from the delta shoulder, at 38% per-bin scatter — acceptable
    because every downstream quantity integrates over many bins.
    """

    time_bandwidth: float = 4.0
    n_tapers: int | None = None
    n_fft: int | None = None
    detrend_mean: bool = True

    def __post_init__(self) -> None:
        if self.time_bandwidth < 1.0:
            message = f"time_bandwidth must be at least 1.0, got {self.time_bandwidth}"
            raise ValueError(message)
        maximum = self.max_tapers()
        if self.n_tapers is not None and self.n_tapers > maximum:
            message = (
                f"n_tapers={self.n_tapers} exceeds 2*NW-1={maximum} for "
                f"time_bandwidth={self.time_bandwidth}. Beyond that the Slepian sequences "
                f"are no longer concentrated in the analysis band -- the next taper leaks "
                f"about 30% of its energy outside -- so adding them trades a small variance "
                f"reduction for out-of-band bias."
            )
            raise ValueError(message)
        if self.n_tapers is not None and self.n_tapers < 1:
            message = f"n_tapers must be at least 1, got {self.n_tapers}"
            raise ValueError(message)

    def max_tapers(self) -> int:
        """`2*NW - 1`: the number of Slepian sequences with eigenvalue close to one."""
        return int(2 * self.time_bandwidth) - 1

    def resolved_n_tapers(self) -> int:
        """The taper count actually used, filling in the default."""
        return self.max_tapers() if self.n_tapers is None else self.n_tapers

    def half_bandwidth_hz(self, n_samples: int, sample_rate: float) -> float:
        """`W = NW / T` in hertz, where `T` is the record duration."""
        return self.time_bandwidth * sample_rate / n_samples


@dataclass(frozen=True, slots=True)
class Spectrum:
    """A one-sided power spectral density and the parameters that produced it.

    `half_bandwidth` is carried rather than recomputed by callers. Peak power and peak
    frequency both integrate over a band whose width is a function of the taper design,
    and a caller recomputing `NW/T` by hand would eventually disagree with the tapers
    that were actually used.
    """

    frequencies: FloatArray
    psd: FloatArray
    sample_rate: float
    half_bandwidth: float
    n_tapers: int

    @property
    def bin_width(self) -> float:
        """Frequency spacing `Δf = fs / n_fft`."""
        return float(self.frequencies[1] - self.frequencies[0])


@dataclass(frozen=True, slots=True)
class SpectralPeak:
    """A spectral peak.

    `power` is the integral over the analysis band and has units of signal squared.
    `peak_psd` is the largest single bin, has units of signal squared per hertz, and is
    useful for plotting but is **not** the signal power — see `find_peak`.
    """

    frequency: float
    power: float
    peak_psd: float
    band_low: float
    band_high: float


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """Least-squares fit of `S(f) ∝ f**-exponent` over a frequency range."""

    exponent: float
    log10_intercept: float
    n_points: int
    fit_low: float
    fit_high: float


def dpss_tapers(*, n_samples: int, time_bandwidth: float, n_tapers: int) -> FloatArray:
    """Orthonormal Slepian (DPSS) tapers, shape `(n_tapers, n_samples)`.

    `norm=2` is load-bearing and passed explicitly: the PSD normalisation below assumes
    unit energy, `sum_n w_k[n]**2 == 1`, for every taper. SciPy offers other
    normalisations — one scales to unit *peak* instead, which would inflate every PSD by
    roughly `N/4` with no error and no warning. `test_dpss_tapers_carry_unit_energy`
    asserts the assumption rather than trusting it.
    """
    if time_bandwidth >= n_samples / 2.0:
        message = (
            f"record of {n_samples} samples is too short for time_bandwidth="
            f"{time_bandwidth}; it must be below n_samples/2"
        )
        raise ValueError(message)
    tapers = dpss(n_samples, time_bandwidth, Kmax=n_tapers, sym=True, norm=2)
    return np.asarray(tapers, dtype=np.float64)


def _validate_signal(signal: object, sample_rate: float) -> FloatArray:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        message = f"signal must be one-dimensional, got shape {samples.shape}"
        raise ValueError(message)
    if samples.size == 0:
        message = "signal must not be empty"
        raise ValueError(message)
    if not np.all(np.isfinite(samples)):
        message = "signal must be finite; found NaN or inf"
        raise ValueError(message)
    if sample_rate <= 0.0:
        message = f"sample_rate must be positive, got {sample_rate}"
        raise ValueError(message)
    return samples


def multitaper_psd(
    signal: object, *, sample_rate: float, config: MultitaperConfig | None = None
) -> Spectrum:
    """Estimate the one-sided power spectral density by the multitaper method.

    With unit-energy tapers and `Δf = fs / n_fft`, the two-sided estimate for taper `k`
    is `|FFT(w_k · x)|**2 / fs`, and the multitaper estimate is their mean over `k`. The
    one-sided spectrum doubles every bin except DC and, for an even transform length,
    Nyquist: those two have no distinct negative-frequency partner, so their power is
    already counted once.

    The resulting estimate satisfies Parseval exactly against the taper-weighted mean
    square of the record (REQ-013) — not approximately, and not only without zero padding.
    """
    config = config or MultitaperConfig()
    samples = _validate_signal(signal, sample_rate)
    n_samples = samples.size

    if config.detrend_mean:
        # Removes the dominant leakage source before the tapers see it. A residual mean
        # smears across the whole analysis band and lifts the lowest bins, which would
        # bias a low-frequency slope fit.
        samples = samples - samples.mean()

    n_tapers = config.resolved_n_tapers()
    n_fft = config.n_fft or n_samples
    # Zero-padding (n_fft > n_samples) is fine and leaves Parseval exact, because the
    # bin width shrinks in step. A SHORTER transform is not padding -- numpy truncates
    # the tapered signal, discarding samples and breaking the normalisation silently:
    # measured 46% Parseval error at n_fft = n/2. Reject it rather than return a
    # plausible-looking spectrum that violates REQ-013.
    if n_fft < n_samples:
        message = (
            f"n_fft={n_fft} is shorter than the {n_samples}-sample record, which would "
            f"truncate it and break the PSD normalisation; use n_fft >= n_samples"
        )
        raise ValueError(message)
    tapers = dpss_tapers(
        n_samples=n_samples, time_bandwidth=config.time_bandwidth, n_tapers=n_tapers
    )

    tapered = tapers * samples
    coefficients = np.fft.rfft(tapered, n=n_fft, axis=-1)
    two_sided = np.mean(np.abs(coefficients) ** 2, axis=0) / sample_rate

    one_sided = two_sided.copy()
    one_sided[1:] *= 2.0
    if n_fft % 2 == 0:
        one_sided[-1] /= 2.0

    frequencies = np.asarray(np.fft.rfftfreq(n_fft, 1.0 / sample_rate), dtype=np.float64)
    return Spectrum(
        frequencies=frequencies,
        psd=np.asarray(one_sided, dtype=np.float64),
        sample_rate=sample_rate,
        half_bandwidth=config.half_bandwidth_hz(n_samples, sample_rate),
        n_tapers=n_tapers,
    )


def band_power(spectrum: Spectrum, low: float, high: float) -> float:
    """Integrate the PSD over `[low, high]` hertz, giving power in signal units squared."""
    if high <= low:
        message = f"high ({high}) must exceed low ({low})"
        raise ValueError(message)
    if high < spectrum.frequencies[0] or low > spectrum.frequencies[-1]:
        return 0.0

    # Integrate between the exact requested edges, not between the nearest bins.
    # Selecting whole bins moves an edge inward -- asking for [8.05, 11.95] Hz would
    # integrate [8.125, 11.875] and quietly return a narrower band than requested.
    # The endpoint PSD values are linearly interpolated, consistent with the
    # trapezoidal rule used across the interior.
    edge_low = max(low, float(spectrum.frequencies[0]))
    edge_high = min(high, float(spectrum.frequencies[-1]))
    interior = (spectrum.frequencies > edge_low) & (spectrum.frequencies < edge_high)

    grid = np.concatenate(([edge_low], spectrum.frequencies[interior], [edge_high]))
    values = np.concatenate(
        (
            [float(np.interp(edge_low, spectrum.frequencies, spectrum.psd))],
            spectrum.psd[interior],
            [float(np.interp(edge_high, spectrum.frequencies, spectrum.psd))],
        )
    )
    return float(np.trapezoid(values, grid))


def band_powers(
    spectrum: Spectrum, bands: Mapping[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """Band powers for a set of named bands, defaulting to the conventional EEG bands."""
    bands = bands or DEFAULT_BANDS
    return {name: band_power(spectrum, low, high) for name, (low, high) in bands.items()}


def total_power(spectrum: Spectrum, *, low: float = 0.5, high: float = 45.0) -> float:
    """Integrated power over the analysis range."""
    return band_power(spectrum, low, high)


def spectral_edge_frequency(
    spectrum: Spectrum, *, fraction: float = 0.95, low: float = 0.5, high: float = 45.0
) -> float:
    """The frequency below which `fraction` of the power in `[low, high]` lies.

    Uses a trapezoidal cumulative integral evaluated at the grid points, then linear
    interpolation. The three plausible conventions (step, right-edge cumulative sum,
    trapezoid) agree to about 0.1 Hz, which is far below the 1.5 Hz seed-to-seed scatter
    of an estimated spectrum — so the convention matters much less than it appears, and
    the algorithm is verified against hand-built PSDs where the answer is exact.
    """
    if not 0.0 < fraction < 1.0:
        message = f"fraction must be strictly between 0 and 1, got {fraction}"
        raise ValueError(message)
    selected = (spectrum.frequencies >= low) & (spectrum.frequencies <= high)
    frequencies = spectrum.frequencies[selected]
    psd = spectrum.psd[selected]
    if frequencies.size < 3:
        message = f"band [{low}, {high}] Hz holds only {frequencies.size} bins; need at least 3"
        raise ValueError(message)

    segment_areas = np.diff(frequencies) * (psd[:-1] + psd[1:]) / 2.0
    cumulative = np.concatenate(([0.0], np.cumsum(segment_areas)))
    if cumulative[-1] <= 0.0:
        message = "total power in the band is zero; spectral edge is undefined"
        raise ValueError(message)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(fraction, cumulative, frequencies))


def find_peak(
    spectrum: Spectrum,
    *,
    search_low: float,
    search_high: float,
    band_half_width_factor: float = 1.5,
) -> SpectralPeak | None:
    """Locate a spectral peak and report its frequency and its power.

    **The frequency is a power-weighted centroid, not the argmax bin.** Averaging `K`
    concentrated tapers does not produce a peak over a spectral line — it produces a
    flat-topped plateau of width about `2W`. The argmax within a plateau is set by taper
    ripple, so its error is bounded by `W` rather than by the bin width: measured 0.32 Hz
    on a 4-second record, which exceeds a full bin and fails the ±0.25 Hz requirement.
    Zero-padding does not help, because it samples a flat plateau more finely. The
    plateau is symmetric about the true frequency, so its first moment is correct where
    its maximum is not — the centroid measures 0.0096 Hz on the same records.

    **The power is the integral over the band, not the peak bin value.** Those are not
    the same quantity: the bin has units of power per hertz and depends on `W`, while the
    line power `A**2/2` has units of power. No tolerance makes the bin value correct.

    The band is `±band_half_width_factor · W` about the argmax. The factor is 1.5 rather
    than 1.0 because `W` is often close to an exact multiple of the bin width, so a `±W`
    boundary includes or excludes an edge bin carrying several percent of the line power
    depending on rounding; at 1.5 the boundary sits where the PSD is already small.

    Returns `None` when the search range holds no interior local maximum. "No peak" is a
    real state — deep suppression has no alpha peak — so it is reported as a distinct
    outcome rather than as an error or as a meaningless number.
    """
    selected = (spectrum.frequencies >= search_low) & (spectrum.frequencies <= search_high)
    indices = np.flatnonzero(selected)
    if indices.size < 3:
        return None

    local = spectrum.psd[indices]
    offset = int(np.argmax(local))
    # An endpoint maximum means the peak lies outside the search range, or there is no
    # peak at all -- a monotone slope through the window.
    if offset == 0 or offset == local.size - 1:
        return None

    peak_index = int(indices[offset])
    peak_frequency = float(spectrum.frequencies[peak_index])
    half_width = band_half_width_factor * spectrum.half_bandwidth

    band_low = max(peak_frequency - half_width, float(spectrum.frequencies[0]))
    band_high = min(peak_frequency + half_width, float(spectrum.frequencies[-1]))
    in_band = (spectrum.frequencies >= band_low) & (spectrum.frequencies <= band_high)

    band_frequencies = spectrum.frequencies[in_band]
    band_psd = spectrum.psd[in_band]
    weight = float(np.sum(band_psd))
    if weight <= 0.0:
        return None

    centroid = float(np.sum(band_frequencies * band_psd) / weight)
    power = float(np.trapezoid(band_psd, band_frequencies))
    return SpectralPeak(
        frequency=centroid,
        power=power,
        peak_psd=float(spectrum.psd[peak_index]),
        band_low=band_low,
        band_high=band_high,
    )


def fit_power_law(spectrum: Spectrum, *, low: float = 2.0, high: float = 40.0) -> PowerLawFit:
    """Fit `S(f) ∝ f**-exponent` by least squares on log10 axes.

    The fit starts at 2 Hz, not at the first bin. Three reasons, and they are
    independent: `log10(0)` is undefined so DC cannot enter at all; bins below `2W` are
    contaminated by leakage of any residual DC through the taper main lobe, which lifts
    them and flattens the fit (measured slope error +0.16 fitting from the first bin
    versus +0.04 from 1 Hz); and an unbounded `f**-a` is not a realisable process, so the
    lowest bins of a real record do not follow the law in any case.

    Note that log-log least squares on a chi-square-distributed PSD is biased, with
    `E[log S_hat] = log S + psi(K) - log K`. Because `K` is constant across frequency the
    bias shifts the intercept and leaves the slope unbiased, which is why the exponent is
    safe to report and the intercept carries this caveat.
    """
    if low <= 0.0:
        message = f"low must be positive for a log-log fit, got {low}"
        raise ValueError(message)
    selected = (spectrum.frequencies >= low) & (spectrum.frequencies <= high) & (spectrum.psd > 0.0)
    frequencies = spectrum.frequencies[selected]
    psd = spectrum.psd[selected]
    if frequencies.size < 3:
        message = f"range [{low}, {high}] Hz holds only {frequencies.size} usable bins"
        raise ValueError(message)

    log_f = np.log10(frequencies)
    log_s = np.log10(psd)
    design = np.column_stack([log_f, np.ones_like(log_f)])
    solution, *_ = np.linalg.lstsq(design, log_s, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])

    return PowerLawFit(
        exponent=-slope,
        log10_intercept=intercept,
        n_points=int(frequencies.size),
        fit_low=float(frequencies[0]),
        fit_high=float(frequencies[-1]),
    )
