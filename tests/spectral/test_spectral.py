"""Spectral estimation against analytically known answers.

Where a closed-form answer exists it is used directly. Where one does not — the shape of
an estimated spectrum under noise — the test states an ensemble property instead of
pretending a single realisation is deterministic.

Several functions here are tested on **hand-constructed PSD arrays** rather than on
estimated ones. `spectral_edge_frequency` of a flat PSD has an exact answer; the SEF of
an estimated white-noise spectrum has 1.5 Hz of seed-to-seed scatter. Testing the
algorithm separately from the estimator gives two precise tests instead of one vague one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eeg_state_estimator.spectral import (
    MultitaperConfig,
    Spectrum,
    band_power,
    band_powers,
    dpss_tapers,
    find_peak,
    fit_power_law,
    multitaper_psd,
    spectral_edge_frequency,
    total_power,
)
from synthetic.generators import (
    DEFAULT_SAMPLE_RATE,
    power_law_noise,
    sinusoid,
    white_noise,
)

FS = DEFAULT_SAMPLE_RATE


def _grid() -> np.ndarray:
    """A frequency grid landing exactly on integer hertz, so band edges are grid points."""
    return np.arange(0.0, FS / 2.0 + 1e-9, 0.125)


def _flat_spectrum(low: float, high: float, level: float = 1.0) -> Spectrum:
    """A PSD array built by hand, so the correct answers are exact."""
    frequencies = _grid()
    psd = np.where((frequencies >= low) & (frequencies <= high), level, 0.0)
    return Spectrum(
        frequencies=frequencies,
        psd=np.asarray(psd, dtype=np.float64),
        sample_rate=FS,
        half_bandwidth=1.0,
        n_tapers=7,
    )


# --------------------------------------------------------------------------- REQ-010


@pytest.mark.req("REQ-010")
def test_dpss_tapers_carry_unit_energy() -> None:
    """The PSD normalisation derives from sum(w**2) == 1 for every taper.

    SciPy's dpss takes a `norm` parameter, and a different choice normalises to unit
    *peak* instead, which would scale every PSD by roughly N/4 with no error and no
    warning. Asserting the assumption converts a silent dependency into a checked one.
    """
    tapers = dpss_tapers(n_samples=512, time_bandwidth=4.0, n_tapers=7)
    assert tapers.shape == (7, 512)
    assert np.allclose(np.sum(tapers**2, axis=1), 1.0, rtol=1e-12)


@pytest.mark.req("REQ-010")
def test_taper_count_defaults_to_twice_time_bandwidth_minus_one() -> None:
    config = MultitaperConfig(time_bandwidth=4.0)
    assert config.resolved_n_tapers() == 7


@pytest.mark.req("REQ-010")
def test_frequency_axis_is_one_sided_from_dc_to_nyquist() -> None:
    spectrum = multitaper_psd(white_noise(n_samples=1024, sample_rate=FS, seed=0), sample_rate=FS)
    assert spectrum.frequencies[0] == 0.0
    assert np.isclose(spectrum.frequencies[-1], FS / 2.0)
    assert np.all(np.diff(spectrum.frequencies) > 0.0)


# --------------------------------------------------------------------------- REQ-001


@pytest.mark.req("REQ-001")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"time_bandwidth": 4.0, "n_tapers": 8}, "n_tapers"),
        ({"time_bandwidth": 0.5}, "time_bandwidth"),
    ],
)
def test_config_rejects_invalid_taper_settings(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MultitaperConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.req("REQ-001")
@pytest.mark.parametrize(
    ("signal", "sample_rate", "match"),
    [
        (np.array([]), FS, "empty"),
        (np.ones((4, 4)), FS, "one-dimensional"),
        (np.array([1.0, np.nan, 2.0]), FS, "finite"),
        (np.ones(128), 0.0, "sample_rate"),
    ],
)
def test_multitaper_psd_rejects_bad_input(
    signal: np.ndarray, sample_rate: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        multitaper_psd(signal, sample_rate=sample_rate)


# --------------------------------------------------------------------------- REQ-013


@pytest.mark.req("REQ-013")
@pytest.mark.parametrize("duration_s", [4, 8, 20])
def test_psd_integral_equals_taper_weighted_mean_square_exactly(duration_s: int) -> None:
    """Parseval, as an exact identity rather than a statistical claim.

    Comparing the PSD integral to the time-domain variance of white noise measures the
    sampling error of a variance estimate (2.8% at 4 s), not the normalisation. Against
    the taper-weighted mean square the identity is exact, so this test pins the taper
    normalisation and the one-sided doubling rule and nothing else.
    """
    n = int(duration_s * FS)
    signal = white_noise(n_samples=n, sample_rate=FS, variance=3.0, seed=0)
    config = MultitaperConfig(detrend_mean=False)
    spectrum = multitaper_psd(signal, sample_rate=FS, config=config)

    tapers = dpss_tapers(
        n_samples=n, time_bandwidth=config.time_bandwidth, n_tapers=config.resolved_n_tapers()
    )
    reference = float(np.mean(np.sum((tapers * signal) ** 2, axis=1)))
    integral = float(np.sum(spectrum.psd) * spectrum.bin_width)

    assert np.isclose(integral, reference, rtol=1e-12)


# --------------------------------------------------------------------------- REQ-014


@pytest.mark.req("REQ-014")
def test_total_power_of_a_deterministic_multi_sinusoid() -> None:
    """Physical power on a deterministic signal: no sampling noise to hide behind."""
    n = int(8 * FS)
    amplitudes = (2.0, 1.0, 0.5)
    frequencies = (6.0, 10.0, 21.0)
    signal = sum(
        sinusoid(n_samples=n, sample_rate=FS, frequency=f, amplitude=a)
        for f, a in zip(frequencies, amplitudes, strict=True)
    )
    spectrum = multitaper_psd(np.asarray(signal), sample_rate=FS)

    expected = sum(a**2 / 2.0 for a in amplitudes)
    integral = float(np.sum(spectrum.psd) * spectrum.bin_width)
    assert np.isclose(integral, expected, rtol=0.02)


# --------------------------------------------------------------------------- REQ-011/012


@pytest.mark.req("REQ-011")
@pytest.mark.parametrize("frequency", [6.3, 9.0, 10.37, 15.5, 23.81])
def test_peak_frequency_within_a_quarter_hertz(frequency: float) -> None:
    """Off-bin frequencies specifically: an on-bin frequency would flatter any method.

    The peak is a power-weighted centroid, not the argmax bin. Averaging K concentrated
    tapers turns a spectral line into a flat-topped plateau of width 2W, so the argmax is
    pinned by taper ripple and its error is bounded by W rather than by the bin width —
    measured 0.32 Hz at this record length, which fails outright. The plateau is symmetric
    about the true frequency, so its first moment is right where its maximum is not.
    """
    n = int(4 * FS)
    signal = sinusoid(n_samples=n, sample_rate=FS, frequency=frequency, amplitude=2.0)
    spectrum = multitaper_psd(signal, sample_rate=FS)

    peak = find_peak(spectrum, search_low=2.0, search_high=45.0)
    assert peak is not None
    assert abs(peak.frequency - frequency) <= 0.25


@pytest.mark.req("REQ-012")
@pytest.mark.parametrize("amplitude", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("duration_s", [4, 8])
def test_peak_power_within_five_percent(amplitude: float, duration_s: int) -> None:
    """The peak PSD *bin* is not the signal power -- it is not even the same unit.

    Multitaper deliberately spreads a line over the analysis bandwidth, so recovering the
    power means integrating over that bandwidth. The bin value has units of power per
    hertz and depends on W; A**2/2 has units of power.
    """
    n = int(duration_s * FS)
    signal = sinusoid(n_samples=n, sample_rate=FS, frequency=11.0, amplitude=amplitude)
    spectrum = multitaper_psd(signal, sample_rate=FS)

    peak = find_peak(spectrum, search_low=2.0, search_high=45.0)
    assert peak is not None
    assert np.isclose(peak.power, amplitude**2 / 2.0, rtol=0.05)


@pytest.mark.req("REQ-011")
def test_find_peak_returns_none_when_there_is_no_peak() -> None:
    """No alpha peak is a real state -- deep suppression -- not an error.

    Reporting it as None rather than raising, or than returning a meaningless number, is
    the same principle as making "no valid signal" a distinct output state.
    """
    frequencies = _grid()
    monotone = 1.0 / (1.0 + frequencies)
    spectrum = Spectrum(
        frequencies=frequencies,
        psd=np.asarray(monotone, dtype=np.float64),
        sample_rate=FS,
        half_bandwidth=1.0,
        n_tapers=7,
    )
    assert find_peak(spectrum, search_low=10.0, search_high=20.0) is None


# --------------------------------------------------------------------------- REQ-015


@pytest.mark.req("REQ-015")
@pytest.mark.parametrize("band", [(1.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)])
def test_white_noise_psd_is_flat_at_the_analytic_level(band: tuple[float, float]) -> None:
    """Flatness is a property of the expected spectrum, so it is tested over an ensemble.

    A single record cannot be flat to 10%: the per-bin relative standard deviation is
    1/sqrt(K) = 38% by design, and measurement gives 200%+ regardless of record length.
    Comparing the ensemble mean to the analytic level 2*sigma**2/fs rather than to the
    record's own band mean makes this strictly stronger -- it checks flatness and
    absolute normalisation in one assertion.
    """
    variance, n_records = 2.0, 50
    n = int(8 * FS)
    accumulated = None
    for seed in range(n_records):
        spectrum = multitaper_psd(
            white_noise(n_samples=n, sample_rate=FS, variance=variance, seed=seed), sample_rate=FS
        )
        accumulated = spectrum.psd if accumulated is None else accumulated + spectrum.psd
    assert accumulated is not None
    mean_psd = accumulated / n_records

    frequencies = spectrum.frequencies
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    analytic_level = 2.0 * variance / FS
    assert np.isclose(float(np.mean(mean_psd[selected])), analytic_level, rtol=0.10)


# --------------------------------------------------------------------------- REQ-016


@pytest.mark.req("REQ-016")
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_power_law_slope_recovered(exponent: float) -> None:
    n = int(8 * FS)
    signal = power_law_noise(
        n_samples=n, sample_rate=FS, exponent=exponent, knee=1.0, variance=1.0, seed=0
    )
    spectrum = multitaper_psd(signal, sample_rate=FS)
    fit = fit_power_law(spectrum, low=2.0, high=40.0)
    assert abs(fit.exponent - exponent) <= 0.15


# --------------------------------------------------------------------------- REQ-017


@pytest.mark.req("REQ-017")
def test_band_powers_over_a_partition_sum_to_the_total() -> None:
    spectrum = multitaper_psd(
        white_noise(n_samples=int(8 * FS), sample_rate=FS, variance=4.0, seed=0), sample_rate=FS
    )
    bands = {"a": (1.0, 10.0), "b": (10.0, 20.0), "c": (20.0, 30.0)}
    parts = band_powers(spectrum, bands)
    assert np.isclose(sum(parts.values()), total_power(spectrum, low=1.0, high=30.0), rtol=0.01)


@pytest.mark.req("REQ-017")
def test_band_power_of_a_hand_built_flat_spectrum_is_exact() -> None:
    spectrum = _flat_spectrum(low=0.0, high=FS / 2.0, level=3.0)
    assert np.isclose(band_power(spectrum, 10.0, 20.0), 3.0 * 10.0, rtol=1e-6)


# --------------------------------------------------------------------------- REQ-018


@pytest.mark.req("REQ-018")
def test_spectral_edge_of_a_flat_spectrum_is_exact() -> None:
    spectrum = _flat_spectrum(low=0.5, high=45.0)
    edge = spectral_edge_frequency(spectrum, fraction=0.95, low=0.5, high=45.0)
    assert np.isclose(edge, 0.5 + 0.95 * (45.0 - 0.5), rtol=1e-3)


@pytest.mark.req("REQ-018")
def test_spectral_edge_is_non_decreasing_in_the_fraction() -> None:
    spectrum = multitaper_psd(
        power_law_noise(n_samples=int(8 * FS), sample_rate=FS, exponent=1.5, seed=0), sample_rate=FS
    )
    edges = [spectral_edge_frequency(spectrum, fraction=f) for f in (0.5, 0.7, 0.9, 0.95, 0.99)]
    assert all(b >= a for a, b in itertools.pairwise(edges))


@pytest.mark.req("REQ-018")
def test_spectral_edge_rejects_a_fraction_outside_the_open_unit_interval() -> None:
    spectrum = _flat_spectrum(low=0.5, high=45.0)
    with pytest.raises(ValueError, match="fraction"):
        spectral_edge_frequency(spectrum, fraction=1.0)


# --------------------------------------------------------------------------- REQ-019/020


@pytest.mark.req("REQ-019")
@pytest.mark.parametrize("scale", [0.5, 3.0, 1000.0])
def test_scaling_the_signal_scales_the_psd_by_the_square(scale: float) -> None:
    signal = white_noise(n_samples=1024, sample_rate=FS, seed=0)
    base = multitaper_psd(signal, sample_rate=FS)
    scaled = multitaper_psd(signal * scale, sample_rate=FS)
    assert np.allclose(scaled.psd, base.psd * scale**2, rtol=1e-10)


@pytest.mark.req("REQ-020")
@pytest.mark.parametrize("sample_rate", [100.0, 128.0, 256.0, 512.0])
def test_the_same_sinusoid_is_found_at_any_sample_rate(sample_rate: float) -> None:
    """Same continuous-time signal, different sampling. Peak and power must agree."""
    duration_s, frequency, amplitude = 8.0, 10.37, 2.0
    n = int(duration_s * sample_rate)
    signal = sinusoid(
        n_samples=n, sample_rate=sample_rate, frequency=frequency, amplitude=amplitude
    )
    spectrum = multitaper_psd(signal, sample_rate=sample_rate)

    peak = find_peak(spectrum, search_low=2.0, search_high=45.0)
    assert peak is not None
    assert abs(peak.frequency - frequency) <= 0.25
    assert np.isclose(peak.power, amplitude**2 / 2.0, rtol=0.05)


# --------------------------------------------------------------------------- REQ-002


@pytest.mark.req("REQ-002")
def test_multitaper_psd_is_bitwise_deterministic() -> None:
    signal = white_noise(n_samples=1024, sample_rate=FS, seed=0)
    first = multitaper_psd(signal, sample_rate=FS)
    second = multitaper_psd(signal, sample_rate=FS)
    assert np.array_equal(first.psd, second.psd)
    assert np.array_equal(first.frequencies, second.frequencies)
