"""Streaming behaviour: equivalence, causality, warm-up and bounded memory.

The two that matter are `test_chunked_output_equals_batch_output` and
`test_appending_future_samples_does_not_change_past_output`. Everything else in the
package can be right while those are wrong, and if they are wrong the estimator cannot be
used on a live signal at all — a filter that needs the end of the recording to compute the
middle of it is an offline analysis wearing a streaming interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_state_estimator.base import StreamingEstimator
from eeg_state_estimator.pipeline import EpochPipeline
from eeg_state_estimator.statespace import RandomWalkModel, ScalarKalmanFilter
from synthetic.generators import DEFAULT_SAMPLE_RATE, power_law_noise, sinusoid, white_noise

FS = DEFAULT_SAMPLE_RATE


def _record(n_seconds: float = 40.0, seed: int = 0) -> np.ndarray:
    """A record with alpha-band structure, so the tracked feature actually moves."""
    n = int(n_seconds * FS)
    background = power_law_noise(n_samples=n, sample_rate=FS, exponent=1.5, variance=4.0, seed=seed)
    alpha = sinusoid(n_samples=n, sample_rate=FS, frequency=10.0, amplitude=6.0)
    envelope = np.linspace(0.2, 1.8, n)
    return np.asarray(background + envelope * alpha, dtype=np.float64)


def _pipeline() -> EpochPipeline:
    return EpochPipeline(
        sample_rate=FS,
        band=(8.0, 12.0),
        epoch_seconds=2.0,
        model=RandomWalkModel(process_variance=0.02, observation_variance=0.05),
    )


# --------------------------------------------------------------------------- REQ-050


@pytest.mark.req("REQ-050")
def test_both_estimators_satisfy_the_streaming_interface() -> None:
    """The ABC is satisfied without adaptation, which is why `step()`/`update()` were
    named that way from the start rather than renamed when streaming arrived."""
    assert isinstance(_pipeline(), StreamingEstimator)
    assert isinstance(ScalarKalmanFilter(RandomWalkModel(0.1, 1.0)), StreamingEstimator)


@pytest.mark.req("REQ-050")
def test_pipeline_emits_one_estimate_per_whole_epoch() -> None:
    pipeline = _pipeline()
    estimates = pipeline.update(_record(n_seconds=10.0))
    assert len(estimates) == 5  # 10 s at 2 s per epoch
    assert [e.epoch_index for e in estimates] == [0, 1, 2, 3, 4]
    assert estimates[0].start_time == 0.0
    assert estimates[1].start_time == 2.0


# --------------------------------------------------------------------------- REQ-051


@pytest.mark.req("REQ-051")
@pytest.mark.parametrize("chunk_size", [1, 7, 128, 257, 1024, 5000])
def test_chunked_output_equals_batch_output(chunk_size: int) -> None:
    """Arbitrary chunk boundaries, including ones that split epochs at awkward points.

    A chunk size of 1 is the strictest case: every sample arrives separately, so any
    dependence on chunk alignment shows up immediately.
    """
    record = _record()

    batch = _pipeline().update(record)

    streamed_pipeline = _pipeline()
    streamed = []
    for start in range(0, record.size, chunk_size):
        streamed.extend(streamed_pipeline.update(record[start : start + chunk_size]))

    assert len(streamed) == len(batch)
    for streamed_estimate, batch_estimate in zip(streamed, batch, strict=True):
        assert streamed_estimate == batch_estimate


@pytest.mark.req("REQ-051")
def test_chunked_kalman_output_equals_batch_output() -> None:
    model = RandomWalkModel(0.05, 1.0)
    observations = white_noise(n_samples=2000, sample_rate=FS, seed=1)

    batch = ScalarKalmanFilter(model).update(observations)

    streaming = ScalarKalmanFilter(model)
    means = []
    for start in range(0, observations.size, 37):
        means.extend(streaming.update(observations[start : start + 37]).mean)

    assert np.array_equal(np.asarray(means), batch.mean)


# --------------------------------------------------------------------------- REQ-052


@pytest.mark.req("REQ-052")
def test_appending_future_samples_does_not_change_past_output() -> None:
    """Strict causality, stated as the property rather than as an inspection of the code.

    A look-ahead can be introduced by something as innocent as a centred moving average or
    a zero-phase filter, neither of which looks like a bug in review. The only reliable
    check is behavioural: extend the record and confirm nothing already emitted moves.
    """
    record = _record(n_seconds=40.0)

    short = _pipeline().update(record[: int(20.0 * FS)])
    long = _pipeline().update(record)

    assert len(long) > len(short)
    for early, later in zip(short, long[: len(short)], strict=True):
        assert early == later


@pytest.mark.req("REQ-052")
def test_future_samples_cannot_reach_an_already_emitted_epoch() -> None:
    """The same property under an adversarial tail rather than a benign one."""
    record = _record(n_seconds=20.0)
    outrageous = np.full(int(5.0 * FS), 1.0e6)

    before = _pipeline().update(record)
    after = _pipeline().update(np.concatenate([record, outrageous]))

    for early, later in zip(before, after[: len(before)], strict=True):
        assert early == later


# --------------------------------------------------------------------------- REQ-053


@pytest.mark.req("REQ-053")
def test_warmup_estimates_are_flagged_invalid() -> None:
    """Warm-up is defined by the posterior variance, not by a hand-picked epoch count.

    The variance recursion is independent of the data, so "has the filter converged" is
    answerable from the Riccati fixed point alone — a self-calibrating criterion that
    adapts when Q or R change instead of silently becoming wrong.
    """
    pipeline = EpochPipeline(
        sample_rate=FS,
        band=(8.0, 12.0),
        epoch_seconds=2.0,
        model=RandomWalkModel(process_variance=0.02, observation_variance=0.05),
        initial_variance=1.0e6,
    )
    estimates = pipeline.update(_record(n_seconds=60.0))

    assert not estimates[0].valid, "the first epoch cannot already be converged"
    assert estimates[0].status == "warmup"
    assert estimates[-1].valid, "the filter must leave warm-up on a record this long"
    assert estimates[-1].status == "ok"

    # Validity must latch on once, not flicker: the variance decreases monotonically to
    # its fixed point, so a valid estimate can never be followed by an invalid one.
    first_valid = next(index for index, e in enumerate(estimates) if e.valid)
    assert all(e.valid for e in estimates[first_valid:])


@pytest.mark.req("REQ-053")
def test_an_invalid_estimate_still_carries_its_numbers() -> None:
    """Flagged, not suppressed. A caller may want to plot the warm-up; what it may not do
    is mistake it for a converged estimate, which the flag prevents."""
    pipeline = _pipeline()
    first = pipeline.update(_record(n_seconds=4.0))[0]
    assert not first.valid
    assert np.isfinite(first.state)
    assert first.variance > 0.0


# --------------------------------------------------------------------------- REQ-054


@pytest.mark.req("REQ-054")
def test_retained_state_is_constant_in_record_length() -> None:
    pipeline = _pipeline()
    pipeline.update(_record(n_seconds=10.0))
    after_short = pipeline.state_nbytes
    pipeline.update(_record(n_seconds=200.0, seed=2))
    assert pipeline.state_nbytes == after_short


@pytest.mark.req("REQ-054")
def test_a_partial_epoch_is_the_most_that_is_ever_retained() -> None:
    """The buffer is drained as epochs complete, so what is held is bounded by one epoch
    regardless of how much has been streamed through."""
    pipeline = _pipeline()
    pipeline.update(_record(n_seconds=100.0))
    assert pipeline.buffered_samples < pipeline.epoch_samples


# --------------------------------------------------------------------------- REQ-055


@pytest.mark.req("REQ-055")
def test_pipeline_tracks_a_rising_alpha_envelope() -> None:
    """End to end: raw samples in, a tracked state with an interval out.

    The record's alpha amplitude rises monotonically, so the tracked log band power must
    rise too. This is the integration test -- it would catch the spectral stage and the
    filter being wired together incorrectly even though both pass their own tests.
    """
    estimates = [e for e in _pipeline().update(_record(n_seconds=60.0)) if e.valid]
    assert len(estimates) > 10

    states = np.array([e.state for e in estimates])
    first_quarter = states[: len(states) // 4].mean()
    last_quarter = states[-len(states) // 4 :].mean()
    assert last_quarter > first_quarter + 0.5


@pytest.mark.req("REQ-055")
def test_every_estimate_carries_a_credible_interval_containing_its_state() -> None:
    for estimate in _pipeline().update(_record(n_seconds=20.0)):
        assert estimate.lower < estimate.state < estimate.upper


@pytest.mark.req("REQ-002")
def test_reset_restores_the_pipeline_to_a_fresh_state() -> None:
    record = _record(n_seconds=20.0)
    pipeline = _pipeline()
    first = pipeline.update(record)
    pipeline.reset()
    again = pipeline.update(record)
    assert first == again


@pytest.mark.req("REQ-064")
def test_pipeline_absorbs_a_non_finite_chunk_instead_of_raising() -> None:
    """Changed deliberately when REQ-064 landed.

    An earlier revision raised on non-finite input. That is wrong at this layer: a monitor
    reports lost packets as NaN, so refusing the chunk means a routine dropout crashes the
    estimator. The numerical core stays strict -- multitaper_psd still raises -- and the
    pipeline is the layer that absorbs real-world data, flagging the epoch rather than
    estimating on it. See test_the_numerical_core_stays_strict_about_non_finite_input.
    """
    record = _record(n_seconds=20.0)
    record[int(5.0 * FS) : int(5.5 * FS)] = np.nan
    estimates = _pipeline().update(record)

    affected = [e for e in estimates if e.start_time <= 5.0 < e.end_time]
    assert affected
    assert not affected[0].valid
    assert affected[0].status == "dropped_samples"


@pytest.mark.req("REQ-001")
def test_pipeline_rejects_an_epoch_shorter_than_the_taper_design_allows() -> None:
    with pytest.raises(ValueError, match="epoch"):
        EpochPipeline(sample_rate=FS, band=(8.0, 12.0), epoch_seconds=0.02)


# --------------------------------------------------------------------------- REQ-056


@pytest.mark.req("REQ-056")
def test_a_dead_channel_is_not_reported_as_a_valid_measurement() -> None:
    """The safety case. A disconnected electrode produces a flat trace, which carries
    almost no band power — and a low-power reading is exactly what the deepest
    physiological state looks like. An estimator that reports the two the same way is
    unsafe, so "no signal" must be a distinct state rather than a number on the scale.
    """
    live = _record(n_seconds=40.0)
    dead = np.random.default_rng(0).normal(0.0, 1e-7, int(40.0 * FS))
    estimates = _pipeline().update(np.concatenate([live, dead]))

    during_signal = [e for e in estimates if 20.0 <= e.start_time <= 38.0]
    after_loss = [e for e in estimates if e.start_time >= 50.0]
    assert during_signal and after_loss

    assert all(e.valid for e in during_signal), "live signal must be usable"
    assert not any(e.valid for e in after_loss), "a dead channel must never read as valid"
    # Since REQ-065 the amplitude detector catches this before any spectral work happens,
    # so the status is the more specific "flatline" rather than the generic "no_signal".
    # The band-power floor path remains live for an epoch that has amplitude but no power
    # in the analysis band.
    assert all(e.status == "flatline" for e in after_loss)


@pytest.mark.req("REQ-056")
def test_the_filter_state_is_not_dragged_down_by_a_dead_channel() -> None:
    """Flagging is not enough: if the floor value is fed to the filter, the state walks to
    it and stays there, so the estimate is wrong for many epochs after signal returns."""
    live = _record(n_seconds=40.0)
    dead = np.zeros(int(20.0 * FS))
    estimates = _pipeline().update(np.concatenate([live, dead]))

    last_live = [e for e in estimates if e.start_time <= 38.0][-1]
    during_loss = [e for e in estimates if e.start_time >= 44.0]
    assert during_loss
    for estimate in during_loss:
        assert estimate.state == pytest.approx(last_live.state), (
            "state must be held, not advanced by a floor-clamped observation"
        )


@pytest.mark.req("REQ-056")
def test_recovery_after_the_channel_returns() -> None:
    live = _record(n_seconds=30.0)
    dead = np.zeros(int(10.0 * FS))
    estimates = _pipeline().update(np.concatenate([live, dead, _record(n_seconds=30.0, seed=5)]))
    after_recovery = [e for e in estimates if e.start_time >= 44.0]
    assert any(e.valid for e in after_recovery), "must resume once signal returns"


@pytest.mark.req("REQ-001")
def test_n_fft_shorter_than_the_record_is_rejected() -> None:
    """A shorter transform silently truncates the signal and breaks Parseval by ~46%,
    which would invalidate REQ-013 without raising anything."""
    from eeg_state_estimator.spectral import MultitaperConfig, multitaper_psd

    signal = white_noise(n_samples=1024, sample_rate=FS, seed=0)
    with pytest.raises(ValueError, match="n_fft"):
        multitaper_psd(signal, sample_rate=FS, config=MultitaperConfig(n_fft=512))


@pytest.mark.req("REQ-017")
def test_band_power_honours_edges_that_fall_between_bins() -> None:
    """Selecting whole bins moves a requested edge inward, quietly narrowing the integral.
    On a flat spectrum the answer is exactly level x width, so the error is visible."""
    from eeg_state_estimator.spectral import Spectrum, band_power

    frequencies = np.arange(0.0, 64.0 + 1e-9, 0.125)
    spectrum = Spectrum(
        frequencies=frequencies,
        psd=np.full(frequencies.size, 3.0),
        sample_rate=FS,
        half_bandwidth=1.0,
        n_tapers=7,
    )
    # 8.05 to 11.95 is 3.9 Hz wide and lands between bins at both ends.
    assert band_power(spectrum, 8.05, 11.95) == pytest.approx(3.0 * 3.9, rel=1e-9)
