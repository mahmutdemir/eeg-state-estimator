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


@pytest.mark.req("REQ-001")
def test_pipeline_rejects_a_non_finite_chunk() -> None:
    with pytest.raises(ValueError, match="finite"):
        _pipeline().update(np.array([1.0, np.nan, 3.0]))


@pytest.mark.req("REQ-001")
def test_pipeline_rejects_an_epoch_shorter_than_the_taper_design_allows() -> None:
    with pytest.raises(ValueError, match="epoch"):
        EpochPipeline(sample_rate=FS, band=(8.0, 12.0), epoch_seconds=0.02)
