"""Artifact detection, and the three-part contract each artifact must satisfy.

Detection on its own is not a useful property. An estimator that notices a fault and then
absorbs it into its state is still wrong, and stays wrong after the fault has gone. So
every artifact here is checked three ways:

1. **detected** - the flag is set on the contaminated epochs
2. **held** - the filter state does not advance while the fault persists
3. **recovered** - normal estimation resumes once it clears

Part 2 is the one that is easy to skip and expensive to omit.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_state_estimator.pipeline import EpochPipeline
from eeg_state_estimator.quality import QualityFlags, assess_epoch
from eeg_state_estimator.statespace import RandomWalkModel
from synthetic.artifacts import (
    inject_clipping,
    inject_dropout,
    inject_electrode_pop,
    inject_flatline,
    inject_line_noise,
)
from synthetic.generators import DEFAULT_SAMPLE_RATE, power_law_noise, sinusoid

FS = DEFAULT_SAMPLE_RATE


def clean_record(seconds: float = 60.0, seed: int = 0) -> np.ndarray:
    """Background plus a steady alpha rhythm - a signal the estimator tracks happily."""
    n = int(seconds * FS)
    background = power_law_noise(n_samples=n, sample_rate=FS, exponent=1.5, variance=4.0, seed=seed)
    alpha = sinusoid(n_samples=n, sample_rate=FS, frequency=10.0, amplitude=6.0)
    return np.asarray(background + alpha, dtype=np.float64)


def pipeline() -> EpochPipeline:
    return EpochPipeline(
        sample_rate=FS,
        band=(8.0, 12.0),
        epoch_seconds=2.0,
        model=RandomWalkModel(process_variance=0.02, observation_variance=0.05),
    )


def epochs_overlapping(estimates, start_s: float, end_s: float):
    return [e for e in estimates if e.start_time < end_s and e.end_time > start_s]


# --------------------------------------------------------------------------- REQ-060


@pytest.mark.req("REQ-060")
def test_a_clean_epoch_is_usable_with_no_flags_set() -> None:
    flags = assess_epoch(clean_record(seconds=2.0)[: int(2 * FS)], sample_rate=FS)
    assert isinstance(flags, QualityFlags)
    assert flags.usable
    assert not flags.any_fault
    assert flags.reason is None


@pytest.mark.req("REQ-060")
def test_the_verdict_summarises_the_individual_flags() -> None:
    """A caller may branch on `usable` alone without knowing which detectors exist."""
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    contaminated = inject_flatline(epoch, sample_rate=FS, start_s=0.0, duration_s=2.0).signal
    flags = assess_epoch(contaminated, sample_rate=FS)
    assert not flags.usable
    assert flags.any_fault
    assert flags.reason == "flatline"


# --------------------------------------------------------------------------- detectors


@pytest.mark.req("REQ-061")
@pytest.mark.parametrize("frequency", [50.0, 60.0])
def test_line_noise_is_detected_and_its_power_reported(frequency: float) -> None:
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    clean_flags = assess_epoch(epoch, sample_rate=FS, line_frequency=frequency)

    noisy = inject_line_noise(
        epoch, sample_rate=FS, start_s=0.0, duration_s=2.0, frequency=frequency, amplitude=20.0
    ).signal
    flags = assess_epoch(noisy, sample_rate=FS, line_frequency=frequency)

    assert not clean_flags.line_noise
    assert flags.line_noise
    assert flags.line_power > 10.0 * clean_flags.line_power
    assert not flags.usable


@pytest.mark.req("REQ-062")
def test_clipping_is_detected() -> None:
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    rail = float(np.percentile(np.abs(epoch), 97))
    clipped = inject_clipping(epoch, sample_rate=FS, start_s=0.0, duration_s=2.0, rail=rail).signal

    assert not assess_epoch(epoch, sample_rate=FS).clipping
    flags = assess_epoch(clipped, sample_rate=FS)
    assert flags.clipping
    assert not flags.usable


@pytest.mark.req("REQ-063")
def test_electrode_pop_is_detected() -> None:
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    popped = inject_electrode_pop(epoch, sample_rate=FS, at_s=1.0, height=300.0).signal

    assert not assess_epoch(epoch, sample_rate=FS).electrode_pop
    flags = assess_epoch(popped, sample_rate=FS)
    assert flags.electrode_pop
    assert not flags.usable


@pytest.mark.req("REQ-064")
def test_dropped_samples_are_detected_without_raising() -> None:
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    dropped = inject_dropout(epoch, sample_rate=FS, start_s=0.5, duration_s=0.2).signal

    flags = assess_epoch(dropped, sample_rate=FS)
    assert flags.dropped_samples
    assert not flags.usable


@pytest.mark.req("REQ-064")
def test_the_numerical_core_stays_strict_about_non_finite_input() -> None:
    """The pipeline absorbs real-world input; the spectral estimator does not. Keeping the
    core strict means a NaN can never reach the arithmetic and quietly produce an all-NaN
    spectrum with nothing raised."""
    from eeg_state_estimator.spectral import multitaper_psd

    epoch = clean_record(seconds=2.0)[: int(2 * FS)].copy()
    epoch[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        multitaper_psd(epoch, sample_rate=FS)


@pytest.mark.req("REQ-065")
def test_flatline_is_detected_from_amplitude() -> None:
    epoch = clean_record(seconds=2.0)[: int(2 * FS)]
    flat = inject_flatline(epoch, sample_rate=FS, start_s=0.0, duration_s=2.0).signal

    assert not assess_epoch(epoch, sample_rate=FS).flatline
    flags = assess_epoch(flat, sample_rate=FS)
    assert flags.flatline
    assert not flags.usable


# ------------------------------------------------------- REQ-066 / REQ-067 contract


# `sustained` records whether every epoch the injection touches should be flagged.
#
# It is False only for the electrode pop, and that asymmetry is a real property rather than
# a concession. A pop is an instantaneous step followed by an exponential decay: the step is
# what makes the epoch unusable, and it lands in exactly one epoch. The epoch after it
# contains only the smooth tail -- a few microvolts of slow offset that the spectral stage's
# mean removal handles -- with no sharp edge to detect and nothing left to reject. Asserting
# otherwise would be demanding that the detector fire on signal that is, by then, fine.
ARTIFACTS = [
    (
        "line noise",
        True,
        lambda x: inject_line_noise(
            x, sample_rate=FS, start_s=20.0, duration_s=10.0, amplitude=25.0
        ),
    ),
    (
        "clipping",
        True,
        lambda x: inject_clipping(
            x,
            sample_rate=FS,
            start_s=20.0,
            duration_s=10.0,
            rail=float(np.percentile(np.abs(x), 90)),
        ),
    ),
    (
        "electrode pop",
        False,
        lambda x: inject_electrode_pop(x, sample_rate=FS, at_s=21.0, height=400.0),
    ),
    ("dropout", True, lambda x: inject_dropout(x, sample_rate=FS, start_s=20.0, duration_s=10.0)),
    ("flatline", True, lambda x: inject_flatline(x, sample_rate=FS, start_s=20.0, duration_s=10.0)),
]
ARTIFACT_IDS = [a[0] for a in ARTIFACTS]


@pytest.mark.req("REQ-066")
@pytest.mark.parametrize(("name", "sustained", "inject"), ARTIFACTS, ids=ARTIFACT_IDS)
def test_contaminated_epochs_are_flagged_and_the_state_is_held(
    name: str, sustained: bool, inject
) -> None:
    """Parts 1 and 2 of the contract, for every artifact type."""
    contaminated = inject(clean_record())
    estimates = pipeline().update(contaminated.signal)
    start_s, end_s = contaminated.window(FS)

    touched = epochs_overlapping(estimates, start_s, end_s)
    assert touched, f"{name}: injection did not overlap any epoch"

    # 1 - detected. The onset epoch always; every touched epoch for a sustained fault.
    assert not touched[0].valid, f"{name}: the onset epoch read as valid"
    if sustained:
        assert not any(e.valid for e in touched), f"{name}: contaminated epochs read as valid"

    # 2 - held. Across the epochs actually flagged, the state must not move.
    flagged = [e for e in touched if not e.valid]
    states = np.round([e.state for e in flagged], 12)
    assert len(set(states)) == 1, f"{name}: state advanced during the artifact"


@pytest.mark.req("REQ-067")
@pytest.mark.parametrize(("name", "sustained", "inject"), ARTIFACTS, ids=ARTIFACT_IDS)
def test_estimation_resumes_after_the_artifact_clears(name: str, sustained: bool, inject) -> None:
    """Part 3. The filter was never corrupted, so recovery is immediate rather than
    something that has to be unwound."""
    contaminated = inject(clean_record())
    estimates = pipeline().update(contaminated.signal)
    _, end_s = contaminated.window(FS)

    after = [e for e in estimates if e.start_time >= end_s + 2.0]
    assert after, f"{name}: no epochs after the artifact"
    assert after[0].valid, f"{name}: did not resume within one epoch"
    assert all(e.valid for e in after), f"{name}: validity flickered after recovery"


@pytest.mark.req("REQ-066")
def test_a_clean_record_is_entirely_usable() -> None:
    """The control. If the detectors fire on clean signal they are worthless, however well
    they perform on contaminated signal."""
    estimates = pipeline().update(clean_record(seconds=60.0))
    settled = [e for e in estimates if e.start_time >= 10.0]
    assert all(e.valid for e in settled)
    assert all(e.status == "ok" for e in settled)
