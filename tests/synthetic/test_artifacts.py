"""Tests for the artifact injection harness.

The harness is ground truth for every detector test, so it is verified first and on its
own, exactly as the signal generators were. A detector scored against a harness that
mislabels where it put the artifact proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthetic.artifacts import (
    inject_clipping,
    inject_dropout,
    inject_electrode_pop,
    inject_flatline,
    inject_line_noise,
)
from synthetic.generators import DEFAULT_SAMPLE_RATE, white_noise

FS = DEFAULT_SAMPLE_RATE


@pytest.fixture
def clean() -> np.ndarray:
    return white_noise(n_samples=int(20 * FS), sample_rate=FS, variance=25.0, seed=0)


def test_injection_leaves_the_rest_of_the_record_untouched(clean: np.ndarray) -> None:
    """The single most important property: an artifact must be local to its window, or
    every detector test silently becomes a test of the whole record."""
    result = inject_line_noise(clean, sample_rate=FS, start_s=5.0, duration_s=2.0)
    assert np.array_equal(result.signal[~result.affected], clean[~result.affected])


def test_line_noise_lands_at_the_requested_time_and_amplitude(clean: np.ndarray) -> None:
    result = inject_line_noise(
        clean, sample_rate=FS, start_s=5.0, duration_s=2.0, frequency=50.0, amplitude=30.0
    )
    start, end = result.window(FS)
    assert start == pytest.approx(5.0, abs=1 / FS)
    assert end == pytest.approx(7.0, abs=2 / FS)

    added = result.signal - clean
    assert np.max(np.abs(added)) == pytest.approx(30.0, rel=0.02)

    # The added component must be at the frequency asked for, not merely "somewhere".
    segment = added[result.affected]
    freqs = np.fft.rfftfreq(segment.size, 1 / FS)
    assert freqs[np.argmax(np.abs(np.fft.rfft(segment)))] == pytest.approx(50.0, abs=1.0)


def test_clipping_marks_only_samples_actually_driven_to_the_rail(clean: np.ndarray) -> None:
    result = inject_clipping(clean, sample_rate=FS, start_s=5.0, duration_s=2.0, rail=4.0)
    window = slice(int(5.0 * FS), int(7.0 * FS))
    assert np.all(np.abs(result.signal[window]) <= 4.0 + 1e-12), "inside the window"
    assert np.any(np.abs(result.signal) > 4.0), "outside it the signal is untouched"
    assert result.n_affected > 0
    # Everything marked affected must sit exactly on a rail.
    assert np.allclose(np.abs(result.signal[result.affected]), 4.0)


def test_clipping_at_a_rail_above_the_signal_changes_nothing(clean: np.ndarray) -> None:
    result = inject_clipping(clean, sample_rate=FS, start_s=5.0, duration_s=2.0, rail=1e6)
    assert result.n_affected == 0
    assert np.array_equal(result.signal, clean)


def test_electrode_pop_is_a_step_that_decays(clean: np.ndarray) -> None:
    result = inject_electrode_pop(clean, sample_rate=FS, at_s=10.0, height=300.0, decay_s=0.25)
    added = result.signal - clean
    peak = int(np.argmax(np.abs(added)))

    assert added[peak] == pytest.approx(300.0, rel=0.02)
    assert peak == pytest.approx(int(10.0 * FS), abs=2)
    # One decay constant later it should have fallen to about 1/e of the step.
    one_tau = peak + int(0.25 * FS)
    assert added[one_tau] == pytest.approx(300.0 / np.e, rel=0.05)


def test_electrode_pop_outside_the_record_is_rejected(clean: np.ndarray) -> None:
    with pytest.raises(ValueError, match="outside the record"):
        inject_electrode_pop(clean, sample_rate=FS, at_s=1000.0)


def test_dropout_is_nan_not_zero(clean: np.ndarray) -> None:
    """A monitor reports lost samples as missing. Substituting zeros would turn a known
    unknown into a confident wrong answer, and would make this indistinguishable from a
    flatline."""
    result = inject_dropout(clean, sample_rate=FS, start_s=5.0, duration_s=1.0)
    assert np.all(np.isnan(result.signal[result.affected]))
    assert np.all(np.isfinite(result.signal[~result.affected]))
    assert result.n_affected == pytest.approx(int(1.0 * FS), abs=1)


def test_flatline_is_small_but_not_exactly_zero(clean: np.ndarray) -> None:
    """A detached electrode still shows amplifier noise. A detector keyed on exact zeros
    would miss the real case, and a test built on exact zeros would never reveal that."""
    result = inject_flatline(clean, sample_rate=FS, start_s=5.0, duration_s=2.0, residual=0.05)
    segment = result.signal[result.affected]
    assert np.std(segment) == pytest.approx(0.05, rel=0.2)
    assert not np.any(segment == 0.0)
    assert np.std(segment) < np.std(clean) / 50.0


@pytest.mark.req("REQ-002")
def test_injections_are_reproducible(clean: np.ndarray) -> None:
    kwargs = {"sample_rate": FS, "start_s": 5.0, "duration_s": 2.0}
    assert np.array_equal(
        inject_flatline(clean, seed=7, **kwargs).signal,
        inject_flatline(clean, seed=7, **kwargs).signal,
    )
