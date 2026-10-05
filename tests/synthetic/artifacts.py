"""Inject known artifacts at known times into a clean signal.

The counterpart to `generators.py`. Those build signals whose *correct answer* is known;
these contaminate them in ways whose *location and magnitude* are known, so a detector can
be scored against ground truth rather than against an impression.

Every function returns an `Injection` carrying the contaminated signal and a boolean mask
marking exactly which samples were touched. A test never has to infer where the artifact
was — the harness says.

Each injection is specified in units that mean something physically, so a test reads as a
statement about the signal rather than about an implementation:

* line noise — amplitude in microvolts, at 50 or 60 Hz
* clipping — the rail voltage the amplifier saturates at
* electrode pop — step height in microvolts
* dropout — samples replaced by NaN, as a monitor reports them
* flatline — the trace collapses to sensor noise, as when an electrode detaches
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_state_estimator.types import FloatArray


@dataclass(frozen=True, slots=True)
class Injection:
    """A contaminated signal and the ground truth of what was done to it."""

    signal: FloatArray
    affected: np.ndarray
    description: str

    @property
    def n_affected(self) -> int:
        return int(self.affected.sum())

    def window(self, sample_rate: float) -> tuple[float, float]:
        """First and last affected sample, in seconds. Useful for marking a plot."""
        indices = np.flatnonzero(self.affected)
        if indices.size == 0:
            return (0.0, 0.0)
        return (float(indices[0] / sample_rate), float(indices[-1] / sample_rate))


def _span(n_samples: int, sample_rate: float, start_s: float, duration_s: float) -> slice:
    start = round(start_s * sample_rate)
    stop = min(round((start_s + duration_s) * sample_rate), n_samples)
    return slice(max(start, 0), stop)


def _mask(n_samples: int, where: slice) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    mask[where] = True
    return mask


def inject_line_noise(
    signal: FloatArray,
    *,
    sample_rate: float,
    start_s: float,
    duration_s: float,
    frequency: float = 50.0,
    amplitude: float = 20.0,
) -> Injection:
    """Add a mains tone over a window.

    Mains interference is the most common contaminant in a clinical recording, and it is
    narrowband — which is what makes it detectable from a spectrum and also what makes it
    harmless to the alpha band if you simply notice it is there.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    where = _span(out.size, sample_rate, start_s, duration_s)
    times = np.arange(where.start, where.stop, dtype=np.float64) / sample_rate
    out[where] += amplitude * np.sin(2.0 * np.pi * frequency * times)
    return Injection(out, _mask(out.size, where), f"{frequency:g} Hz line noise, {amplitude:g} uV")


def inject_clipping(
    signal: FloatArray, *, sample_rate: float, start_s: float, duration_s: float, rail: float = 60.0
) -> Injection:
    """Saturate the amplifier at +/- `rail` over a window.

    Only the samples actually driven to the rail are marked affected: clipping a window
    whose signal never reaches the rail would change nothing, and the mask should say so.
    """
    # A non-finite rail turns np.clip into a NaN fill, which would silently inject a
    # *dropout* while claiming to inject clipping. That happens easily: a caller computing
    # the rail with np.percentile over a record that already contains NaN gets NaN back.
    # Fail loudly instead -- a harness that mislabels its own artifact is worse than useless,
    # because every detector scored against it is scored against the wrong ground truth.
    if not np.isfinite(rail) or rail <= 0.0:
        message = f"rail must be finite and positive, got {rail}"
        raise ValueError(message)

    out = np.array(signal, dtype=np.float64, copy=True)
    where = _span(out.size, sample_rate, start_s, duration_s)
    segment = out[where]
    clipped = np.clip(segment, -rail, rail)

    mask = np.zeros(out.size, dtype=bool)
    mask[where] = segment != clipped
    out[where] = clipped
    return Injection(out, mask, f"clipped at +/-{rail:g} uV")


def inject_electrode_pop(
    signal: FloatArray,
    *,
    sample_rate: float,
    at_s: float,
    height: float = 300.0,
    decay_s: float = 0.25,
) -> Injection:
    """A step transient that decays back — the signature of a disturbed electrode.

    Modelled as an instantaneous jump followed by an exponential return, because that is
    what the electrode-skin half-cell does when it is mechanically disturbed and then
    settles. The sharp edge is what a detector keys on.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    start = round(at_s * sample_rate)
    if not 0 <= start < out.size:
        message = f"at_s={at_s} falls outside the record"
        raise ValueError(message)

    length = min(round(decay_s * 5.0 * sample_rate), out.size - start)
    tail = np.arange(length, dtype=np.float64) / sample_rate
    out[start : start + length] += height * np.exp(-tail / decay_s)

    mask = np.zeros(out.size, dtype=bool)
    mask[start : start + length] = True
    return Injection(out, mask, f"electrode pop, {height:g} uV step")


def inject_dropout(
    signal: FloatArray, *, sample_rate: float, start_s: float, duration_s: float
) -> Injection:
    """Replace a window with NaN, the way a monitor reports lost samples.

    Note this is *not* zeros. A monitor that loses packets reports them as missing, and
    silently substituting zeros would turn a known unknown into a confident wrong answer —
    which is the whole reason this is a separate artifact from a flatline.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    where = _span(out.size, sample_rate, start_s, duration_s)
    out[where] = np.nan
    return Injection(out, _mask(out.size, where), f"{duration_s:g} s dropout")


def inject_flatline(
    signal: FloatArray,
    *,
    sample_rate: float,
    start_s: float,
    duration_s: float,
    residual: float = 0.05,
    seed: int = 0,
) -> Injection:
    """Collapse a window to sensor noise, as when an electrode detaches.

    `residual` is not zero on purpose. A detached electrode still shows amplifier noise, so
    a detector that only triggers on exact zeros would miss the real case — and a test built
    on exact zeros would never reveal that.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    where = _span(out.size, sample_rate, start_s, duration_s)
    rng = np.random.default_rng(seed)
    out[where] = rng.normal(0.0, residual, out[where].size)
    return Injection(out, _mask(out.size, where), f"{duration_s:g} s flatline")
