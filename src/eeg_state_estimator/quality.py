"""Per-epoch signal quality assessment.

Implements REQ-060 … REQ-065.

Five detectors for the contaminants that are *loud* — loud enough to change a spectral
estimate by more than the quantity being measured. Each is deliberately simple, because a
quality detector that is itself hard to reason about moves the problem rather than solving
it.

**What this is not.** It is not general artifact rejection. A low-amplitude contaminant
that resembles EEG — a slow drift inside the analysis band, muscle activity at a level
comparable to the signal — is not detected and is not meant to be. The honest claim is
narrow: these five faults are caught, and the estimator refuses to produce a confident
number while one is present.

**Thresholds are expressed in robust units.** Every amplitude test is scaled by the median
absolute deviation of the epoch rather than its standard deviation, because the standard
deviation of an epoch containing an electrode pop is dominated by the pop — the statistic
you are using to find the artifact is corrupted by the artifact. MAD has a 50% breakdown
point, so a transient affecting a minority of samples leaves it essentially unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_state_estimator.types import FloatArray

#: Scaling that makes the median absolute deviation match the standard deviation for
#: Gaussian data, so thresholds can be read as "multiples of sigma" without implying the
#: non-robust estimator was used.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class QualityFlags:
    """The outcome of assessing one epoch.

    The individual flags are independent and inspectable; `usable` is the single verdict a
    caller may branch on without knowing which detectors exist. `reason` names the first
    disqualifying fault, in a fixed priority order, so a status string is deterministic
    rather than dependent on dictionary iteration.
    """

    dropped_samples: bool = False
    flatline: bool = False
    clipping: bool = False
    electrode_pop: bool = False
    line_noise: bool = False

    #: Power at the mains frequency, in signal units squared. Reported whether or not the
    #: flag is set, so a caller can watch it trend before it crosses the threshold.
    line_power: float = 0.0
    #: Robust amplitude scale of the epoch, in signal units.
    robust_scale: float = 0.0

    @property
    def any_fault(self) -> bool:
        return bool(
            self.dropped_samples
            or self.flatline
            or self.clipping
            or self.electrode_pop
            or self.line_noise
        )

    @property
    def usable(self) -> bool:
        """False if any disqualifying fault is present."""
        return not self.any_fault

    @property
    def reason(self) -> str | None:
        """The first fault present, in priority order, or None if the epoch is clean.

        The order runs from "there is no signal here" to "the signal is contaminated",
        because that is the order in which a reader wants to be told: a dropout makes every
        other measurement on the epoch meaningless, so reporting line noise on an epoch
        that is half NaN would be answering the wrong question.
        """
        for flag, name in (
            (self.dropped_samples, "dropped_samples"),
            (self.flatline, "flatline"),
            (self.clipping, "clipping"),
            (self.electrode_pop, "electrode_pop"),
            (self.line_noise, "line_noise"),
        ):
            if flag:
                return name
        return None


def robust_scale(epoch: FloatArray) -> float:
    """Median absolute deviation, scaled to be comparable with a standard deviation.

    Used in place of `np.std` throughout this module. On an epoch containing a 300 µV
    electrode pop the standard deviation is dominated by the pop itself, so a threshold
    expressed in standard deviations silently rescales to accommodate the very artifact it
    is meant to catch.
    """
    finite = epoch[np.isfinite(epoch)]
    if finite.size == 0:
        return 0.0
    return float(np.median(np.abs(finite - np.median(finite))) * _MAD_TO_SIGMA)


def _line_power(epoch: FloatArray, sample_rate: float, line_frequency: float) -> float:
    """Power within ±1 Hz of the mains frequency, by direct projection.

    A full multitaper estimate is unnecessary here and would couple quality assessment to
    the spectral configuration. Projecting onto a sine and cosine at the mains frequency
    gives the power in that component directly, which is both cheaper and easier to defend:
    it is a single inner product, not an estimator with its own bias and variance.
    """
    finite = epoch[np.isfinite(epoch)]
    if finite.size < 2:
        return 0.0
    if line_frequency >= sample_rate / 2.0:
        # Above Nyquist the mains tone is not representable; report nothing rather than
        # alias and produce a confident wrong answer.
        return 0.0

    times = np.arange(finite.size, dtype=np.float64) / sample_rate
    angle = 2.0 * np.pi * line_frequency * times
    cosine = float(np.mean(finite * np.cos(angle)))
    sine = float(np.mean(finite * np.sin(angle)))
    # Amplitude of the fitted tone is 2*sqrt(c^2 + s^2); its mean-square power is half the
    # square of that amplitude.
    return 2.0 * (cosine**2 + sine**2)


def assess_epoch(
    epoch: FloatArray,
    *,
    sample_rate: float,
    line_frequency: float = 50.0,
    flatline_scale: float = 0.5,
    clipping_fraction: float = 0.005,
    pop_multiple: float = 8.0,
    line_noise_ratio: float = 0.25,
) -> QualityFlags:
    """Assess one epoch and return its quality flags.

    Parameters
    ----------
    line_frequency
        Mains frequency, 50 Hz in most of the world and 60 Hz in North America. There is no
        reliable way to infer it from a short epoch, so it is configuration, not detection.
    flatline_scale
        Robust amplitude below which the epoch is considered to carry no signal, in signal
        units. The default suits microvolt-scale EEG, where even a quiet cortex is well
        above 0.5 µV.
    clipping_fraction
        Fraction of samples sitting at the extreme value before saturation is declared.
        Kept small because clipping is never benign: a handful of rail-bound samples means
        the amplifier range is wrong and every amplitude on the epoch is suspect.
    pop_multiple
        Sample-to-sample jump, in robust scales, that counts as a step transient. Gaussian
        EEG essentially never produces an 8-sigma single-sample jump, so this is specific
        without needing to be clever.
    line_noise_ratio
        Mains power relative to the epoch's total robust power before the flag is set.
    """
    samples = np.asarray(epoch, dtype=np.float64)
    if samples.ndim != 1:
        message = f"epoch must be one-dimensional, got shape {samples.shape}"
        raise ValueError(message)
    if samples.size == 0:
        message = "epoch must not be empty"
        raise ValueError(message)

    # Dropped samples first: every other measurement on this epoch is conditional on the
    # data actually being there.
    dropped = bool(np.any(~np.isfinite(samples)))
    scale = robust_scale(samples)
    finite = samples[np.isfinite(samples)]

    flatline = bool(scale < flatline_scale)

    # Saturation: many samples resting at exactly one value. That repetition is the actual
    # signature of a rail -- a continuous signal does not revisit a single value hundreds of
    # times -- and the rail itself is a property of the amplifier, which this function does
    # not know.
    #
    # An earlier version compared each sample to the epoch's own maximum. That works for a
    # fully clipped epoch and fails for a partially clipped one: if the unclipped half
    # happens to reach higher than the rail, the maximum is an ordinary sample and nothing
    # is found. Counting the most-repeated large-amplitude value has no such blind spot,
    # because the rail is defined by how often it recurs rather than by how large it is.
    clipping = False
    if finite.size and not flatline:
        large = np.abs(finite) >= np.percentile(np.abs(finite), 75)
        if np.any(large):
            _, counts = np.unique(finite[large], return_counts=True)
            repeated = int(counts.max())
            clipping = bool(repeated / finite.size >= clipping_fraction and repeated >= 3)

    # Electrode pop: a single-sample jump far outside what the epoch's own scale allows.
    electrode_pop = False
    if finite.size > 1 and scale > 0.0:
        electrode_pop = bool(np.max(np.abs(np.diff(finite))) > pop_multiple * scale)

    line_power = _line_power(samples, sample_rate, line_frequency)
    total_power = scale**2
    line_noise = bool(total_power > 0.0 and line_power / total_power >= line_noise_ratio)

    return QualityFlags(
        dropped_samples=dropped,
        flatline=flatline,
        clipping=clipping,
        electrode_pop=electrode_pop,
        line_noise=line_noise,
        line_power=line_power,
        robust_scale=scale,
    )
