"""The composed streaming estimator: raw samples in, tracked brain-state feature out.

Implements REQ-001, REQ-002, REQ-051 … REQ-055.

    samples -> epoch buffer -> multitaper PSD -> log band power -> Kalman filter -> estimate

Each stage is already verified on its own. This module is the wiring, and its tests are
about the wiring: that chunk boundaries do not matter, that nothing already emitted ever
changes, and that the warm-up is flagged.

**On preprocessing.** There is deliberately no band-pass filter here. The mean is removed
inside the spectral estimate, which is what the slope and leakage requirements need, and
nothing more is done. A zero-phase filter — the usual choice offline, because it has no
group delay — works by filtering forwards and then backwards, which is look-ahead and is
forbidden on a causal path. A causal IIR filter would be legitimate but introduces a
frequency-dependent group delay that would then have to be characterised and reported.
Neither is needed for what is estimated here, so neither is present. See
docs/design-decisions.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from eeg_state_estimator.base import StreamingEstimator
from eeg_state_estimator.quality import QualityFlags, assess_epoch
from eeg_state_estimator.spectral import MultitaperConfig, band_power, multitaper_psd
from eeg_state_estimator.statespace.kalman import RandomWalkModel, ScalarKalmanFilter
from eeg_state_estimator.types import FloatArray

#: Status of an estimate.
#:
#: * ``ok`` -- a settled estimate from usable signal.
#: * ``warmup`` -- the filter has not converged yet; the value is present but is not a
#:   settled estimate.
#: * ``no_signal`` -- the epoch carried no usable signal. This is a **distinct state, not a
#:   value on the measurement scale**, and that distinction is the point. A disconnected
#:   electrode produces a flat trace with almost no band power, and low band power is also
#:   what the deepest physiological state looks like. An estimator that reports the two the
#:   same way claims maximum depth precisely when it has no input.
EstimateStatus = Literal[
    "warmup",
    "ok",
    "no_signal",
    "dropped_samples",
    "flatline",
    "clipping",
    "electrode_pop",
    "line_noise",
]


@dataclass(frozen=True, slots=True)
class EpochEstimate:
    """One epoch's worth of output.

    `valid` is the single boolean a caller should branch on. The numbers are present even
    when it is false — flagged, not suppressed — so that a warm-up can be plotted, but
    they cannot be mistaken for a converged estimate.
    """

    epoch_index: int
    start_time: float
    end_time: float
    observed: float
    state: float
    variance: float
    lower: float
    upper: float
    valid: bool
    status: EstimateStatus
    #: The full quality assessment for this epoch, so a caller can inspect *why* rather
    #: than only *whether*. Present on every estimate, including usable ones.
    quality: QualityFlags


class EpochPipeline(StreamingEstimator[list[EpochEstimate]]):
    """Track the log power in a frequency band, with an uncertainty on every estimate.

    Epochs are non-overlapping. Overlapping windows are the conventional choice on a
    clinical monitor, because they raise the output rate without shortening the analysis
    window, and they are a natural extension of the buffering here. They are not included
    because nothing in the requirements needs them and a half-built version would be worse
    than their absence.

    The tracked quantity is the **log** band power. Band power is positive and varies
    multiplicatively, so its logarithm is the quantity for which a Gaussian random walk is
    a defensible model — tracking the raw power would put a symmetric Gaussian prior on a
    strictly positive quantity and produce credible intervals extending below zero.
    """

    def __init__(
        self,
        *,
        sample_rate: float,
        band: tuple[float, float] = (8.0, 12.0),
        epoch_seconds: float = 2.0,
        config: MultitaperConfig | None = None,
        model: RandomWalkModel | None = None,
        initial_variance: float = 1.0e6,
        warmup_tolerance: float = 0.05,
        power_floor: float = 1e-12,
        line_frequency: float = 50.0,
    ) -> None:
        if sample_rate <= 0.0:
            message = f"sample_rate must be positive, got {sample_rate}"
            raise ValueError(message)
        if band[1] <= band[0]:
            message = f"band must be (low, high) with high > low, got {band}"
            raise ValueError(message)
        if warmup_tolerance <= 0.0:
            message = f"warmup_tolerance must be positive, got {warmup_tolerance}"
            raise ValueError(message)

        self._config = config or MultitaperConfig()
        self._epoch_samples = round(epoch_seconds * sample_rate)
        if self._epoch_samples <= 2 * self._config.time_bandwidth:
            message = (
                f"epoch of {epoch_seconds} s is {self._epoch_samples} samples, too short for "
                f"time_bandwidth={self._config.time_bandwidth}; it must exceed "
                f"2*NW = {2 * self._config.time_bandwidth}"
            )
            raise ValueError(message)

        self._sample_rate = sample_rate
        self._band = band
        self._model = model or RandomWalkModel(process_variance=0.05, observation_variance=0.1)
        self._initial_variance = initial_variance
        self._warmup_tolerance = warmup_tolerance
        self._power_floor = power_floor
        self._line_frequency = line_frequency

        # The convergence target is computable in advance: the variance recursion does not
        # involve the data, so whether the filter has settled is knowable from the model
        # alone. That makes the warm-up criterion self-calibrating -- change Q or R and it
        # adapts, where a hand-picked epoch count would quietly become wrong.
        self._steady_state_variance = self._model.steady_state_posterior_variance()

        self._filter = ScalarKalmanFilter(
            self._model, initial_mean=0.0, initial_variance=initial_variance
        )
        self._buffer: FloatArray = np.empty(0, dtype=np.float64)
        self._epoch_index = 0
        self._seen_first_epoch = False

    @property
    def epoch_samples(self) -> int:
        """Samples per analysis epoch."""
        return self._epoch_samples

    @property
    def buffered_samples(self) -> int:
        """Samples currently held awaiting a complete epoch. Always below `epoch_samples`."""
        return int(self._buffer.size)

    @property
    def state_nbytes(self) -> int:
        """Bytes retained between calls: the filter's state plus the partial epoch.

        Bounded by one epoch however long the recording runs, because the buffer is drained
        as epochs complete. The transient during a single `update()` depends on the size of
        the chunk handed in, which is the caller's choice and not a function of how much
        has been streamed so far.
        """
        return self._filter.state_nbytes + self._buffer.nbytes + 8

    def reset(self) -> None:
        self._filter.reset()
        self._buffer = np.empty(0, dtype=np.float64)
        self._epoch_index = 0
        self._seen_first_epoch = False

    def update(self, chunk: object) -> list[EpochEstimate]:
        """Consume samples and return an estimate for each epoch that completed."""
        samples = np.asarray(chunk, dtype=np.float64)
        if samples.ndim != 1:
            message = f"chunk must be one-dimensional, got shape {samples.shape}"
            raise ValueError(message)
        # Non-finite samples are NOT rejected here. A monitor reports lost packets as
        # NaN, and refusing the chunk would mean a dropout crashes the estimator rather
        # than being handled. The numerical core stays strict -- multitaper_psd still
        # raises on non-finite input -- and this is the layer that absorbs real-world
        # data, detecting the dropout and declining to estimate on that epoch (REQ-064).

        self._buffer = np.concatenate([self._buffer, samples])

        estimates: list[EpochEstimate] = []
        while self._buffer.size >= self._epoch_samples:
            epoch = self._buffer[: self._epoch_samples]
            estimates.append(self._process_epoch(epoch))
            # Drop the consumed samples. This is what keeps the retained state bounded,
            # and it is also why nothing already emitted can be revised: the samples an
            # earlier epoch was computed from are gone.
            self._buffer = self._buffer[self._epoch_samples :]
        return estimates

    def _process_epoch(self, epoch: FloatArray) -> EpochEstimate:
        # Quality is assessed before anything numerical is attempted. A contaminated epoch
        # must not reach the spectral estimator at all: a NaN would poison the transform,
        # and an electrode pop would produce a perfectly well-formed spectrum of an
        # artifact, which is worse because nothing would look wrong.
        flags = assess_epoch(
            epoch, sample_rate=self._sample_rate, line_frequency=self._line_frequency
        )
        if not flags.usable:
            return self._held_estimate(self._epoch_index, flags)

        spectrum = multitaper_psd(epoch, sample_rate=self._sample_rate, config=self._config)
        power = band_power(spectrum, self._band[0], self._band[1])

        index = self._epoch_index
        if power <= self._power_floor:
            return self._no_signal_estimate(index, power, flags)

        observed = math.log(power)
        step = self._filter.step(observed)
        lower, upper = step.credible_interval(0.95)

        converged = step.variance <= self._steady_state_variance * (1.0 + self._warmup_tolerance)
        # The variance decreases monotonically to its fixed point from a diffuse prior, so
        # once this is true it stays true: validity latches rather than flickering.
        valid = self._seen_first_epoch and converged
        self._seen_first_epoch = True

        self._epoch_index += 1
        epoch_seconds = self._epoch_samples / self._sample_rate

        return EpochEstimate(
            epoch_index=index,
            start_time=index * epoch_seconds,
            end_time=(index + 1) * epoch_seconds,
            observed=observed,
            state=step.mean,
            variance=step.variance,
            lower=lower,
            upper=upper,
            valid=valid,
            status="ok" if valid else "warmup",
            quality=flags,
        )

    def _held_estimate(self, index: int, flags: QualityFlags) -> EpochEstimate:
        """Report a contaminated epoch without advancing the filter (REQ-066).

        Holding rather than advancing is the part that is easy to omit and expensive to
        get wrong. An estimator that flags a fault and then feeds the contaminated
        observation to its filter is still wrong, and stays wrong for many epochs after the
        fault has cleared -- a transient becomes persistent. Because the state is never
        corrupted, recovery needs no unwinding: the next usable epoch simply continues
        (REQ-067).
        """
        self._epoch_index += 1
        epoch_seconds = self._epoch_samples / self._sample_rate
        reason = flags.reason or "no_signal"
        return EpochEstimate(
            epoch_index=index,
            start_time=index * epoch_seconds,
            end_time=(index + 1) * epoch_seconds,
            observed=math.nan,
            state=self._filter.mean,
            variance=self._filter.variance,
            lower=math.nan,
            upper=math.nan,
            valid=False,
            status=reason,  # type: ignore[arg-type]
            quality=flags,
        )

    def _no_signal_estimate(self, index: int, power: float, flags: QualityFlags) -> EpochEstimate:
        """Report an epoch that carried no usable signal, without advancing the filter.

        Flagging alone would not be enough. If the floor value were fed to the filter as
        an observation, the state would walk down to it and stay there, so the estimate
        would remain wrong for many epochs after the signal came back -- a transient fault
        would become a persistent one. Instead the filter is left untouched and its last
        state is reported, held and clearly labelled.

        Note the narrowness of what this detects: an epoch with *no* power in the band,
        which covers a disconnected electrode or a flatline. It is not artifact detection.
        Line noise, muscle activity, electrode pop and amplifier saturation all produce
        plenty of band power and are not caught here.
        """
        self._epoch_index += 1
        epoch_seconds = self._epoch_samples / self._sample_rate
        return EpochEstimate(
            epoch_index=index,
            start_time=index * epoch_seconds,
            end_time=(index + 1) * epoch_seconds,
            observed=math.log(max(power, self._power_floor)),
            state=self._filter.mean,
            variance=self._filter.variance,
            lower=math.nan,
            upper=math.nan,
            valid=False,
            status="no_signal",
            quality=flags,
        )
