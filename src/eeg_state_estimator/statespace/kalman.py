"""Scalar Kalman filter with a random-walk process model.

Implements REQ-001, REQ-002, REQ-030, REQ-031, REQ-032, REQ-036.

The model is

    x[k] = A x[k-1] + w,   w ~ N(0, Q)
    y[k] = C x[k]   + v,   v ~ N(0, R)

and the filter is the standard predict/update recursion, which for a linear-Gaussian
model is exactly recursive Bayes: the prior is Gaussian, the likelihood is Gaussian, so
the posterior is Gaussian and only its mean and variance need to be propagated.

The per-sample kernel is `step()` and the chunk-level method is `update()`. That naming
is chosen now rather than later so that a streaming interface — whose method is
`update(chunk)` — is a natural extension rather than a rename of an established API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eeg_state_estimator.base import StreamingEstimator
from eeg_state_estimator.types import FloatArray


@dataclass(frozen=True, slots=True)
class RandomWalkModel:
    """The state-space model parameters.

    `transition` defaults to 1.0, the random walk: the state has no preferred value and
    simply diffuses. It is a field rather than a hard-coded constant so that the `Q = 0`
    static-state oracle and an AR(1) process share one code path.

    `process_variance` (Q) is how fast the truth is allowed to change; `observation_variance`
    (R) is how noisy the sensor is. Their *ratio* sets the filter's character — large Q/R
    tracks quickly and stays jumpy, small Q/R is smooth and sluggish. It is the
    bias-variance trade in different clothes.
    """

    process_variance: float
    observation_variance: float
    transition: float = 1.0
    observation_gain: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.process_variance) or self.process_variance < 0.0:
            message = (
                f"process_variance must be finite and non-negative, got {self.process_variance}"
            )
            raise ValueError(message)
        if not math.isfinite(self.observation_variance) or self.observation_variance <= 0.0:
            message = (
                f"observation_variance must be finite and positive, got {self.observation_variance}"
            )
            raise ValueError(message)

    def steady_state_posterior_variance(self) -> float:
        """The fixed point of the Riccati recursion, in closed form.

        The variance recursion does not involve the data at all, so a filter's uncertainty
        schedule is knowable before a single sample is collected. For `A = C = 1` the prior
        fixed point solves `p**2 - Q p - Q R = 0`, giving
        `p = (Q + sqrt(Q**2 + 4QR)) / 2`, and the posterior fixed point is `p R / (p + R)`.
        """
        if not (math.isclose(self.transition, 1.0) and math.isclose(self.observation_gain, 1.0)):
            message = "the closed form is derived for transition = observation_gain = 1"
            raise ValueError(message)
        q, r = self.process_variance, self.observation_variance
        prior = (q + math.sqrt(q * q + 4.0 * q * r)) / 2.0
        return prior * r / (prior + r)


@dataclass(frozen=True, slots=True)
class KalmanStep:
    """The result of one filter update.

    Frozen so that a caller cannot mutate it after the filter has moved on, which is the
    worst class of bug in stateful numerical code. The innovation and its variance are
    included unconditionally rather than as optional diagnostics: they are the only
    observables on which filter correctness can be assessed without ground truth, which
    is the situation on any real recording.
    """

    mean: float
    variance: float
    innovation: float
    innovation_variance: float
    gain: float

    @property
    def normalized_innovation(self) -> float:
        """`nu / sqrt(S)` — unit variance under a correctly specified model."""
        return self.innovation / math.sqrt(self.innovation_variance)

    def credible_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Symmetric Gaussian credible interval around the posterior mean."""
        if not 0.0 < confidence < 1.0:
            message = f"confidence must be strictly between 0 and 1, got {confidence}"
            raise ValueError(message)
        # Two-sided normal quantile without pulling in scipy for one number.
        z = math.sqrt(2.0) * _inverse_erf(confidence)
        half_width = z * math.sqrt(self.variance)
        return (self.mean - half_width, self.mean + half_width)


def _inverse_erf(confidence: float) -> float:
    """Inverse error function at `confidence`, by bisection.

    Bisection rather than a rational approximation because this is called once per
    interval, never in a hot loop, and fifty halvings of a bracketed monotone function is
    something a reader can check in their head. Accurate to about 1e-15.
    """
    low, high = 0.0, 6.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if math.erf(middle) < confidence:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True, slots=True)
class KalmanTrack:
    """A filtered series: one entry per observation."""

    mean: FloatArray
    variance: FloatArray
    innovation: FloatArray
    innovation_variance: FloatArray

    def __len__(self) -> int:
        return int(self.mean.size)

    @property
    def normalized_innovation(self) -> FloatArray:
        return np.asarray(self.innovation / np.sqrt(self.innovation_variance), dtype=np.float64)

    def credible_interval(self, confidence: float = 0.95) -> tuple[FloatArray, FloatArray]:
        z = math.sqrt(2.0) * _inverse_erf(confidence)
        half_width = z * np.sqrt(self.variance)
        lower = np.asarray(self.mean - half_width, dtype=np.float64)
        upper = np.asarray(self.mean + half_width, dtype=np.float64)
        return lower, upper


class ScalarKalmanFilter(StreamingEstimator[KalmanTrack]):
    """A stateful scalar Kalman filter.

    State is genuinely carried between calls — the posterior mean and variance — so this
    is a class rather than a function. `step()` consumes one observation; `update()`
    consumes a chunk and returns a `KalmanTrack`.

    It satisfies `StreamingEstimator` without adaptation, which is the point of having
    named the kernel `step()` and the chunk method `update()` from the beginning rather
    than renaming an established API once streaming arrived.
    """

    #: The numerical state carried between updates. Declared rather than introspected so
    #: `state_nbytes` means something specific — see that property.
    _STATE_FIELDS = ("mean", "variance", "n_updates")

    def __init__(
        self,
        model: RandomWalkModel,
        *,
        initial_mean: float = 0.0,
        initial_variance: float = 1.0e6,
    ) -> None:
        if not math.isfinite(initial_variance) or initial_variance <= 0.0:
            message = f"initial_variance must be finite and positive, got {initial_variance}"
            raise ValueError(message)
        self._model = model
        self._initial_mean = float(initial_mean)
        self._initial_variance = float(initial_variance)
        self._mean = float(initial_mean)
        self._variance = float(initial_variance)
        self._n_updates = 0

    @property
    def model(self) -> RandomWalkModel:
        return self._model

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def state_nbytes(self) -> int:
        """Bytes of numerical state carried between updates — constant in record length.

        Computed from the declared state rather than with `sys.getsizeof`, which reports
        CPython object overhead (24 bytes for a bare float) and is both misleading and
        unstable across versions. The point of this property is the bounded-memory
        property, so it reports the numerical state and nothing else.
        """
        return len(self._STATE_FIELDS) * 8

    def reset(self) -> None:
        """Return to the constructor's initial state."""
        self._mean = self._initial_mean
        self._variance = self._initial_variance
        self._n_updates = 0

    def step(self, observation: float) -> KalmanStep:
        """Advance one observation. This is the only place filtering arithmetic lives."""
        a = self._model.transition
        c = self._model.observation_gain
        q = self._model.process_variance
        r = self._model.observation_variance

        # Predict: the state moves and the uncertainty grows.
        prior_mean = a * self._mean
        prior_variance = a * a * self._variance + q

        # Innovation: the part of this observation that could not be predicted.
        innovation = float(observation) - c * prior_mean
        innovation_variance = c * c * prior_variance + r

        # Update: a precision-weighted compromise between prior and measurement.
        gain = prior_variance * c / innovation_variance
        posterior_mean = prior_mean + gain * innovation

        # The posterior variance is computed as P * R / S, NOT as (1 - K*C) * P.
        # The two are algebraically identical -- substituting K = P*C/S into the Joseph
        # form and simplifying gives exactly this -- but the second forms (1 - K*C) as a
        # difference, and K -> 1 is precisely what a diffuse prior produces. With
        # P = 1e16 and R = 2, K differs from 1 only in the sixteenth significant digit,
        # so the subtraction cancels away almost every bit of information and returns
        # 2.22 where the answer is 2.00. This form is a quotient of strictly positive
        # quantities: it cannot cancel and cannot return a negative variance.
        posterior_variance = prior_variance * r / innovation_variance

        self._mean = posterior_mean
        self._variance = posterior_variance
        self._n_updates += 1

        return KalmanStep(
            mean=posterior_mean,
            variance=posterior_variance,
            innovation=innovation,
            innovation_variance=innovation_variance,
            gain=gain,
        )

    def update(self, chunk: object) -> KalmanTrack:
        """Advance over a chunk of observations, returning one entry per sample."""
        samples = np.asarray(chunk, dtype=np.float64)
        if samples.ndim != 1:
            message = f"chunk must be one-dimensional, got shape {samples.shape}"
            raise ValueError(message)
        if not np.all(np.isfinite(samples)):
            message = "chunk must be finite; found NaN or inf"
            raise ValueError(message)

        n = samples.size
        mean = np.empty(n, dtype=np.float64)
        variance = np.empty(n, dtype=np.float64)
        innovation = np.empty(n, dtype=np.float64)
        innovation_variance = np.empty(n, dtype=np.float64)

        for index in range(n):
            result = self.step(float(samples[index]))
            mean[index] = result.mean
            variance[index] = result.variance
            innovation[index] = result.innovation
            innovation_variance[index] = result.innovation_variance

        return KalmanTrack(
            mean=mean,
            variance=variance,
            innovation=innovation,
            innovation_variance=innovation_variance,
        )


def filter_series(
    observations: object,
    model: RandomWalkModel,
    *,
    initial_mean: float = 0.0,
    initial_variance: float = 1.0e6,
) -> KalmanTrack:
    """Filter a whole record.

    Constructs a filter and calls `update()` once. It deliberately contains **no
    filtering arithmetic of its own**, so there is a single implementation and the batch
    and streaming paths cannot drift apart — which is what makes an online/offline
    equivalence check trivially true rather than something to chase.
    """
    filt = ScalarKalmanFilter(model, initial_mean=initial_mean, initial_variance=initial_variance)
    return filt.update(observations)
