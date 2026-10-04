"""The streaming estimator interface.

Implements REQ-050.

Three members, which is the whole contract an estimator needs in order to run on a live
signal: consume a chunk, go back to the beginning, and say how much state is being carried.

The last one is not bookkeeping. An estimator whose state grows with the length of the
recording cannot run for the duration of a procedure, so `state_nbytes` is part of the
interface rather than a diagnostic bolted on afterwards.

The class is generic in its output type because the two implementations return different
things — the filter returns a track of states, the pipeline returns per-epoch estimates —
and flattening that into a common supertype would lose information at every call site for
no benefit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

OutputT = TypeVar("OutputT")


class StreamingEstimator(ABC, Generic[OutputT]):
    """An estimator that consumes a signal in arbitrary chunks.

    Implementations must satisfy two properties, both of which are tested behaviourally
    rather than asserted here, because neither can be enforced by a base class:

    **Chunk-invariance (REQ-051).** The output must not depend on how the input was split.
    Feeding a record in one call and in a thousand calls must give the same answer.

    **Causality (REQ-052).** Output already emitted must never change when more input
    arrives. An implementation that buffers and revises is an offline analysis wearing a
    streaming interface.
    """

    @abstractmethod
    def update(self, chunk: object) -> OutputT:
        """Consume a chunk of samples and return whatever became available."""

    @abstractmethod
    def reset(self) -> None:
        """Return to the state the estimator was constructed in."""

    @property
    @abstractmethod
    def state_nbytes(self) -> int:
        """Bytes of state retained between calls. Must be constant in record length."""
