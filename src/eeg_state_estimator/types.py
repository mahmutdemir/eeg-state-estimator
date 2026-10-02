"""Shared type aliases.

Collected in one module so the array contract is stated once. Every public function
in this package takes and returns float64 arrays; mixing in float32 would change the
numerical tolerances the requirements are stated against, so the alias is explicit
about the dtype rather than using a bare `np.ndarray`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: A one-dimensional array of 64-bit floats — a signal, a frequency axis, or a PSD.
FloatArray = NDArray[np.float64]
