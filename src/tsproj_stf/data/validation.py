"""时间轴约束。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_regular_timestamps(timestamps: NDArray[np.generic]) -> np.generic:
    """校验严格递增且等间隔的时间轴，并返回采样间隔。"""

    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError(f"timestamps must be 1-dimensional, got {values.shape}")
    if len(values) < 2:
        raise ValueError("timestamps must contain at least two values")

    differences = np.diff(values)
    if not np.all(differences > 0) or not np.all(differences == differences[0]):
        raise ValueError("timestamps must be strictly increasing and regularly spaced")
    return differences[0]
