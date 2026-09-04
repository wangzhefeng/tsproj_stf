"""多步预测 horizon 评估。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .metrics import evaluate_metrics


def horizon_to_index(horizon: int, output_len: int) -> int:
    """把用户侧 1-based horizon 转为内部 0-based 索引。"""

    if horizon < 1 or horizon > output_len:
        raise ValueError(f"horizon {horizon} is outside 1..{output_len}")
    return horizon - 1


def evaluate_horizons(
    prediction: ArrayLike,
    targets: ArrayLike,
    observed: ArrayLike,
    horizons: Iterable[int],
) -> dict[str, dict[str, float]]:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(targets)
    observed_array = np.asarray(observed)
    if prediction_array.ndim < 2:
        raise ValueError("forecast arrays must include batch and horizon axes")
    if not (
        prediction_array.shape == target_array.shape == observed_array.shape
    ):
        raise ValueError("prediction, targets, and observed must have matching shapes")

    output_len = prediction_array.shape[1]
    return {
        f"h{horizon}": evaluate_metrics(
            prediction_array[:, horizon_to_index(horizon, output_len), ...],
            target_array[:, horizon_to_index(horizon, output_len), ...],
            observed_array[:, horizon_to_index(horizon, output_len), ...],
        )
        for horizon in horizons
    }
