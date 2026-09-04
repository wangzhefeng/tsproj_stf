"""带显式观测掩码的预测指标。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricFunction = Callable[[ArrayLike, ArrayLike, ArrayLike], float]


def _valid_arrays(
    prediction: ArrayLike,
    targets: ArrayLike,
    observed: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=bool)
    if not (
        prediction_array.shape == target_array.shape == observed_array.shape
    ):
        raise ValueError(
            "prediction, targets, and observed must have matching shapes, got "
            f"{prediction_array.shape}, {target_array.shape}, {observed_array.shape}"
        )
    if not np.any(observed_array):
        raise ValueError("no valid targets remain after applying observed mask")
    if not np.all(np.isfinite(prediction_array[observed_array])) or not np.all(
        np.isfinite(target_array[observed_array])
    ):
        raise ValueError("observed prediction and targets contain non-finite values")
    return prediction_array[observed_array], target_array[observed_array]


def masked_mae(prediction: ArrayLike, targets: ArrayLike, observed: ArrayLike) -> float:
    valid_prediction, valid_targets = _valid_arrays(prediction, targets, observed)
    return float(np.mean(np.abs(valid_prediction - valid_targets)))


def masked_rmse(prediction: ArrayLike, targets: ArrayLike, observed: ArrayLike) -> float:
    valid_prediction, valid_targets = _valid_arrays(prediction, targets, observed)
    return float(np.sqrt(np.mean(np.square(valid_prediction - valid_targets))))


def masked_mape(
    prediction: ArrayLike,
    targets: ArrayLike,
    observed: ArrayLike,
    epsilon: float = 5e-5,
) -> float:
    valid_prediction, valid_targets = _valid_arrays(prediction, targets, observed)
    nonzero = np.abs(valid_targets) > epsilon
    if not np.any(nonzero):
        raise ValueError("no valid targets remain after excluding near-zero MAPE denominators")
    percentage_errors = np.abs(
        (valid_prediction[nonzero] - valid_targets[nonzero]) / valid_targets[nonzero]
    )
    return float(np.mean(percentage_errors))


def masked_wape(
    prediction: ArrayLike,
    targets: ArrayLike,
    observed: ArrayLike,
    epsilon: float = 5e-5,
) -> float:
    valid_prediction, valid_targets = _valid_arrays(prediction, targets, observed)
    denominator = float(np.sum(np.abs(valid_targets)))
    if denominator <= epsilon:
        raise ValueError("WAPE denominator is zero after applying observed mask")
    return float(np.sum(np.abs(valid_prediction - valid_targets)) / denominator)


def evaluate_metrics(
    prediction: ArrayLike,
    targets: ArrayLike,
    observed: ArrayLike,
) -> dict[str, float]:
    """计算项目标准确定性指标，比例指标返回 0～1 尺度。"""

    return {
        "MAE": masked_mae(prediction, targets, observed),
        "RMSE": masked_rmse(prediction, targets, observed),
        "MAPE": masked_mape(prediction, targets, observed),
        "WAPE": masked_wape(prediction, targets, observed),
    }
