"""最后有效值 Persistence 基线。"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def persistence_forecast(
    inputs: NDArray[np.floating],
    observed: NDArray[np.bool_],
    output_len: int,
    fallback_value: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """将每条序列的最后有效历史值重复到全部未来步。"""

    if output_len <= 0:
        raise ValueError("output_len must be positive")
    value_array = np.asarray(inputs, dtype=np.float32)
    observed_array = np.asarray(observed, dtype=bool)
    if value_array.shape != observed_array.shape:
        raise ValueError(
            f"observed shape {observed_array.shape} must equal inputs shape {value_array.shape}"
        )
    if value_array.ndim < 3:
        raise ValueError("inputs must have [batch, time, series...] axes")

    batch_size, input_len = value_array.shape[:2]
    series_shape = value_array.shape[2:]
    flat_values = value_array.reshape(batch_size, input_len, -1)
    flat_observed = observed_array.reshape(batch_size, input_len, -1)
    valid_history = flat_observed.any(axis=1)
    if not np.all(valid_history) and fallback_value is None:
        missing = np.argwhere(~valid_history).tolist()
        preview = missing[:10]
        raise ValueError(
            "no observed input history for "
            f"{len(missing)} batch/series pairs; first indices={preview}"
        )

    last_from_end = np.argmax(flat_observed[:, ::-1, :], axis=1)
    last_indices = input_len - 1 - last_from_end
    last_values = np.take_along_axis(flat_values, last_indices[:, None, :], axis=1)
    if fallback_value is not None:
        try:
            fallback = np.broadcast_to(
                np.asarray(fallback_value, dtype=np.float32), series_shape
            ).reshape(1, 1, -1)
        except ValueError as error:
            raise ValueError(
                f"fallback_value shape must broadcast to series shape {series_shape}"
            ) from error
        last_values = np.where(valid_history[:, None, :], last_values, fallback)
    last_values = last_values.reshape(batch_size, 1, *series_shape)
    return np.repeat(last_values, output_len, axis=1).astype(np.float32, copy=False)
