"""仅使用训练集有效观测拟合的 Z-score。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ZScoreStats:
    """按节点/特征保存、保留广播轴的统计量。"""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]


class ZScoreScaler:
    """沿时间轴拟合的 Z-score scaler。"""

    def __init__(self, stats: ZScoreStats) -> None:
        self.stats = stats

    @classmethod
    def fit(
        cls,
        values: NDArray[np.floating],
        observed: NDArray[np.bool_],
    ) -> ZScoreScaler:
        value_array = np.asarray(values, dtype=np.float32)
        observed_array = np.asarray(observed, dtype=bool)
        if observed_array.shape != value_array.shape:
            raise ValueError(
                f"observed shape {observed_array.shape} must equal values shape {value_array.shape}"
            )
        if value_array.ndim < 2:
            raise ValueError("values must contain a time axis and at least one value axis")

        counts = observed_array.sum(axis=0, keepdims=True)
        if np.any(counts == 0):
            missing_indices = np.argwhere(counts[0] == 0).tolist()
            raise ValueError(f"no observed training values for series at indices {missing_indices}")

        masked_values = np.where(observed_array, value_array, 0.0)
        mean = masked_values.sum(axis=0, keepdims=True) / counts
        centered = np.where(observed_array, value_array - mean, 0.0)
        variance = np.square(centered).sum(axis=0, keepdims=True) / counts
        std = np.sqrt(variance)
        std = np.where(std == 0, 1.0, std)

        return cls(
            ZScoreStats(
                mean=mean.astype(np.float32),
                std=std.astype(np.float32),
            )
        )

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.float32]:
        value_array = np.asarray(values, dtype=np.float32)
        return ((value_array - self.stats.mean) / self.stats.std).astype(np.float32)

    def inverse_transform(self, values: NDArray[np.floating]) -> NDArray[np.float32]:
        value_array = np.asarray(values, dtype=np.float32)
        return (value_array * self.stats.std + self.stats.mean).astype(np.float32)
