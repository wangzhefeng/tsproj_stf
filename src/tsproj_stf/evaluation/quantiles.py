"""分位数预测验证与损失。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch


def validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """验证分位点位于开区间且严格递增。"""

    values = tuple(float(value) for value in quantiles)
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("quantiles must be non-empty and inside (0, 1)")
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError("quantiles must be strictly increasing")
    return values


def pinball_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    *,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """按显式 target mask 计算全部分位点的平均 pinball loss。"""

    levels = validate_quantiles(quantiles)
    if prediction.shape != targets.shape + (len(levels),):
        raise ValueError("prediction shape must equal targets shape plus quantile axis")
    if observed.shape != targets.shape:
        raise ValueError("observed shape must equal targets shape")
    valid = observed.to(dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("no valid targets remain after applying observed mask")
    if not torch.isfinite(targets[valid]).all():
        raise ValueError("observed targets contain non-finite values")
    expanded_valid = valid.unsqueeze(-1).expand_as(prediction)
    if not torch.isfinite(prediction[expanded_valid]).all():
        raise ValueError("observed predictions contain non-finite values")
    level_tensor = prediction.new_tensor(levels)
    errors = targets.unsqueeze(-1) - prediction
    losses = torch.maximum(level_tensor * errors, (level_tensor - 1.0) * errors)
    return losses[expanded_valid].mean()
