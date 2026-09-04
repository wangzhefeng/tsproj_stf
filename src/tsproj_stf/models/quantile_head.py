"""保证分位点单调的可选预测头。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tsproj_stf.evaluation.quantiles import validate_quantiles


class QuantileHead(nn.Module):
    """用 base 加累计非负增量生成不交叉分位数。"""

    def __init__(self, in_features: int, quantiles: Sequence[float]) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        self.in_features = in_features
        self.quantiles = validate_quantiles(quantiles)
        self.base_projection = nn.Linear(in_features, 1)
        self.increment_projection = (
            nn.Linear(in_features, len(self.quantiles) - 1)
            if len(self.quantiles) > 1
            else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.in_features:
            raise ValueError(
                f"expected final feature axis {self.in_features}, got {features.shape[-1]}"
            )
        base = self.base_projection(features)
        if self.increment_projection is None:
            return base
        increments = F.softplus(self.increment_projection(features))
        return torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
