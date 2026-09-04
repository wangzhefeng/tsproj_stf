"""Graph WaveNet 时间与图卷积层。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def receptive_field(kernel_size: int, dilations: tuple[int, ...]) -> int:
    """计算串联无填充扩张卷积覆盖的历史步数。"""

    if kernel_size <= 0 or not dilations or any(dilation <= 0 for dilation in dilations):
        raise ValueError("kernel_size and dilations must be positive")
    return 1 + (kernel_size - 1) * sum(dilations)


def graph_diffusion(
    inputs: torch.Tensor,
    support: torch.Tensor,
    order: int,
) -> tuple[torch.Tensor, ...]:
    """返回从零阶到指定阶的逐节点图扩散结果。"""

    if inputs.ndim != 4:
        raise ValueError(f"inputs must have shape [B,C,N,T], got {tuple(inputs.shape)}")
    if support.ndim != 2 or support.shape != (inputs.shape[2], inputs.shape[2]):
        raise ValueError("support shape must match the inputs node axis")
    if order < 0:
        raise ValueError("order must be non-negative")
    terms = [inputs]
    for _ in range(order):
        terms.append(torch.einsum("nm,bcmt->bcnt", support, terms[-1]))
    return tuple(terms)


class DiffusionGraphConv(nn.Module):
    """拼接多 support 的各阶扩散结果并执行逐点通道投影。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_supports: int,
        order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, num_supports) <= 0 or order < 0:
            raise ValueError(
                "channels and num_supports must be positive; order must be non-negative"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_supports = num_supports
        self.order = order
        diffusion_channels = in_channels * (1 + num_supports * order)
        self.projection = nn.Conv2d(diffusion_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(supports) != self.num_supports:
            raise ValueError(
                f"expected {self.num_supports} supports, got {len(supports)}"
            )
        terms = [inputs]
        for support in supports:
            terms.extend(graph_diffusion(inputs, support, self.order)[1:])
        return self.dropout(self.projection(torch.cat(terms, dim=1)))


class GatedTemporalConv(nn.Module):
    """沿时间轴执行无右侧填充的扩张门控卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation) <= 0:
            raise ValueError("channels, kernel_size, and dilation must be positive")
        kernel = (1, kernel_size)
        dilation_pair = (1, dilation)
        self.filter_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            dilation=dilation_pair,
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            dilation=dilation_pair,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.filter_conv(inputs)) * torch.sigmoid(self.gate_conv(inputs))


class SpatioTemporalBlock(nn.Module):
    """一个带 residual 与 skip 输出的 Graph WaveNet 时空块。"""

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        num_supports: int,
        diffusion_order: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = GatedTemporalConv(
            residual_channels,
            dilation_channels,
            kernel_size,
            dilation,
        )
        self.graph = DiffusionGraphConv(
            dilation_channels,
            residual_channels,
            num_supports,
            diffusion_order,
            dropout,
        )
        self.skip_projection = nn.Conv2d(dilation_channels, skip_channels, kernel_size=1)
        self.normalization = nn.BatchNorm2d(residual_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.temporal(inputs)
        skip = self.skip_projection(hidden)
        transformed = self.graph(hidden, supports)
        residual = transformed + inputs[..., -transformed.shape[-1] :]
        return self.normalization(residual), skip
