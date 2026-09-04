"""项目自有 Graph WaveNet 模型。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsproj_stf.models.graph_wavenet.config import GraphWaveNetConfig
from tsproj_stf.models.graph_wavenet.graph import AdaptiveAdjacency, diffusion_supports
from tsproj_stf.models.graph_wavenet.layers import (
    SpatioTemporalBlock,
    receptive_field,
)


class GraphWaveNet(nn.Module):
    """支持显式有向图的 one-shot Graph WaveNet。"""

    def __init__(self, config: GraphWaveNetConfig) -> None:
        super().__init__()
        self.config = config
        self.receptive_field = receptive_field(config.kernel_size, config.dilations)
        self._fixed_support_names: tuple[str, ...] = ()
        if config.graph_mode in {"fixed", "hybrid"}:
            graph_path = Path(str(config.fixed_graph_path))
            with np.load(graph_path) as archive:
                if config.graph_name not in archive:
                    raise ValueError(
                        f"graph {config.graph_name!r} not found in {graph_path}; "
                        f"available={list(archive.files)}"
                    )
                adjacency = torch.from_numpy(
                    np.asarray(archive[config.graph_name], dtype=np.float32)
                )
            if adjacency.shape != (config.num_nodes, config.num_nodes):
                raise ValueError(
                    f"fixed graph shape {tuple(adjacency.shape)} does not match "
                    f"num_nodes={config.num_nodes}"
                )
            forward, reverse = diffusion_supports(adjacency)
            self.register_buffer("fixed_forward", forward)
            self.register_buffer("fixed_reverse", reverse)
            self._fixed_support_names = ("fixed_forward", "fixed_reverse")
        self.adaptive_adjacency = (
            AdaptiveAdjacency(config.num_nodes, config.adaptive_embedding_dim)
            if config.graph_mode in {"adaptive", "hybrid"}
            else None
        )
        num_supports = len(self._fixed_support_names) + int(
            self.adaptive_adjacency is not None
        )

        self.input_projection = nn.Conv2d(1, config.residual_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    residual_channels=config.residual_channels,
                    dilation_channels=config.dilation_channels,
                    skip_channels=config.skip_channels,
                    num_supports=num_supports,
                    diffusion_order=config.diffusion_order,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
                for dilation in config.dilations
            ]
        )
        self.end_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(config.skip_channels, config.end_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(config.end_channels, config.output_len, kernel_size=1),
        )

    def graph_supports(self) -> tuple[torch.Tensor, ...]:
        supports = [getattr(self, name) for name in self._fixed_support_names]
        if self.adaptive_adjacency is not None:
            supports.append(self.adaptive_adjacency())
        return tuple(supports)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [B,T,N], got {tuple(inputs.shape)}")
        if inputs.shape[1:] != (self.config.input_len, self.config.num_nodes):
            raise ValueError(
                f"expected input tail {(self.config.input_len, self.config.num_nodes)}, "
                f"got {tuple(inputs.shape[1:])}"
            )
        hidden = inputs.transpose(1, 2).unsqueeze(1)
        if hidden.shape[-1] < self.receptive_field:
            hidden = F.pad(hidden, (self.receptive_field - hidden.shape[-1], 0, 0, 0))
        hidden = self.input_projection(hidden)
        skip_total: torch.Tensor | None = None
        supports = self.graph_supports()
        for block in self.blocks:
            hidden, skip = block(hidden, supports)
            if skip_total is None:
                skip_total = skip
            else:
                skip_total = skip_total[..., -skip.shape[-1] :] + skip
        if skip_total is None:
            raise RuntimeError("Graph WaveNet requires at least one spatio-temporal block")
        prediction = self.end_projection(skip_total)[..., -1]
        return prediction
