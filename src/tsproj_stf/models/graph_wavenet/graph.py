"""Graph WaveNet 图支持构造。"""

from __future__ import annotations

import torch
from torch import nn


def row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    """按出边归一化非负方阵；孤立节点保留全零行。"""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency contains non-finite values")
    if torch.any(adjacency < 0):
        raise ValueError("adjacency must be non-negative")
    row_sum = adjacency.sum(dim=1, keepdim=True)
    denominator = row_sum.clamp_min(torch.finfo(adjacency.dtype).eps)
    return torch.where(row_sum > 0, adjacency / denominator, 0)


def diffusion_supports(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回有向图的正向与反向随机游走矩阵。"""

    return row_normalize(adjacency), row_normalize(adjacency.transpose(0, 1))


class AdaptiveAdjacency(nn.Module):
    """以源/目标节点嵌入学习非对称 row-stochastic 邻接。"""

    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        if num_nodes <= 0 or embedding_dim <= 0:
            raise ValueError("num_nodes and embedding_dim must be positive")
        self.source_embeddings = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target_embeddings = nn.Parameter(torch.empty(embedding_dim, num_nodes))
        nn.init.xavier_uniform_(self.source_embeddings)
        nn.init.xavier_uniform_(self.target_embeddings)

    def forward(self) -> torch.Tensor:
        logits = torch.relu(self.source_embeddings @ self.target_embeddings)
        return torch.softmax(logits, dim=1)
