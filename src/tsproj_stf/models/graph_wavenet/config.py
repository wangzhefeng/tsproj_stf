"""Graph WaveNet 模型配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from basicts.configs.model_config import BasicTSModelConfig


@dataclass
class GraphWaveNetConfig(BasicTSModelConfig):
    """可序列化的 Graph WaveNet 配置。"""

    input_len: int
    output_len: int
    num_nodes: int
    graph_mode: str
    fixed_graph_path: str | None = None
    graph_name: str = "physical"
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    kernel_size: int = 2
    dilations: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8))
    diffusion_order: int = 2
    adaptive_embedding_dim: int = 10
    dropout: float = 0.3

    def __post_init__(self) -> None:
        positive = (
            self.input_len,
            self.output_len,
            self.num_nodes,
            self.residual_channels,
            self.dilation_channels,
            self.skip_channels,
            self.end_channels,
            self.kernel_size,
            self.adaptive_embedding_dim,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Graph WaveNet dimensions must be positive")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive values")
        if self.diffusion_order < 0:
            raise ValueError("diffusion_order must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.graph_mode not in {"fixed", "adaptive", "hybrid"}:
            raise ValueError(f"unsupported graph_mode: {self.graph_mode}")
        if self.graph_mode in {"fixed", "hybrid"} and not self.fixed_graph_path:
            raise ValueError(f"{self.graph_mode} mode requires fixed_graph_path")
