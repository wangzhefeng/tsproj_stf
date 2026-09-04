from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tsproj_stf.models.graph_wavenet.config import GraphWaveNetConfig
from tsproj_stf.models.graph_wavenet.model import GraphWaveNet


def write_graph(tmp_path: Path, num_nodes: int = 4) -> Path:
    path = tmp_path / "graphs.npz"
    adjacency = np.eye(num_nodes, k=1, dtype=np.float32)
    adjacency[-1, 0] = 1.0
    np.savez_compressed(path, physical=adjacency)
    return path


def make_config(tmp_path: Path, mode: str = "fixed") -> GraphWaveNetConfig:
    return GraphWaveNetConfig(
        input_len=12,
        output_len=3,
        num_nodes=4,
        graph_mode=mode,
        fixed_graph_path=str(write_graph(tmp_path)),
        graph_name="physical",
        residual_channels=8,
        dilation_channels=8,
        skip_channels=16,
        end_channels=32,
        kernel_size=2,
        dilations=(1, 2, 4),
        diffusion_order=2,
        adaptive_embedding_dim=4,
        dropout=0.0,
    )


def test_fixed_mode_outputs_one_shot_forecast(tmp_path: Path) -> None:
    model = GraphWaveNet(make_config(tmp_path))
    inputs = torch.randn(2, 12, 4)
    timestamps = torch.zeros(2, 12, 2)

    prediction = model(inputs, timestamps)

    assert prediction.shape == (2, 3, 4)
    assert torch.isfinite(prediction).all()
    assert model.adaptive_adjacency is None
    assert not any("adaptive_adjacency" in name for name, _ in model.named_parameters())


def test_adaptive_mode_does_not_read_fixed_graph(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path, mode="adaptive"),
        fixed_graph_path=str(tmp_path / "missing.npz"),
    )

    model = GraphWaveNet(config)
    prediction = model(torch.randn(2, 12, 4), torch.zeros(2, 12, 2))

    assert model.adaptive_adjacency is not None
    assert len(model.graph_supports()) == 1
    assert prediction.shape == (2, 3, 4)
    assert torch.isfinite(prediction).all()


def test_hybrid_mode_combines_fixed_and_adaptive_supports(tmp_path: Path) -> None:
    model = GraphWaveNet(make_config(tmp_path, mode="hybrid"))

    supports = model.graph_supports()
    prediction = model(torch.randn(2, 12, 4), torch.zeros(2, 12, 2))

    assert model.adaptive_adjacency is not None
    assert len(supports) == 3
    assert prediction.shape == (2, 3, 4)
    assert torch.isfinite(prediction).all()
