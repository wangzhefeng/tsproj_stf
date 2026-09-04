import torch

from tsproj_stf.models.graph_wavenet.graph import (
    AdaptiveAdjacency,
    diffusion_supports,
    row_normalize,
)


def test_row_normalize_preserves_isolated_rows() -> None:
    adjacency = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    normalized = row_normalize(adjacency)

    torch.testing.assert_close(normalized[0], torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0]))
    torch.testing.assert_close(normalized[1], torch.zeros(3))
    torch.testing.assert_close(normalized[2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.isfinite(normalized).all()


def test_diffusion_supports_preserve_forward_and_reverse_direction() -> None:
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    forward, reverse = diffusion_supports(adjacency)

    assert forward.shape == reverse.shape == (3, 3)
    torch.testing.assert_close(forward[0], torch.tensor([0.0, 0.25, 0.75]))
    torch.testing.assert_close(reverse[0], torch.tensor([0.0, 1.0, 0.0]))
    assert not torch.equal(forward, reverse)


def test_adaptive_adjacency_is_row_stochastic_and_asymmetric() -> None:
    graph = AdaptiveAdjacency(num_nodes=3, embedding_dim=2)
    with torch.no_grad():
        graph.source_embeddings.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        graph.target_embeddings.copy_(
            torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        )

    adjacency = graph()

    assert adjacency.shape == (3, 3)
    assert torch.all(adjacency >= 0)
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(3))
    assert not torch.allclose(adjacency, adjacency.transpose(0, 1))
