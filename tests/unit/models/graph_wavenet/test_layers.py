import pytest
import torch

from tsproj_stf.models.graph_wavenet.layers import (
    DiffusionGraphConv,
    GatedTemporalConv,
    SpatioTemporalBlock,
    graph_diffusion,
    receptive_field,
)


def test_gated_temporal_conv_crops_only_the_time_axis() -> None:
    layer = GatedTemporalConv(
        in_channels=2,
        out_channels=4,
        kernel_size=2,
        dilation=3,
    )
    inputs = torch.randn(3, 2, 5, 10)

    outputs = layer(inputs)

    assert outputs.shape == (3, 4, 5, 7)
    assert torch.isfinite(outputs).all()


def test_receptive_field_matches_dilation_schedule() -> None:
    assert receptive_field(kernel_size=2, dilations=(1, 2, 4)) == 8


def test_graph_diffusion_returns_orders_zero_one_and_two() -> None:
    inputs = torch.tensor([[[[2.0], [5.0]]]])
    support = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    order_zero, order_one, order_two = graph_diffusion(inputs, support, order=2)

    torch.testing.assert_close(order_zero, inputs)
    torch.testing.assert_close(order_one, torch.tensor([[[[5.0], [5.0]]]]))
    torch.testing.assert_close(order_two, torch.tensor([[[[5.0], [5.0]]]]))


def test_diffusion_graph_conv_projects_all_support_orders() -> None:
    layer = DiffusionGraphConv(
        in_channels=2,
        out_channels=3,
        num_supports=2,
        order=2,
        dropout=0.0,
    )
    inputs = torch.randn(4, 2, 5, 7)
    supports = (torch.eye(5), torch.flip(torch.eye(5), dims=(1,)))

    outputs = layer(inputs, supports)

    assert outputs.shape == (4, 3, 5, 7)
    assert torch.isfinite(outputs).all()


def test_diffusion_graph_conv_rejects_missing_supports() -> None:
    layer = DiffusionGraphConv(
        in_channels=2,
        out_channels=3,
        num_supports=2,
        order=2,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="expected 2 supports"):
        layer(torch.randn(1, 2, 5, 7), ())


def test_spatiotemporal_block_aligns_residual_and_skip_outputs() -> None:
    block = SpatioTemporalBlock(
        residual_channels=4,
        dilation_channels=4,
        skip_channels=8,
        num_supports=2,
        diffusion_order=2,
        kernel_size=2,
        dilation=2,
        dropout=0.0,
    )
    inputs = torch.randn(2, 4, 3, 8)

    residual, skip = block(inputs, (torch.eye(3), torch.eye(3)))

    assert residual.shape == (2, 4, 3, 6)
    assert skip.shape == (2, 8, 3, 6)


def test_spatiotemporal_block_has_finite_gradients() -> None:
    block = SpatioTemporalBlock(
        residual_channels=4,
        dilation_channels=4,
        skip_channels=8,
        num_supports=2,
        diffusion_order=2,
        kernel_size=2,
        dilation=1,
        dropout=0.0,
    )
    inputs = torch.randn(2, 4, 3, 6, requires_grad=True)

    residual, skip = block(inputs, (torch.eye(3), torch.eye(3)))
    (residual.square().mean() + skip.square().mean()).backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )
