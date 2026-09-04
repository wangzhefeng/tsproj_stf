import pytest
import torch

from tsproj_stf.evaluation.quantiles import pinball_loss, validate_quantiles


def test_validate_quantiles_requires_strict_order_inside_unit_interval() -> None:
    assert validate_quantiles((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_quantiles((0.5, 0.1, 0.9))
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        validate_quantiles((0.0, 0.5, 0.9))


def test_pinball_loss_matches_hand_calculation_and_observed_mask() -> None:
    prediction = torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    targets = torch.tensor([1.0, 3.0])
    observed = torch.tensor([True, False])

    loss = pinball_loss(
        prediction,
        targets,
        observed,
        quantiles=(0.1, 0.5, 0.9),
    )

    assert loss.item() == pytest.approx((0.1 + 0.0 + 0.1) / 3.0)
