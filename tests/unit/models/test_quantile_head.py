import torch

from tsproj_stf.models.quantile_head import QuantileHead


def test_quantile_head_outputs_ordered_quantiles() -> None:
    head = QuantileHead(in_features=5, quantiles=(0.1, 0.5, 0.9))
    features = torch.randn(2, 3, 4, 5)

    prediction = head(features)

    assert prediction.shape == (2, 3, 4, 3)
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction[..., 0] <= prediction[..., 1])
    assert torch.all(prediction[..., 1] <= prediction[..., 2])
