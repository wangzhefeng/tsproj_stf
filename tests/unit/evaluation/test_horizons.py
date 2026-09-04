import numpy as np
import pytest

from tsproj_stf.evaluation.horizons import evaluate_horizons, horizon_to_index


def test_converts_one_based_horizons() -> None:
    assert horizon_to_index(3, output_len=12) == 2
    assert horizon_to_index(6, output_len=12) == 5
    assert horizon_to_index(12, output_len=12) == 11


@pytest.mark.parametrize("horizon", [0, 13])
def test_rejects_out_of_range_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="outside 1..12"):
        horizon_to_index(horizon, output_len=12)


def test_evaluates_requested_horizons_on_axis_one() -> None:
    target = np.arange(24, dtype=np.float32).reshape(1, 12, 2)
    prediction = target.copy()
    prediction[:, 2, :] += 2.0
    observed = np.ones_like(target, dtype=bool)

    metrics = evaluate_horizons(prediction, target, observed, horizons=(3, 6, 12))

    assert metrics["h3"]["MAE"] == pytest.approx(2.0)
    assert metrics["h6"]["MAE"] == pytest.approx(0.0)
    assert metrics["h12"]["MAE"] == pytest.approx(0.0)
