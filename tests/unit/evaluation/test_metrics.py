import math

import numpy as np
import pytest

from tsproj_stf.evaluation.metrics import evaluate_metrics, masked_mae, masked_mape


def test_metrics_match_hand_calculation() -> None:
    prediction = np.array([2.0, 4.0])
    target = np.array([1.0, 2.0])
    observed = np.array([True, True])

    metrics = evaluate_metrics(prediction, target, observed)

    assert metrics == pytest.approx(
        {
            "MAE": 1.5,
            "RMSE": math.sqrt(2.5),
            "MAPE": 1.0,
            "WAPE": 1.0,
        }
    )


def test_all_metrics_use_observed_mask() -> None:
    prediction = np.array([2.0, 1000.0])
    target = np.array([1.0, 2.0])
    observed = np.array([True, False])

    metrics = evaluate_metrics(prediction, target, observed)

    assert metrics == {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "WAPE": 1.0}


def test_mape_excludes_zero_and_near_zero_targets() -> None:
    prediction = np.array([100.0, 4.0])
    target = np.array([0.0, 2.0])

    assert masked_mape(prediction, target, np.array([True, True])) == pytest.approx(1.0)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        masked_mae(np.ones(2), np.ones(3), np.ones(3, dtype=bool))


def test_rejects_empty_valid_mask() -> None:
    with pytest.raises(ValueError, match="no valid targets"):
        masked_mae(np.ones(2), np.ones(2), np.zeros(2, dtype=bool))


def test_rejects_non_finite_observed_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        masked_mae(np.array([np.nan]), np.array([1.0]), np.array([True]))
