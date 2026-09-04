import numpy as np
import pytest
import torch
from basicts.metrics import masked_mae as basicts_mae
from basicts.metrics import masked_mape as basicts_mape
from basicts.metrics import masked_rmse as basicts_rmse
from basicts.metrics import masked_wape as basicts_wape

from tsproj_stf.evaluation.metrics import evaluate_metrics


def test_project_metrics_match_basicts_after_inverse_scaling() -> None:
    mean = np.array([[[10.0, 20.0]]], dtype=np.float64)
    std = np.array([[[2.0, 5.0]]], dtype=np.float64)
    normalized_targets = np.array(
        [
            [[-5.0, 0.0], [1.0, -1.0]],
            [[2.0, 1.0], [0.5, 2.0]],
        ],
        dtype=np.float64,
    )
    normalized_prediction = normalized_targets + np.array(
        [
            [[1.0, -0.5], [-0.5, 0.2]],
            [[0.25, 100.0], [1.0, -0.4]],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [[True, True], [True, True]],
            [[True, False], [True, True]],
        ],
        dtype=bool,
    )
    targets = normalized_targets * std + mean
    prediction = normalized_prediction * std + mean

    project = evaluate_metrics(prediction, targets, observed)
    prediction_tensor = torch.from_numpy(prediction)
    targets_tensor = torch.from_numpy(targets)
    observed_tensor = torch.from_numpy(observed)
    backend = {
        "MAE": basicts_mae(prediction_tensor, targets_tensor, observed_tensor).item(),
        "RMSE": basicts_rmse(prediction_tensor, targets_tensor, observed_tensor).item(),
        "MAPE": basicts_mape(prediction_tensor, targets_tensor, observed_tensor).item(),
    }

    assert project["MAE"] == pytest.approx(backend["MAE"])
    assert project["RMSE"] == pytest.approx(backend["RMSE"])
    assert project["MAPE"] == pytest.approx(backend["MAPE"])


def test_wape_aggregation_semantics_are_intentionally_different() -> None:
    prediction = np.array([[[2.0], [10.0]], [[2.0], [2.0]]])
    targets = np.array([[[1.0], [1.0]], [[1.0], [100.0]]])
    observed = np.ones_like(targets, dtype=bool)

    project_wape = evaluate_metrics(prediction, targets, observed)["WAPE"]
    backend_wape = basicts_wape(
        torch.from_numpy(prediction),
        torch.from_numpy(targets),
        torch.from_numpy(observed),
    ).item()

    assert project_wape != pytest.approx(backend_wape)
