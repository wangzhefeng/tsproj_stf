import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tsproj_stf.data.adapters.basicts import (
    ProjectBasicTSScaler,
    ProjectForecastingDataset,
    ProjectForecastingTaskFlow,
)


def write_processed(tmp_path: Path) -> Path:
    output = tmp_path / "processed"
    output.mkdir()
    values = np.arange(1, 61, dtype=np.float32).reshape(30, 2, 1)
    observed = np.ones_like(values, dtype=bool)
    timestamps = np.arange(
        np.datetime64("2026-01-05T00:00"),
        np.datetime64("2026-01-05T02:30"),
        np.timedelta64(5, "m"),
    )
    np.save(output / "values.npy", values)
    np.save(output / "observed.npy", observed)
    np.save(output / "timestamps.npy", timestamps)
    np.savez_compressed(output / "graphs.npz", physical=np.eye(2, dtype=np.float32))
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "node_ids": ["s1", "s2"],
                "feature_names": ["speed"],
                "target_features": ["speed"],
            }
        ),
        encoding="utf-8",
    )
    return output


def test_builds_train_windows_without_crossing_split(tmp_path: Path) -> None:
    dataset = ProjectForecastingDataset(
        data_file_path=write_processed(tmp_path),
        dataset_name="fixture",
        input_len=3,
        output_len=2,
        mode="train",
        split_ratios=(0.5, 0.25, 0.25),
        target_feature="speed",
    )

    assert len(dataset) == 11
    assert dataset.data.shape == (15, 2)
    sample = dataset[0]
    assert sample["inputs"].shape == (3, 2)
    assert sample["inputs_observed"].shape == (3, 2)
    assert sample["inputs_observed"].all()
    assert sample["targets"].shape == (2, 2)
    assert sample["inputs_timestamps"].shape == (3, 2)
    assert sample["targets_observed"].shape == (2, 2)
    assert sample["targets_observed"].all()


def test_validation_starts_at_chronological_offset(tmp_path: Path) -> None:
    dataset = ProjectForecastingDataset(
        data_file_path=write_processed(tmp_path),
        dataset_name="fixture",
        input_len=3,
        output_len=2,
        mode="val",
        split_ratios=(0.5, 0.25, 0.25),
        target_feature="speed",
    )

    np.testing.assert_array_equal(dataset[0]["inputs"][0], [31.0, 32.0])
    assert len(dataset) == 3


def test_replaces_unobserved_values_with_configured_null(tmp_path: Path) -> None:
    processed = write_processed(tmp_path)
    observed = np.load(processed / "observed.npy")
    observed[0, 0, 0] = False
    np.save(processed / "observed.npy", observed)

    dataset = ProjectForecastingDataset(
        data_file_path=processed,
        dataset_name="fixture",
        input_len=3,
        output_len=2,
        mode="train",
        split_ratios=(0.5, 0.25, 0.25),
        target_feature="speed",
        null_value=-999.0,
    )

    assert dataset[0]["inputs"][0, 0] == -999.0


def test_requires_explicit_existing_target_feature(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown target feature"):
        ProjectForecastingDataset(
            data_file_path=write_processed(tmp_path),
            dataset_name="fixture",
            input_len=3,
            output_len=2,
            mode="train",
            split_ratios=(0.5, 0.25, 0.25),
            target_feature="load",
        )


def test_project_scaler_excludes_unobserved_training_values(tmp_path: Path) -> None:
    processed = write_processed(tmp_path)
    values = np.load(processed / "values.npy")
    observed = np.load(processed / "observed.npy")
    values[1, 0, 0] = 10_000.0
    observed[1, 0, 0] = False
    np.save(processed / "values.npy", values)
    np.save(processed / "observed.npy", observed)

    scaler = ProjectBasicTSScaler(
        data_file_path=processed,
        split_ratios=(0.5, 0.25, 0.25),
        target_feature="speed",
        rescale=True,
    )
    original_mean = scaler.stats["mean"].clone()
    scaler.fit(np.full((15, 2), -999.0, dtype=np.float32))
    normalized = scaler.transform(torch.tensor([[3.0, 4.0]]))
    restored = scaler.inverse_transform(normalized)

    assert scaler.stats["mean"][0, 0] < 100.0
    assert torch.equal(scaler.stats["mean"], original_mean)
    torch.testing.assert_close(restored, torch.tensor([[3.0, 4.0]]))


def test_taskflow_uses_observed_mask_and_preserves_valid_zero() -> None:
    flow = ProjectForecastingTaskFlow()
    runner = SimpleNamespace(
        cfg=SimpleNamespace(null_to_num=-999.0),
        scaler=None,
    )
    data = {
        "inputs": torch.tensor([[0.0, 8.0]]),
        "inputs_observed": torch.tensor([[True, False]]),
        "targets": torch.tensor([[0.0, 9.0]]),
        "targets_observed": torch.tensor([[True, False]]),
    }

    processed = flow.preprocess(runner, data)

    torch.testing.assert_close(processed["inputs"], torch.tensor([[0.0, -999.0]]))
    torch.testing.assert_close(processed["targets"], torch.tensor([[0.0, -999.0]]))
    assert torch.equal(processed["targets_mask"], torch.tensor([[True, False]]))
