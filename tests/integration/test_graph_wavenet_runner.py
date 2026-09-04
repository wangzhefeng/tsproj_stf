import json
from pathlib import Path

import numpy as np
import pytest

from tsproj_stf.experiments.artifacts import fingerprint_file
from tsproj_stf.experiments.config import ExperimentConfig
from tsproj_stf.experiments.runner import run_experiment


def write_processed_fixture(tmp_path: Path, length: int = 120, num_nodes: int = 3) -> Path:
    output = tmp_path / "processed"
    output.mkdir()
    time = np.arange(length, dtype=np.float32)
    values = np.stack(
        [20.0 + node + np.sin(time / (3.0 + node)) for node in range(num_nodes)],
        axis=1,
    )[:, :, None].astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    timestamps = np.arange(
        np.datetime64("2026-01-05T00:00"),
        np.datetime64("2026-01-05T00:00") + length * np.timedelta64(5, "m"),
        np.timedelta64(5, "m"),
    )
    np.save(output / "values.npy", values)
    np.save(output / "observed.npy", observed)
    np.save(output / "timestamps.npy", timestamps)
    adjacency = np.eye(num_nodes, k=1, dtype=np.float32)
    adjacency[-1, 0] = 1.0
    np.savez_compressed(output / "graphs.npz", physical=adjacency)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "node_ids": [f"n{index}" for index in range(num_nodes)],
                "feature_names": ["speed"],
                "target_features": ["speed"],
            }
        ),
        encoding="utf-8",
    )
    processed_files = {
        filename: fingerprint_file(output / filename)
        for filename in (
            "values.npy",
            "observed.npy",
            "timestamps.npy",
            "graphs.npz",
            "metadata.json",
        )
    }
    (output / "data_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "shape": list(values.shape),
                "processed_files": processed_files,
            }
        ),
        encoding="utf-8",
    )
    return output


def make_config(tmp_path: Path, data_path: Path, mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"fixture_gwn_{mode}",
        dataset="fixture",
        data_path=str(data_path),
        model="graph_wavenet",
        input_len=6,
        output_len=3,
        split_ratios=(0.6, 0.2, 0.2),
        horizons=(1, 3),
        seed=42,
        results_dir=str(tmp_path / "results"),
        rescale=True,
        model_params={
            "target_feature": "speed",
            "null_value": 0.0,
            "graph_mode": mode,
            "graph_name": "physical",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "patience": 2,
            "residual_channels": 4,
            "dilation_channels": 4,
            "skip_channels": 8,
            "end_channels": 16,
            "kernel_size": 2,
            "dilations": [1, 2],
            "diffusion_order": 2,
            "adaptive_embedding_dim": 4,
            "dropout": 0.0,
        },
    )


@pytest.mark.parametrize("mode", ["fixed", "adaptive", "hybrid"])
def test_graph_wavenet_modes_run_one_cpu_epoch(tmp_path: Path, mode: str) -> None:
    data_path = write_processed_fixture(tmp_path)

    result = run_experiment(make_config(tmp_path, data_path, mode))

    assert result.prediction_shape == (16, 3, 3)
    assert np.isfinite(result.metrics["overall"]["MAE"])
    assert (result.run_dir / "run.log").is_file()
    assert list((result.run_dir / "checkpoint").rglob("*.pt"))
