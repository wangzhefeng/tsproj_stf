import json
from pathlib import Path

import numpy as np
import pytest

from tsproj_stf.experiments.artifacts import RunConflictError, fingerprint_file
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
    np.savez_compressed(output / "graphs.npz", physical=np.eye(num_nodes, dtype=np.float32))
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


def make_config(
    tmp_path: Path,
    data_path: Path,
    *,
    model: str,
    model_params: dict[str, object] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"fixture_{model}",
        dataset="fixture",
        data_path=str(data_path),
        model=model,
        input_len=6,
        output_len=3,
        split_ratios=(0.6, 0.2, 0.2),
        horizons=(1, 3),
        seed=42,
        results_dir=str(tmp_path / "results"),
        rescale=True,
        model_params=model_params or {},
    )


def assert_standard_artifacts(run_dir: Path) -> None:
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "data_manifest.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "run.log").is_file()


def test_runs_persistence_and_writes_standard_artifacts(tmp_path: Path) -> None:
    data_path = write_processed_fixture(tmp_path)

    result = run_experiment(make_config(tmp_path, data_path, model="persistence"))

    assert_standard_artifacts(result.run_dir)
    assert result.prediction_shape == (16, 3, 3)
    assert set(result.metrics) == {"overall", "horizons", "metadata"}
    assert np.isfinite(result.metrics["overall"]["MAE"])
    run_manifest = json.loads((result.run_dir / "data_manifest.json").read_text())
    assert run_manifest["split_bounds"] == {
        "train": [0, 72],
        "val": [72, 96],
        "test": [96, 120],
    }
    assert len(run_manifest["source_manifest_sha256"]) == 64
    with np.load(result.run_dir / "predictions.npz") as archive:
        assert archive["prediction"].shape == (16, 3, 3)
        assert archive["observed"].all()


def test_completed_run_requires_force_new_run(tmp_path: Path) -> None:
    data_path = write_processed_fixture(tmp_path)
    config = make_config(tmp_path, data_path, model="persistence")
    original = run_experiment(config)

    with pytest.raises(RunConflictError, match="completed run"):
        run_experiment(config, resume=True)
    forced = run_experiment(config, force_new_run=True)

    assert forced.run_dir.name == f"{original.run_dir.name}-run2"


def test_rejects_stale_processed_file_fingerprint(tmp_path: Path) -> None:
    data_path = write_processed_fixture(tmp_path)
    values = np.load(data_path / "values.npy")
    values[0, 0, 0] += 1.0
    np.save(data_path / "values.npy", values)

    with pytest.raises(ValueError, match="processed file fingerprint mismatch"):
        run_experiment(make_config(tmp_path, data_path, model="persistence"))
    assert not (tmp_path / "results" / "fixture_persistence-seed42").exists()


def test_persistence_cold_start_uses_train_only_node_mean(tmp_path: Path) -> None:
    data_path = write_processed_fixture(tmp_path)
    observed = np.load(data_path / "observed.npy")
    observed[96:102, 0, 0] = False
    np.save(data_path / "observed.npy", observed)
    manifest_path = data_path / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["processed_files"]["observed.npy"] = fingerprint_file(
        data_path / "observed.npy"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = make_config(
        tmp_path,
        data_path,
        model="persistence",
        model_params={
            "target_feature": "speed",
            "fallback_strategy": "train_mean",
        },
    )

    result = run_experiment(config)

    train_mean = np.load(data_path / "values.npy")[:72, 0, 0].mean()
    with np.load(result.run_dir / "predictions.npz") as archive:
        np.testing.assert_allclose(archive["prediction"][0, :, 0], train_mean)


def test_runs_stid_for_one_cpu_epoch_and_saves_checkpoint(tmp_path: Path) -> None:
    data_path = write_processed_fixture(tmp_path)
    config = make_config(
        tmp_path,
        data_path,
        model="stid",
        model_params={
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "patience": 2,
            "input_hidden_size": 8,
            "spatial_hidden_size": 8,
            "tid_hidden_size": 4,
            "diw_hidden_size": 4,
            "num_layers": 1,
        },
    )

    result = run_experiment(config)

    assert_standard_artifacts(result.run_dir)
    assert result.prediction_shape == (16, 3, 3)
    assert list((result.run_dir / "checkpoint").rglob("*.pt"))
    assert np.isfinite(result.metrics["overall"]["MAE"])

    (result.run_dir / "run.log").unlink()
    # 中断的训练 run 残留 checkpoint：resume 语义歧义，必须响亮失败
    with pytest.raises(RunConflictError, match="stale training checkpoints"):
        run_experiment(config, resume=True)

    fresh = run_experiment(config, force_new_run=True)
    assert fresh.run_dir != result.run_dir
    assert np.isfinite(fresh.metrics["overall"]["MAE"])
