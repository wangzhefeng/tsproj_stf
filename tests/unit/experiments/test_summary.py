import json
from pathlib import Path

import pytest
import yaml

from tsproj_stf.experiments.summary import IncompleteRunError, summarize_runs

REQUIRED_FILES = (
    "resolved_config.yaml",
    "environment.json",
    "data_manifest.json",
    "metrics.json",
    "predictions.npz",
    "run.log",
)


def write_run(
    root: Path,
    prefix: str,
    seed: int,
    mae: float,
    *,
    omit: str | None = None,
    model: str = "stid",
    config_width: int = 32,
    data_id: str = "same-data",
) -> None:
    run_dir = root / f"{prefix}-seed{seed}"
    run_dir.mkdir(parents=True)
    metrics = {
        "overall": {"MAE": mae, "RMSE": mae + 1.0},
        "horizons": {"h3": {"MAE": mae + 0.5}},
        "metadata": {"sample_count": 10, "valid_target_count": 20},
    }
    payloads: dict[str, bytes] = {
        "resolved_config.yaml": yaml.safe_dump(
            {
                "name": prefix,
                "model": model,
                "seed": seed,
                "model_params": {"width": config_width},
            }
        ).encode(),
        "environment.json": b"{}",
        "data_manifest.json": json.dumps(
            {
                "source_manifest_sha256": data_id,
                "split_bounds": {"train": [0, 6], "val": [6, 8], "test": [8, 10]},
            }
        ).encode(),
        "metrics.json": json.dumps(metrics).encode(),
        "predictions.npz": b"complete",
        "run.log": b'{"status":"completed"}',
    }
    for filename, content in payloads.items():
        if filename == omit:
            continue
        (run_dir / filename).write_bytes(content)
    if model != "persistence" and omit != "checkpoint":
        checkpoint = run_dir / "checkpoint" / "model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "best.pt").write_bytes(b"checkpoint")


def test_summarizes_mean_and_sample_standard_deviation(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0)
    write_run(tmp_path, "stid", 43, 2.0)
    write_run(tmp_path, "stid", 44, 3.0)

    summary = summarize_runs(tmp_path, "stid", seeds=(42, 43, 44))

    assert summary["seeds"] == [42, 43, 44]
    assert summary["overall"]["MAE"] == {"mean": 2.0, "std": 1.0}
    assert summary["horizons"]["h3"]["MAE"] == {"mean": 2.5, "std": 1.0}
    saved = json.loads((tmp_path / "stid-summary.json").read_text())
    assert saved == summary


def test_rejects_incomplete_seed_run(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0, omit="predictions.npz")
    write_run(tmp_path, "stid", 43, 2.0)

    with pytest.raises(IncompleteRunError, match="predictions.npz"):
        summarize_runs(tmp_path, "stid", seeds=(42, 43))


def test_requires_at_least_two_seeds_for_variance(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0)

    with pytest.raises(ValueError, match="at least two seeds"):
        summarize_runs(tmp_path, "stid", seeds=(42,))


def test_rejects_incompatible_resolved_configs(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0, config_width=32)
    write_run(tmp_path, "stid", 43, 2.0, config_width=64)

    with pytest.raises(ValueError, match="resolved configs differ"):
        summarize_runs(tmp_path, "stid", seeds=(42, 43))


def test_rejects_incompatible_data_manifests(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0, data_id="data-a")
    write_run(tmp_path, "stid", 43, 2.0, data_id="data-b")

    with pytest.raises(ValueError, match="data manifests differ"):
        summarize_runs(tmp_path, "stid", seeds=(42, 43))


def test_rejects_trained_run_without_checkpoint(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, 1.0, omit="checkpoint")
    write_run(tmp_path, "stid", 43, 2.0)

    with pytest.raises(IncompleteRunError, match="checkpoint"):
        summarize_runs(tmp_path, "stid", seeds=(42, 43))


def test_rejects_non_finite_metrics(tmp_path: Path) -> None:
    write_run(tmp_path, "stid", 42, float("inf"))
    write_run(tmp_path, "stid", 43, 2.0)

    with pytest.raises(ValueError, match="non-finite metric"):
        summarize_runs(tmp_path, "stid", seeds=(42, 43))
