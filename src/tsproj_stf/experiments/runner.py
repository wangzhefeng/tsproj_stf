"""统一实验执行入口。"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from tsproj_stf.baselines import persistence_forecast
from tsproj_stf.data.adapters import ProjectForecastingDataset
from tsproj_stf.data.split import chronological_split
from tsproj_stf.evaluation import evaluate_horizons, evaluate_metrics
from tsproj_stf.experiments.artifacts import ArtifactStore, fingerprint_file
from tsproj_stf.experiments.config import ExperimentConfig


@dataclass(frozen=True)
class RunResult:
    """一次完成实验的关键返回信息。"""

    run_dir: Path
    metrics: dict[str, Any]
    prediction_shape: tuple[int, ...]


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else None,
    }


def _environment_metadata() -> dict[str, Any]:
    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "BasicTS": version("BasicTS"),
            "numpy": version("numpy"),
            "torch": version("torch"),
            "tsproj-stf": version("tsproj-stf"),
        },
        "git": _git_metadata(),
        "device": "cpu",
        "lockfile_sha256": fingerprint_file(lock_path) if lock_path.is_file() else None,
    }


def _load_manifest(
    data_path: Path,
    split_ratios: tuple[float, float, float],
) -> dict[str, Any]:
    manifest_path = data_path / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"processed dataset manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must be a JSON object: {manifest_path}")
    processed_files = payload.get("processed_files")
    if not isinstance(processed_files, dict) or not processed_files:
        raise ValueError("dataset manifest must contain processed_files fingerprints")
    required_files = {
        "values.npy",
        "observed.npy",
        "timestamps.npy",
        "graphs.npz",
        "metadata.json",
    }
    missing_fingerprints = required_files - set(processed_files)
    if missing_fingerprints:
        raise ValueError(
            "dataset manifest missing processed fingerprints: "
            f"{sorted(missing_fingerprints)}"
        )
    for filename, expected in processed_files.items():
        if Path(filename).name != filename:
            raise ValueError(f"invalid processed filename in manifest: {filename!r}")
        path = data_path / filename
        actual = fingerprint_file(path)
        if actual != expected:
            raise ValueError(
                f"processed file fingerprint mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
    values = np.load(data_path / "values.npy", mmap_mode="r")
    run_manifest = dict(payload)
    run_manifest["source_manifest_sha256"] = fingerprint_file(manifest_path)
    run_manifest["split_bounds"] = chronological_split(
        len(values), split_ratios
    ).as_dict()
    return run_manifest


def _build_dataset(config: ExperimentConfig, mode: str) -> ProjectForecastingDataset:
    return ProjectForecastingDataset(
        data_file_path=config.data_path,
        dataset_name=config.dataset,
        input_len=config.input_len,
        output_len=config.output_len,
        mode=mode,
        split_ratios=config.split_ratios,
        target_feature=str(config.model_params.get("target_feature", "speed")),
        null_value=float(config.model_params.get("null_value", 0.0)),
        use_timestamps=True,
    )


def _collect_dataset_arrays(
    dataset: ProjectForecastingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = [dataset[index] for index in range(len(dataset))]
    inputs = np.stack([sample["inputs"] for sample in samples]).astype(np.float32)
    inputs_observed = np.stack([sample["inputs_observed"] for sample in samples]).astype(bool)
    targets = np.stack([sample["targets"] for sample in samples]).astype(np.float32)
    targets_observed = np.stack([sample["targets_observed"] for sample in samples]).astype(bool)
    return inputs, inputs_observed, targets, targets_observed


def _persistence_fallback(config: ExperimentConfig) -> np.ndarray | float | None:
    strategy = config.model_params.get("fallback_strategy")
    if strategy is None:
        fallback_value = config.model_params.get("fallback_value")
        return None if fallback_value is None else float(fallback_value)
    if strategy != "train_mean":
        raise ValueError(f"unsupported persistence fallback_strategy: {strategy}")
    data_path = Path(config.data_path)
    metadata = json.loads((data_path / "metadata.json").read_text(encoding="utf-8"))
    feature_names = tuple(metadata["feature_names"])
    target_feature = str(config.model_params.get("target_feature", "speed"))
    if target_feature not in feature_names:
        raise ValueError(f"unknown target feature: {target_feature!r}")
    target_index = feature_names.index(target_feature)
    values = np.load(data_path / "values.npy", mmap_mode="r")
    observed = np.load(data_path / "observed.npy", mmap_mode="r")
    train = chronological_split(len(values), config.split_ratios).train
    train_values = np.asarray(values[train, :, target_index], dtype=np.float64)
    train_observed = np.asarray(observed[train, :, target_index], dtype=bool)
    counts = train_observed.sum(axis=0)
    if np.any(counts == 0):
        missing_nodes = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"cannot compute train_mean fallback for nodes {missing_nodes}")
    return (
        np.where(train_observed, train_values, 0.0).sum(axis=0) / counts
    ).astype(np.float32)


def _standard_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "overall": evaluate_metrics(prediction, targets, observed),
        "horizons": evaluate_horizons(prediction, targets, observed, horizons),
        "metadata": {
            "sample_count": int(prediction.shape[0]),
            "valid_target_count": int(observed.sum()),
        },
    }


def _write_standard_artifacts(
    store: ArtifactStore,
    run_manifest: dict[str, Any],
    metrics: dict[str, Any],
    prediction: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
) -> None:
    store.write_json("environment.json", _environment_metadata())
    store.write_json("data_manifest.json", run_manifest)
    store.write_json("metrics.json", metrics)
    store.write_predictions(prediction, targets, observed)
    store.write_text(
        "run.log",
        json.dumps(
            {
                "status": "completed",
                "prediction_shape": list(prediction.shape),
                "valid_target_count": int(observed.sum()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    force_new_run: bool = False,
) -> RunResult:
    """执行配置指定的单次实验。"""

    if config.model not in {"persistence", "stid", "graph_wavenet"}:
        raise ValueError(f"unsupported model: {config.model}")

    run_manifest = _load_manifest(Path(config.data_path), config.split_ratios)
    resolved_config = config.to_dict()
    run_id = f"{config.name}-seed{config.seed}"
    store = ArtifactStore.initialize(
        config.results_dir,
        run_id,
        resolved_config,
        resume=resume,
        force_new_run=force_new_run,
    )

    if config.model == "persistence":
        dataset = _build_dataset(config, "test")
        inputs, inputs_observed, targets, targets_observed = _collect_dataset_arrays(dataset)
        prediction = persistence_forecast(
            inputs,
            inputs_observed,
            config.output_len,
            fallback_value=_persistence_fallback(config),
        )
    elif config.model == "stid":
        from tsproj_stf.experiments.stid import run_stid_backend

        prediction, targets, targets_observed = run_stid_backend(config, store)
    else:
        from tsproj_stf.experiments.graph_wavenet import run_graph_wavenet_backend

        prediction, targets, targets_observed = run_graph_wavenet_backend(config, store)
    metrics = _standard_metrics(prediction, targets, targets_observed, config.horizons)

    _write_standard_artifacts(
        store,
        run_manifest,
        metrics,
        prediction,
        targets,
        targets_observed,
    )
    return RunResult(store.run_dir, metrics, tuple(prediction.shape))
