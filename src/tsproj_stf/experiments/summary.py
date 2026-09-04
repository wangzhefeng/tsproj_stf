"""多 seed 实验汇总。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REQUIRED_RUN_FILES = (
    "resolved_config.yaml",
    "environment.json",
    "data_manifest.json",
    "metrics.json",
    "predictions.npz",
    "run.log",
)


class IncompleteRunError(RuntimeError):
    """指定 run 缺少标准产物。"""


def _aggregate_metric_maps(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = set(items[0])
    if any(set(item) != keys for item in items[1:]):
        raise ValueError("metric keys differ between seed runs")
    return {
        key: {
            "mean": float(np.mean([item[key] for item in items])),
            "std": float(np.std([item[key] for item in items], ddof=1)),
        }
        for key in sorted(keys)
    }


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("seed", None)
    return normalized


def _validate_finite_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    sections = [metrics.get("overall")]
    horizons = metrics.get("horizons")
    if isinstance(horizons, dict):
        sections.extend(horizons.values())
    if not sections or any(
        not isinstance(section, dict)
        or any(not math.isfinite(float(value)) for value in section.values())
        for section in sections
    ):
        raise ValueError(f"non-finite metric in {run_dir}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def summarize_runs(
    root: str | Path,
    prefix: str,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """汇总指定模型前缀下全部明确 seed；不静默跳过 incomplete run。"""

    if len(seeds) < 2:
        raise ValueError("at least two seeds are required for sample standard deviation")
    root_path = Path(root)
    run_dirs = [root_path / f"{prefix}-seed{seed}" for seed in seeds]
    metrics_per_run: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        missing = [
            filename
            for filename in REQUIRED_RUN_FILES
            if not (run_dir / filename).is_file()
        ]
        if missing:
            raise IncompleteRunError(f"incomplete run {run_dir}: missing {', '.join(missing)}")
        config = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(config, dict):
            raise ValueError(f"resolved config must be a mapping: {run_dir}")
        if config.get("model") != "persistence" and not list(
            (run_dir / "checkpoint").rglob("*.pt")
        ):
            raise IncompleteRunError(f"incomplete run {run_dir}: missing checkpoint")
        run_status = json.loads((run_dir / "run.log").read_text(encoding="utf-8"))
        if run_status.get("status") != "completed":
            raise IncompleteRunError(f"incomplete run {run_dir}: run status is not completed")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        _validate_finite_metrics(metrics, run_dir)
        configs.append(config)
        manifests.append(
            json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
        )
        metrics_per_run.append(metrics)

    reference_config = _normalized_config(configs[0])
    if any(_normalized_config(config) != reference_config for config in configs[1:]):
        raise ValueError("resolved configs differ between seed runs")
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("data manifests differ between seed runs")

    horizon_keys = set(metrics_per_run[0]["horizons"])
    if any(set(item["horizons"]) != horizon_keys for item in metrics_per_run[1:]):
        raise ValueError("horizon keys differ between seed runs")
    summary: dict[str, Any] = {
        "prefix": prefix,
        "seeds": list(seeds),
        "runs": [str(path) for path in run_dirs],
        "overall": _aggregate_metric_maps([item["overall"] for item in metrics_per_run]),
        "horizons": {
            horizon: _aggregate_metric_maps(
                [item["horizons"][horizon] for item in metrics_per_run]
            )
            for horizon in sorted(horizon_keys)
        },
    }
    root_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(root_path / f"{prefix}-summary.json", summary)
    return summary
