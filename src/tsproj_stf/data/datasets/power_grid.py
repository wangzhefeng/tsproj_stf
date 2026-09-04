"""通用电力宽表数据适配。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsproj_stf.data.schema import SpatioTemporalData
from tsproj_stf.data.validation import validate_regular_timestamps
from tsproj_stf.experiments.artifacts import fingerprint_bytes, fingerprint_file


def load_power_grid_wide(
    csv_path: str | Path,
    *,
    timestamp_column: str,
    node_ids: Sequence[str],
    feature_columns: Mapping[str, Mapping[str, str]],
    target_features: Sequence[str],
    fill_value: float = 0.0,
) -> SpatioTemporalData:
    """按显式 feature/node/column 映射读取电力宽表。"""

    ordered_nodes = tuple(str(node_id) for node_id in node_ids)
    if not ordered_nodes or len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError("node_ids must be non-empty and unique")
    ordered_features = tuple(str(name) for name in feature_columns)
    if not ordered_features:
        raise ValueError("feature_columns must explicitly define at least one feature")
    targets = tuple(str(name) for name in target_features)
    if not targets or set(targets) - set(ordered_features):
        raise ValueError("target_features must be a non-empty subset of feature_columns")

    frame = pd.read_csv(csv_path)
    if timestamp_column not in frame:
        raise ValueError(f"timestamp column {timestamp_column!r} not found")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("timestamps must be unique")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    timestamps_array = timestamps.iloc[order].to_numpy(dtype="datetime64[ns]")
    validate_regular_timestamps(timestamps_array)

    values = np.empty(
        (len(frame), len(ordered_nodes), len(ordered_features)),
        dtype=np.float32,
    )
    for feature_index, feature_name in enumerate(ordered_features):
        mapping = feature_columns[feature_name]
        if set(mapping) != set(ordered_nodes):
            raise ValueError(
                f"feature {feature_name!r} must map every node exactly once"
            )
        columns = [mapping[node_id] for node_id in ordered_nodes]
        missing_columns = [column for column in columns if column not in frame]
        if missing_columns:
            raise ValueError(f"mapped CSV columns not found: {missing_columns}")
        values[:, :, feature_index] = frame.loc[:, columns].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float32)
    observed = np.isfinite(values)
    stored_values = np.where(observed, values, np.float32(fill_value))
    return SpatioTemporalData(
        values=stored_values,
        observed=observed,
        timestamps=timestamps_array,
        node_ids=ordered_nodes,
        feature_names=ordered_features,
        target_features=targets,
        graphs={},
    )


def save_power_grid_processed(
    data: SpatioTemporalData,
    output_dir: str | Path,
    csv_path: str | Path,
    *,
    graph_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """保存标准电力数据、多图和可追溯 manifest。"""

    provenance = dict(graph_provenance or {})
    if set(provenance) != set(data.graphs):
        raise ValueError("graph_provenance keys must exactly match graph names")
    for graph_name, details in provenance.items():
        if tuple(details.get("node_ids", ())) != data.node_ids:
            raise ValueError(f"graph provenance node order mismatch: {graph_name}")
        source_sha256 = details.get("source_sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError(f"graph provenance requires source_sha256: {graph_name}")
        if details.get("kind") not in {"physical", "distance", "correlation"}:
            raise ValueError(f"unsupported graph provenance kind: {graph_name}")
        required_by_kind = {
            "physical": {"directed"},
            "distance": {"sigma", "threshold"},
            "correlation": {"target_feature", "train_bounds", "top_k", "threshold"},
        }
        missing = required_by_kind[str(details["kind"])] - set(details)
        if missing:
            raise ValueError(
                f"graph provenance missing parameters for {graph_name}: {sorted(missing)}"
            )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "values.npy", data.values)
    np.save(destination / "observed.npy", data.observed)
    np.save(destination / "timestamps.npy", data.timestamps)
    np.savez_compressed(destination / "graphs.npz", **data.graphs)
    metadata = {
        "node_ids": list(data.node_ids),
        "feature_names": list(data.feature_names),
        "target_features": list(data.target_features),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    frequency = validate_regular_timestamps(data.timestamps)
    node_bytes = json.dumps(list(data.node_ids), ensure_ascii=False).encode("utf-8")
    manifest: dict[str, Any] = {
        "dataset": "POWER-GRID",
        "shape": list(data.shape),
        "dtype": str(data.values.dtype),
        "time_start": str(data.timestamps[0]),
        "time_end": str(data.timestamps[-1]),
        "frequency_minutes": int(frequency / np.timedelta64(1, "m")),
        "num_observed": int(data.observed.sum()),
        "num_missing": int(data.observed.size - data.observed.sum()),
        "node_order_sha256": fingerprint_bytes(node_bytes),
        "graph_names": list(data.graphs),
        "graph_provenance": provenance,
        "source_files": {
            "values": {
                "path": str(Path(csv_path)),
                "sha256": fingerprint_file(csv_path),
            }
        },
        "processed_files": {},
    }
    filenames = (
        "values.npy",
        "observed.npy",
        "timestamps.npy",
        "graphs.npz",
        "metadata.json",
    )
    for filename in filenames:
        manifest["processed_files"][filename] = fingerprint_file(destination / filename)
    (destination / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_power_grid(config: Mapping[str, Any]) -> dict[str, Any]:
    """按显式本地配置读取并保存电力宽表。"""

    if config.get("dataset") != "POWER-GRID":
        raise ValueError(f"unsupported dataset: {config.get('dataset')}")
    csv_path = Path(str(config["csv_path"]))
    data = load_power_grid_wide(
        csv_path,
        timestamp_column=str(config["timestamp_column"]),
        node_ids=config["node_ids"],
        feature_columns=config["feature_columns"],
        target_features=config["target_features"],
        fill_value=float(config.get("fill_value", 0.0)),
    )
    return save_power_grid_processed(data, config["output_dir"], csv_path)
