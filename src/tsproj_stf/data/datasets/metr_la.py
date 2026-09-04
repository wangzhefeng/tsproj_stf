"""METR-LA 数据准备。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsproj_stf.data.schema import SpatioTemporalData
from tsproj_stf.data.validation import validate_regular_timestamps
from tsproj_stf.experiments.artifacts import fingerprint_bytes, fingerprint_file


class _RestrictedGraphUnpickler(pickle.Unpickler):
    """仅允许官方 NumPy ndarray pickle 所需的构造器。"""

    _ALLOWED_GLOBALS = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return super().find_class(module, name)


def _load_graph(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open("rb") as handle:
        payload = _RestrictedGraphUnpickler(handle, encoding="latin1").load()
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ValueError("METR-LA graph pickle must contain sensor_ids, mapping, adjacency")
    sensor_ids, sensor_to_index, adjacency = payload
    node_ids = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in sensor_ids
    )
    if set(sensor_to_index) != set(sensor_ids):
        raise ValueError("graph sensor mapping does not match sensor IDs")
    graph = np.asarray(adjacency, dtype=np.float32)
    if graph.shape != (len(node_ids), len(node_ids)):
        raise ValueError(f"graph shape {graph.shape} does not match {len(node_ids)} sensor IDs")
    return node_ids, graph


def load_metr_la_raw(
    csv_path: str | Path,
    graph_path: str | Path,
    *,
    null_value: float = 0.0,
) -> SpatioTemporalData:
    """读取 CSV 速度序列和 DCRNN 邻接矩阵。"""

    csv_file = Path(csv_path)
    graph_file = Path(graph_path)
    frame = pd.read_csv(csv_file)
    if frame.shape[1] < 2:
        raise ValueError("METR-LA CSV must contain timestamp and sensor columns")

    timestamp_column = "timestamp" if "timestamp" in frame.columns else str(frame.columns[0])
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise").to_numpy(
        dtype="datetime64[ns]"
    )
    validate_regular_timestamps(timestamps)

    node_ids, graph = _load_graph(graph_file)
    csv_node_columns = tuple(str(column) for column in frame.columns if column != timestamp_column)
    if set(csv_node_columns) != set(node_ids):
        missing = sorted(set(node_ids) - set(csv_node_columns))
        extra = sorted(set(csv_node_columns) - set(node_ids))
        raise ValueError(
            "CSV node columns do not match graph node IDs: "
            f"missing={missing}, extra={extra}"
        )

    values_2d = frame.loc[:, list(node_ids)].to_numpy(dtype=np.float32)
    observed_2d = np.isfinite(values_2d) & ~np.isclose(values_2d, null_value)
    values_2d = np.where(np.isfinite(values_2d), values_2d, null_value)
    return SpatioTemporalData(
        values=values_2d[:, :, None],
        observed=observed_2d[:, :, None],
        timestamps=timestamps,
        node_ids=node_ids,
        feature_names=("speed",),
        target_features=("speed",),
        graphs={"physical": graph},
    )


def save_metr_la_processed(
    data: SpatioTemporalData,
    output_dir: str | Path,
    csv_path: str | Path,
    graph_path: str | Path,
) -> dict[str, Any]:
    """保存后端无关的 processed 数据和可追溯 manifest。"""

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
    frequency_minutes = int(frequency / np.timedelta64(1, "m"))
    node_bytes = json.dumps(list(data.node_ids), ensure_ascii=False).encode("utf-8")
    manifest: dict[str, Any] = {
        "dataset": "METR-LA",
        "shape": list(data.shape),
        "dtype": str(data.values.dtype),
        "time_start": str(data.timestamps[0]),
        "time_end": str(data.timestamps[-1]),
        "frequency_minutes": frequency_minutes,
        "num_observed": int(data.observed.sum()),
        "num_missing": int(data.observed.size - data.observed.sum()),
        "node_order_sha256": fingerprint_bytes(node_bytes),
        "source_files": {
            "values": {
                "path": str(Path(csv_path)),
                "sha256": fingerprint_file(csv_path),
            },
            "graph": {
                "path": str(Path(graph_path)),
                "sha256": fingerprint_file(graph_path),
            },
        },
        "processed_files": {},
    }
    for filename in ("values.npy", "observed.npy", "timestamps.npy", "graphs.npz", "metadata.json"):
        manifest["processed_files"][filename] = fingerprint_file(destination / filename)
    (destination / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_metr_la_processed(input_dir: str | Path) -> SpatioTemporalData:
    """读取 `save_metr_la_processed` 产生的数据。"""

    source = Path(input_dir)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    with np.load(source / "graphs.npz") as graph_archive:
        graphs = {name: graph_archive[name] for name in graph_archive.files}
    return SpatioTemporalData(
        values=np.load(source / "values.npy"),
        observed=np.load(source / "observed.npy"),
        timestamps=np.load(source / "timestamps.npy"),
        node_ids=tuple(metadata["node_ids"]),
        feature_names=tuple(metadata["feature_names"]),
        target_features=tuple(metadata["target_features"]),
        graphs=graphs,
    )
