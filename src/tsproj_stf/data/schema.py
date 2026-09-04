"""时空数据契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpatioTemporalData:
    """与训练后端无关的标准时空数据。"""

    values: NDArray[np.floating]
    observed: NDArray[np.bool_]
    timestamps: NDArray[np.generic]
    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_features: tuple[str, ...]
    graphs: Mapping[str, NDArray[np.floating]]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        observed = np.asarray(self.observed)
        timestamps = np.asarray(self.timestamps)

        if values.ndim != 3:
            raise ValueError(
                f"values must be 3-dimensional [time, node, feature], got {values.shape}"
            )
        if observed.shape != values.shape:
            raise ValueError(
                f"observed shape {observed.shape} must equal values shape {values.shape}"
            )
        if timestamps.ndim != 1 or len(timestamps) != values.shape[0]:
            raise ValueError(
                f"timestamps length {len(timestamps)} must equal time dimension {values.shape[0]}"
            )
        if len(self.node_ids) != values.shape[1]:
            raise ValueError(
                f"node_ids length {len(self.node_ids)} must equal node dimension {values.shape[1]}"
            )
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if len(self.feature_names) != values.shape[2]:
            raise ValueError(
                "feature_names length "
                f"{len(self.feature_names)} must equal feature dimension {values.shape[2]}"
            )
        unknown_targets = set(self.target_features) - set(self.feature_names)
        if unknown_targets:
            raise ValueError(f"unknown target feature(s): {sorted(unknown_targets)}")

        graph_arrays: dict[str, NDArray[np.float32]] = {}
        expected_graph_shape = (values.shape[1], values.shape[1])
        for name, graph in self.graphs.items():
            graph_array = np.asarray(graph)
            if graph_array.shape != expected_graph_shape:
                raise ValueError(
                    f"graph '{name}' shape {graph_array.shape} must equal {expected_graph_shape}"
                )
            graph_arrays[name] = graph_array.astype(np.float32, copy=False)

        object.__setattr__(self, "values", values.astype(np.float32, copy=False))
        object.__setattr__(self, "observed", observed.astype(bool, copy=False))
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "node_ids", tuple(str(item) for item in self.node_ids))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "target_features", tuple(self.target_features))
        object.__setattr__(self, "graphs", MappingProxyType(graph_arrays))

    @property
    def shape(self) -> tuple[int, int, int]:
        """返回 `[time, node, feature]` shape。"""

        return self.values.shape

    @property
    def num_timesteps(self) -> int:
        return self.values.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.values.shape[1]

    @property
    def num_features(self) -> int:
        return self.values.shape[2]
