"""电力场景显式多图构建。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tsproj_stf.data.schema import SpatioTemporalData


def physical_graph(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    *,
    directed: bool,
) -> NDArray[np.float32]:
    """按 canonical node order 从加权 edge list 构造物理图。"""

    ordered_nodes = tuple(str(node_id) for node_id in node_ids)
    if not ordered_nodes or len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError("node_ids must be non-empty and unique")
    node_index = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    adjacency = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=np.float32)
    for source, target, weight in edges:
        if source not in node_index or target not in node_index:
            raise ValueError(f"edge contains unknown node: {(source, target)}")
        numeric_weight = float(weight)
        if not np.isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError("edge weights must be finite and non-negative")
        adjacency[node_index[source], node_index[target]] = numeric_weight
        if not directed:
            adjacency[node_index[target], node_index[source]] = numeric_weight
    return adjacency


def distance_graph(
    node_ids: Sequence[str],
    distances: NDArray[np.floating],
    *,
    distance_node_ids: Sequence[str],
    sigma: float,
    threshold: float | None = None,
) -> NDArray[np.float32]:
    """将带显式节点顺序的距离矩阵转为 canonical 高斯相似图。"""

    ordered_nodes = tuple(str(node_id) for node_id in node_ids)
    source_nodes = tuple(str(node_id) for node_id in distance_node_ids)
    if len(set(ordered_nodes)) != len(ordered_nodes) or set(source_nodes) != set(
        ordered_nodes
    ):
        raise ValueError("distance node IDs must match unique canonical node_ids")
    distance_array = np.asarray(distances, dtype=np.float64)
    if distance_array.shape != (len(source_nodes), len(source_nodes)):
        raise ValueError("distance matrix shape must match distance_node_ids")
    if not np.all(np.isfinite(distance_array)) or np.any(distance_array < 0):
        raise ValueError("distances must be finite and non-negative")
    if sigma <= 0 or (threshold is not None and threshold < 0):
        raise ValueError("sigma must be positive and threshold must be non-negative")
    source_index = {node_id: index for index, node_id in enumerate(source_nodes)}
    reorder = [source_index[node_id] for node_id in ordered_nodes]
    canonical_distances = distance_array[np.ix_(reorder, reorder)]
    graph = np.exp(-np.square(canonical_distances / sigma))
    if threshold is not None:
        graph = np.where(canonical_distances <= threshold, graph, 0.0)
    np.fill_diagonal(graph, 0.0)
    return graph.astype(np.float32)


def correlation_graph(
    data: SpatioTemporalData,
    *,
    target_feature: str,
    train_slice: slice,
    top_k: int | None = None,
    threshold: float = 0.0,
) -> NDArray[np.float32]:
    """仅从显式训练区间构造绝对 Pearson 相关图。"""

    if target_feature not in data.feature_names:
        raise ValueError(f"unknown target feature: {target_feature!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if top_k is not None and not 1 <= top_k < data.num_nodes:
        raise ValueError("top_k must be between 1 and num_nodes - 1")
    indices = np.arange(data.num_timesteps)[train_slice]
    if len(indices) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("train_slice must select at least two contiguous timestamps")
    feature_index = data.feature_names.index(target_feature)
    values = data.values[indices, :, feature_index]
    observed = data.observed[indices, :, feature_index]
    graph = np.zeros((data.num_nodes, data.num_nodes), dtype=np.float32)
    for source in range(data.num_nodes):
        for target in range(source + 1, data.num_nodes):
            valid = observed[:, source] & observed[:, target]
            if valid.sum() < 2:
                continue
            source_values = values[valid, source]
            target_values = values[valid, target]
            if np.std(source_values) == 0 or np.std(target_values) == 0:
                continue
            correlation = abs(float(np.corrcoef(source_values, target_values)[0, 1]))
            if np.isfinite(correlation):
                graph[source, target] = correlation
                graph[target, source] = correlation
    graph[graph < threshold] = 0.0
    if top_k is not None:
        sparse_graph = np.zeros_like(graph)
        node_indices = np.arange(data.num_nodes)
        for source in range(data.num_nodes):
            candidates = node_indices[graph[source] > 0]
            ranked = sorted(
                candidates,
                key=lambda target: (-float(graph[source, target]), int(target)),
            )
            selected = ranked[:top_k]
            sparse_graph[source, selected] = graph[source, selected]
        graph = sparse_graph
    return graph
