import numpy as np
import pytest

from tsproj_stf.data.graphs import correlation_graph, distance_graph, physical_graph
from tsproj_stf.data.schema import SpatioTemporalData


def test_physical_graph_supports_directed_and_undirected_edges() -> None:
    node_ids = ("c", "a", "b")
    edges = (("a", "b", 2.0), ("c", "a", 1.0))

    directed = physical_graph(node_ids, edges, directed=True)
    undirected = physical_graph(node_ids, edges, directed=False)

    np.testing.assert_array_equal(
        directed,
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(undirected, directed + directed.T)


def test_physical_graph_rejects_unknown_nodes() -> None:
    with pytest.raises(ValueError, match="unknown node"):
        physical_graph(("a", "b"), (("a", "missing", 1.0),), directed=True)


def test_distance_graph_reorders_nodes_and_applies_thresholded_gaussian() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    graph = distance_graph(
        ("c", "a", "b"),
        distances,
        distance_node_ids=("a", "b", "c"),
        sigma=2.0,
        threshold=2.0,
    )

    assert graph.shape == (3, 3)
    assert graph[0, 1] == pytest.approx(np.exp(-1.0))
    assert graph[1, 2] == pytest.approx(np.exp(-0.25))
    assert graph[0, 2] == 0.0
    assert np.all(np.diag(graph) == 0.0)


def make_correlation_data(test_tail: np.ndarray) -> SpatioTemporalData:
    train = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, 2.0],
            [4.0, 4.0, 2.0],
        ]
    )
    values = np.concatenate([train, test_tail], axis=0)[:, :, None]
    return SpatioTemporalData(
        values=values,
        observed=np.ones_like(values, dtype=bool),
        timestamps=np.arange(len(values)),
        node_ids=("a", "b", "c"),
        feature_names=("load",),
        target_features=("load",),
        graphs={},
    )


def test_correlation_graph_uses_only_explicit_train_slice() -> None:
    first = make_correlation_data(np.array([[100.0, -100.0, 0.0], [200.0, -200.0, 1.0]]))
    second = make_correlation_data(np.array([[-9.0, 5.0, 90.0], [-8.0, 4.0, 80.0]]))

    first_graph = correlation_graph(first, target_feature="load", train_slice=slice(0, 4))
    second_graph = correlation_graph(second, target_feature="load", train_slice=slice(0, 4))

    np.testing.assert_array_equal(first_graph, second_graph)
    assert first_graph[0, 1] == pytest.approx(1.0)


def test_correlation_graph_sparsification_is_reproducible() -> None:
    data = make_correlation_data(np.zeros((2, 3)))

    top_k_graph = correlation_graph(
        data,
        target_feature="load",
        train_slice=slice(0, 4),
        top_k=1,
        threshold=0.0,
    )
    threshold_graph = correlation_graph(
        data,
        target_feature="load",
        train_slice=slice(0, 4),
        top_k=2,
        threshold=0.9,
    )

    np.testing.assert_allclose(
        top_k_graph,
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.8944272, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        threshold_graph,
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
