import numpy as np
import pytest

from tsproj_stf.data.schema import SpatioTemporalData


def make_data(**overrides: object) -> SpatioTemporalData:
    values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    kwargs: dict[str, object] = {
        "values": values,
        "observed": np.ones_like(values, dtype=bool),
        "timestamps": np.arange(4, dtype=np.int64),
        "node_ids": ("n0", "n1", "n2"),
        "feature_names": ("load", "temperature"),
        "target_features": ("load",),
        "graphs": {"physical": np.eye(3, dtype=np.float32)},
    }
    kwargs.update(overrides)
    return SpatioTemporalData(**kwargs)


def test_accepts_valid_spatio_temporal_data() -> None:
    data = make_data()

    assert data.shape == (4, 3, 2)
    assert data.num_timesteps == 4
    assert data.num_nodes == 3
    assert data.num_features == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("values", np.zeros((4, 3), dtype=np.float32), "values must be 3-dimensional"),
        ("observed", np.ones((4, 3, 1), dtype=bool), "observed shape"),
        ("timestamps", np.arange(3), "timestamps length"),
        ("node_ids", ("n0", "n1"), "node_ids length"),
        ("feature_names", ("load",), "feature_names length"),
        ("node_ids", ("n0", "n0", "n2"), "node_ids must be unique"),
        ("target_features", ("missing",), "unknown target feature"),
        ("graphs", {"bad": np.eye(2, dtype=np.float32)}, "graph 'bad' shape"),
    ],
)
def test_rejects_invalid_contract(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_data(**{field: value})


def test_converts_arrays_to_contract_dtypes_without_copying_semantics() -> None:
    data = make_data(
        values=np.ones((4, 3, 2), dtype=np.float64),
        observed=np.ones((4, 3, 2), dtype=np.int8),
        graphs={"physical": np.eye(3, dtype=np.float64)},
    )

    assert data.values.dtype == np.float32
    assert data.observed.dtype == np.bool_
    assert data.graphs["physical"].dtype == np.float32
