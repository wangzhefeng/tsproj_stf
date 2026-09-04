from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsproj_stf.data.datasets.power_grid import load_power_grid_wide


def write_power_csv(
    tmp_path: Path,
    *,
    duplicate: bool = False,
    irregular: bool = False,
) -> Path:
    timestamps = [
        "2026-01-01 00:10",
        "2026-01-01 00:00",
        "2026-01-01 00:05",
    ]
    if duplicate:
        timestamps[-1] = "2026-01-01 00:00"
    if irregular:
        timestamps[0] = "2026-01-01 00:15"
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "load_a": [12.0, 10.0, np.nan],
            "load_b": [22.0, 20.0, 21.0],
            "price_a": [102.0, 100.0, 101.0],
            "price_b": [202.0, 200.0, 201.0],
        }
    )
    path = tmp_path / "power.csv"
    frame.to_csv(path, index=False)
    return path


def load_fixture(path: Path):
    return load_power_grid_wide(
        path,
        timestamp_column="time",
        node_ids=("b", "a"),
        feature_columns={
            "load": {"b": "load_b", "a": "load_a"},
            "price": {"b": "price_b", "a": "price_a"},
        },
        target_features=("load",),
    )


def test_loads_sorted_multifeature_data_in_explicit_node_order(tmp_path: Path) -> None:
    data = load_fixture(write_power_csv(tmp_path))

    assert data.shape == (3, 2, 2)
    assert data.node_ids == ("b", "a")
    assert data.feature_names == ("load", "price")
    assert data.target_features == ("load",)
    np.testing.assert_array_equal(data.values[:, 0, 0], [20.0, 21.0, 22.0])
    assert not data.observed[1, 1, 0]
    assert data.values[1, 1, 0] == 0.0


def test_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timestamps must be unique"):
        load_fixture(write_power_csv(tmp_path, duplicate=True))


def test_rejects_irregular_timestamps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing and regularly spaced"):
        load_fixture(write_power_csv(tmp_path, irregular=True))


def test_requires_every_feature_to_map_every_node(tmp_path: Path) -> None:
    path = write_power_csv(tmp_path)

    with pytest.raises(ValueError, match="must map every node exactly once"):
        load_power_grid_wide(
            path,
            timestamp_column="time",
            node_ids=("a", "b"),
            feature_columns={"load": {"a": "load_a"}},
            target_features=("load",),
        )


def test_does_not_guess_undefined_target_features(tmp_path: Path) -> None:
    path = write_power_csv(tmp_path)

    with pytest.raises(ValueError, match="subset of feature_columns"):
        load_power_grid_wide(
            path,
            timestamp_column="time",
            node_ids=("a", "b"),
            feature_columns={"load": {"a": "load_a", "b": "load_b"}},
            target_features=("load", "price"),
        )
