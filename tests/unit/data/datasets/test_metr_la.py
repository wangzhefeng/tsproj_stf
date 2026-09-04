import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsproj_stf.data.datasets.metr_la import (
    load_metr_la_processed,
    load_metr_la_raw,
    save_metr_la_processed,
)


def write_fixture(
    tmp_path: Path,
    *,
    irregular: bool = False,
    mismatch: bool = False,
) -> tuple[Path, Path]:
    timestamps = pd.to_datetime(
        ["2012-03-01 00:00", "2012-03-01 00:05", "2012-03-01 00:10"]
    )
    if irregular:
        timestamps = timestamps.to_series().reset_index(drop=True)
        timestamps.iloc[-1] = pd.Timestamp("2012-03-01 00:15")
    columns = ["s1", "wrong"] if mismatch else ["s1", "s2"]
    frame = pd.DataFrame([[10.0, 20.0], [0.0, 21.0], [12.0, 22.0]], columns=columns)
    frame.insert(0, "timestamp", timestamps)
    csv_path = tmp_path / "METR-LA.csv"
    frame.to_csv(csv_path, index=False)

    graph_path = tmp_path / "adj_mx.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(
            (["s1", "s2"], {"s1": 0, "s2": 1}, np.array([[0.0, 0.5], [0.25, 0.0]])),
            handle,
        )
    return csv_path, graph_path


def test_loads_csv_and_graph_in_graph_node_order(tmp_path: Path) -> None:
    csv_path, graph_path = write_fixture(tmp_path)

    data = load_metr_la_raw(csv_path, graph_path)

    assert data.shape == (3, 2, 1)
    assert data.node_ids == ("s1", "s2")
    assert data.feature_names == ("speed",)
    assert not data.observed[1, 0, 0]
    np.testing.assert_array_equal(data.graphs["physical"], [[0.0, 0.5], [0.25, 0.0]])


def test_rejects_csv_nodes_that_do_not_match_graph(tmp_path: Path) -> None:
    csv_path, graph_path = write_fixture(tmp_path, mismatch=True)

    with pytest.raises(ValueError, match="CSV node columns do not match graph node IDs"):
        load_metr_la_raw(csv_path, graph_path)


def test_rejects_irregular_time_axis(tmp_path: Path) -> None:
    csv_path, graph_path = write_fixture(tmp_path, irregular=True)

    with pytest.raises(ValueError, match="strictly increasing and regularly spaced"):
        load_metr_la_raw(csv_path, graph_path)


def test_rejects_pickle_globals_outside_numpy_allowlist(tmp_path: Path) -> None:
    csv_path, graph_path = write_fixture(tmp_path)
    graph_path.write_bytes(pickle.dumps(Path("untrusted")))

    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        load_metr_la_raw(csv_path, graph_path)


def test_processed_round_trip_and_manifest(tmp_path: Path) -> None:
    csv_path, graph_path = write_fixture(tmp_path)
    data = load_metr_la_raw(csv_path, graph_path)
    output_dir = tmp_path / "processed"

    manifest = save_metr_la_processed(data, output_dir, csv_path, graph_path)
    restored = load_metr_la_processed(output_dir)

    assert restored.shape == data.shape
    assert restored.node_ids == data.node_ids
    np.testing.assert_array_equal(restored.values, data.values)
    np.testing.assert_array_equal(restored.observed, data.observed)
    assert manifest["shape"] == [3, 2, 1]
    assert manifest["frequency_minutes"] == 5
    assert json.loads((output_dir / "data_manifest.json").read_text()) == manifest
