import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsproj_stf.data.datasets.power_grid import (
    load_power_grid_wide,
    prepare_power_grid,
    save_power_grid_processed,
)
from tsproj_stf.data.graphs import correlation_graph, distance_graph, physical_graph
from tsproj_stf.data.schema import SpatioTemporalData
from tsproj_stf.data.split import chronological_split
from tsproj_stf.evaluation.quantiles import pinball_loss
from tsproj_stf.experiments.artifacts import fingerprint_bytes
from tsproj_stf.experiments.config import ExperimentConfig
from tsproj_stf.experiments.runner import run_experiment
from tsproj_stf.models.quantile_head import QuantileHead


def write_power_csv(tmp_path: Path, length: int = 80) -> Path:
    time = np.arange(length, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=length, freq="15min"),
            "load_a": 20.0 + np.sin(time / 4.0),
            "load_b": 30.0 + np.sin(time / 4.0) * 2.0,
            "load_c": 25.0 + np.cos(time / 5.0),
        }
    )
    frame.loc[10, "load_b"] = np.nan
    path = tmp_path / "power.csv"
    frame.to_csv(path, index=False)
    return path


def build_processed_power_fixture(tmp_path: Path) -> Path:
    csv_path = write_power_csv(tmp_path)
    data = load_power_grid_wide(
        csv_path,
        timestamp_column="timestamp",
        node_ids=("a", "b", "c"),
        feature_columns={
            "load": {"a": "load_a", "b": "load_b", "c": "load_c"}
        },
        target_features=("load",),
    )
    split = chronological_split(data.num_timesteps, (0.6, 0.2, 0.2))
    graphs = {
        "physical": physical_graph(
            data.node_ids,
            (("a", "b", 1.0), ("b", "c", 1.0)),
            directed=False,
        ),
        "electrical_distance": distance_graph(
            data.node_ids,
            np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]),
            distance_node_ids=("a", "b", "c"),
            sigma=2.0,
            threshold=2.0,
        ),
        "train_correlation": correlation_graph(
            data,
            target_feature="load",
            train_slice=split.train,
            top_k=2,
            threshold=0.1,
        ),
    }
    data_with_graphs = SpatioTemporalData(
        values=data.values,
        observed=data.observed,
        timestamps=data.timestamps,
        node_ids=data.node_ids,
        feature_names=data.feature_names,
        target_features=data.target_features,
        graphs=graphs,
    )
    output = tmp_path / "processed"
    node_order = list(data.node_ids)
    graph_provenance = {
        "physical": {
            "kind": "physical",
            "node_ids": node_order,
            "source_sha256": fingerprint_bytes(b"a,b,1.0\nb,c,1.0\n"),
            "directed": False,
        },
        "electrical_distance": {
            "kind": "distance",
            "node_ids": node_order,
            "source_sha256": fingerprint_bytes(b"synthetic-distance-matrix"),
            "sigma": 2.0,
            "threshold": 2.0,
        },
        "train_correlation": {
            "kind": "correlation",
            "node_ids": node_order,
            "source_sha256": fingerprint_bytes(data.values[split.train].tobytes()),
            "target_feature": "load",
            "train_bounds": [split.train.start, split.train.stop],
            "top_k": 2,
            "threshold": 0.1,
        },
    }
    save_power_grid_processed(
        data_with_graphs,
        output,
        csv_path,
        graph_provenance=graph_provenance,
    )
    return output


def experiment_config(
    tmp_path: Path,
    data_path: Path,
    *,
    model: str,
) -> ExperimentConfig:
    model_params: dict[str, object] = {
        "target_feature": "load",
        "null_value": 0.0,
        "fallback_value": 0.0,
    }
    if model == "graph_wavenet":
        model_params.update(
            {
                "graph_mode": "fixed",
                "graph_name": "physical",
                "epochs": 1,
                "test_interval": 2,
                "batch_size": 8,
                "patience": 2,
                "residual_channels": 4,
                "dilation_channels": 4,
                "skip_channels": 8,
                "end_channels": 16,
                "kernel_size": 2,
                "dilations": [1, 2],
                "diffusion_order": 2,
                "adaptive_embedding_dim": 4,
                "dropout": 0.0,
            }
        )
    return ExperimentConfig(
        name=f"power_{model}",
        dataset="POWER-GRID",
        data_path=str(data_path),
        model=model,
        input_len=6,
        output_len=3,
        split_ratios=(0.6, 0.2, 0.2),
        horizons=(1, 3),
        seed=42,
        results_dir=str(tmp_path / "results"),
        rescale=True,
        model_params=model_params,
    )


def test_power_grid_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    data_path = build_processed_power_fixture(tmp_path)

    persistence = run_experiment(
        experiment_config(tmp_path, data_path, model="persistence")
    )
    graph_wavenet = run_experiment(
        experiment_config(tmp_path, data_path, model="graph_wavenet")
    )
    head = QuantileHead(in_features=4, quantiles=(0.1, 0.5, 0.9))
    quantile_prediction = head(torch.randn(2, 3, 3, 4))
    quantile_loss = pinball_loss(
        quantile_prediction,
        torch.ones(2, 3, 3),
        torch.ones(2, 3, 3, dtype=torch.bool),
        quantiles=(0.1, 0.5, 0.9),
    )

    with np.load(data_path / "graphs.npz") as archive:
        assert set(archive.files) == {
            "physical",
            "electrical_distance",
            "train_correlation",
        }
    manifest = json.loads((data_path / "data_manifest.json").read_text())
    assert manifest["dataset"] == "POWER-GRID"
    assert set(manifest["graph_provenance"]) == {
        "physical",
        "electrical_distance",
        "train_correlation",
    }
    assert manifest["graph_provenance"]["train_correlation"]["train_bounds"] == [0, 48]
    assert persistence.prediction_shape == (8, 3, 3)
    assert graph_wavenet.prediction_shape == (8, 3, 3)
    assert list((graph_wavenet.run_dir / "checkpoint").rglob("*.pt"))
    assert torch.isfinite(quantile_loss)
    assert torch.all(quantile_prediction[..., :-1] <= quantile_prediction[..., 1:])


def test_prepare_power_grid_uses_explicit_config_mapping(tmp_path: Path) -> None:
    csv_path = write_power_csv(tmp_path)
    output = tmp_path / "prepared"

    manifest = prepare_power_grid(
        {
            "dataset": "POWER-GRID",
            "csv_path": str(csv_path),
            "output_dir": str(output),
            "timestamp_column": "timestamp",
            "node_ids": ["a", "b", "c"],
            "feature_columns": {
                "load": {"a": "load_a", "b": "load_b", "c": "load_c"}
            },
            "target_features": ["load"],
            "fill_value": 0.0,
        }
    )

    assert manifest["shape"] == [80, 3, 1]
    assert (output / "values.npy").is_file()
