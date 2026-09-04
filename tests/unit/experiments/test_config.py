from pathlib import Path

import pytest
import yaml

from tsproj_stf.experiments.config import ExperimentConfig, load_experiment_config


def valid_mapping() -> dict[str, object]:
    return {
        "name": "metr_la_stid",
        "dataset": "METR-LA",
        "data_path": "data/processed/METR-LA",
        "model": "stid",
        "input_len": 12,
        "output_len": 12,
        "split_ratios": [0.7, 0.1, 0.2],
        "horizons": [3, 6, 12],
        "seed": 42,
        "results_dir": "results",
        "rescale": True,
        "model_params": {"hidden_size": 32},
    }


def test_loads_valid_yaml_into_immutable_config(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(valid_mapping()), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.name == "metr_la_stid"
    assert config.data_path == "data/processed/METR-LA"
    assert config.split_ratios == (0.7, 0.1, 0.2)
    assert config.horizons == (3, 6, 12)
    assert config.to_dict()["model_params"] == {"hidden_size": 32}


def test_rejects_unknown_config_field() -> None:
    mapping = valid_mapping()
    mapping["typo_field"] = True

    with pytest.raises(ValueError, match="unknown experiment config fields"):
        ExperimentConfig.from_mapping(mapping)


def test_rejects_horizon_beyond_output_length() -> None:
    mapping = valid_mapping()
    mapping["horizons"] = [3, 13]

    with pytest.raises(ValueError, match="outside 1..12"):
        ExperimentConfig.from_mapping(mapping)


def test_rejects_invalid_split_ratios() -> None:
    mapping = valid_mapping()
    mapping["split_ratios"] = [0.7, 0.2, 0.2]

    with pytest.raises(ValueError, match="sum to 1"):
        ExperimentConfig.from_mapping(mapping)
