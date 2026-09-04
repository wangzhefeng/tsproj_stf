"""实验配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isclose
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from tsproj_stf.evaluation.horizons import horizon_to_index


@dataclass(frozen=True)
class ExperimentConfig:
    """经过校验的单次实验配置。"""

    name: str
    dataset: str
    data_path: str
    model: str
    input_len: int
    output_len: int
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
    horizons: tuple[int, ...] = (3, 6, 12)
    seed: int = 42
    results_dir: str = "results"
    rescale: bool = True
    model_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.dataset or not self.data_path or not self.model:
            raise ValueError("name, dataset, data_path, and model must be non-empty")
        if self.input_len <= 0 or self.output_len <= 0:
            raise ValueError("input_len and output_len must be positive")
        if len(self.split_ratios) != 3 or any(ratio <= 0 for ratio in self.split_ratios):
            raise ValueError("split_ratios must contain three positive values")
        if not isclose(sum(self.split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"split_ratios must sum to 1, got {sum(self.split_ratios)}")
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        for horizon in self.horizons:
            horizon_to_index(horizon, self.output_len)

        object.__setattr__(self, "split_ratios", tuple(float(x) for x in self.split_ratios))
        object.__setattr__(self, "horizons", tuple(int(x) for x in self.horizons))
        object.__setattr__(self, "model_params", MappingProxyType(dict(self.model_params)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExperimentConfig:
        known_fields = {item.name for item in fields(cls)}
        unknown_fields = set(mapping) - known_fields
        if unknown_fields:
            raise ValueError(f"unknown experiment config fields: {sorted(unknown_fields)}")

        converted = dict(mapping)
        if "split_ratios" in converted:
            converted["split_ratios"] = tuple(converted["split_ratios"])
        if "horizons" in converted:
            converted["horizons"] = tuple(converted["horizons"])
        return cls(**converted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "model": self.model,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "split_ratios": list(self.split_ratios),
            "horizons": list(self.horizons),
            "seed": self.seed,
            "results_dir": self.results_dir,
            "rescale": self.rescale,
            "model_params": dict(self.model_params),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """从 YAML 加载并校验实验配置。"""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a mapping: {config_path}")
    return ExperimentConfig.from_mapping(payload)
