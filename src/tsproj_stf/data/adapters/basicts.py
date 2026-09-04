"""BasicTS 数据集适配。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from basicts.data.base_dataset import BasicTSDataset
from basicts.runners.taskflow.forecasting_taskflow import BasicTSForecastingTaskFlow
from basicts.scaler import BasicTSScaler
from basicts.utils import BasicTSMode

from tsproj_stf.data.scaling import ZScoreScaler
from tsproj_stf.data.split import chronological_split


class ProjectForecastingTaskFlow(BasicTSForecastingTaskFlow):
    """强制 BasicTS 使用项目 observed mask，不从数值 sentinel 重建缺失。"""

    def preprocess(self, runner: Any, data: dict[str, Any]) -> dict[str, Any]:
        inputs_mask = data["inputs_observed"].to(dtype=torch.bool)
        targets_mask = data["targets_observed"].to(dtype=torch.bool)
        if runner.scaler is not None:
            data["inputs"] = runner.scaler.transform(data["inputs"], inputs_mask)
            data["targets"] = runner.scaler.transform(data["targets"], targets_mask)
        null_value = torch.as_tensor(
            runner.cfg.null_to_num,
            dtype=data["inputs"].dtype,
            device=data["inputs"].device,
        )
        data["inputs"] = torch.where(inputs_mask, data["inputs"], null_value)
        data["targets"] = torch.where(targets_mask, data["targets"], null_value)
        data["targets_mask"] = targets_mask
        return data


class ProjectBasicTSScaler(BasicTSScaler):
    """使用项目 observed mask 预先拟合、对 BasicTS 暴露 Torch 接口的 scaler。"""

    def __init__(
        self,
        data_file_path: str | Path,
        split_ratios: tuple[float, float, float],
        target_feature: str,
        rescale: bool,
    ) -> None:
        source = Path(data_file_path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        feature_names = tuple(metadata["feature_names"])
        if target_feature not in feature_names:
            raise ValueError(
                f"unknown target feature {target_feature!r}; available={list(feature_names)}"
            )
        target_index = feature_names.index(target_feature)
        values = np.load(source / "values.npy")[:, :, target_index]
        observed = np.load(source / "observed.npy")[:, :, target_index]
        train_slice = chronological_split(len(values), split_ratios).train
        project_scaler = ZScoreScaler.fit(values[train_slice], observed[train_slice])
        super().__init__(
            norm_each_channel=True,
            rescale=rescale,
            stats={
                "mean": torch.from_numpy(project_scaler.stats.mean.copy()),
                "std": torch.from_numpy(project_scaler.stats.std.copy()),
            },
        )

    def fit(self, data: np.ndarray | torch.Tensor) -> None:
        """统计量已从项目 train mask 拟合；BasicTS 的二次 fit 必须是 no-op。"""

    def transform(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = self.stats["mean"].to(input_data.device)
        std = self.stats["std"].to(input_data.device)
        transformed = (input_data - mean) / std
        return torch.where(mask, transformed, input_data) if mask is not None else transformed

    def inverse_transform(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = self.stats["mean"].to(input_data.device)
        std = self.stats["std"].to(input_data.device)
        restored = input_data * std + mean
        return torch.where(mask, restored, input_data) if mask is not None else restored


class ProjectForecastingDataset(BasicTSDataset):
    """从项目 processed 数据构造 split 内 BasicTS 滑窗。"""

    def __init__(
        self,
        data_file_path: str | Path,
        dataset_name: str,
        input_len: int,
        output_len: int,
        mode: BasicTSMode | str,
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        target_feature: str = "speed",
        null_value: float = 0.0,
        use_timestamps: bool = True,
        memmap: bool = False,
    ) -> None:
        super().__init__(dataset_name=dataset_name, mode=mode, memmap=memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.use_timestamps = use_timestamps

        source = Path(data_file_path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        feature_names = tuple(metadata["feature_names"])
        if target_feature not in feature_names:
            raise ValueError(
                f"unknown target feature {target_feature!r}; available={list(feature_names)}"
            )
        target_index = feature_names.index(target_feature)

        mmap_mode = "r" if memmap else None
        values = np.load(source / "values.npy", mmap_mode=mmap_mode)
        observed = np.load(source / "observed.npy", mmap_mode=mmap_mode)
        timestamps = np.load(source / "timestamps.npy", mmap_mode=mmap_mode)
        bounds = chronological_split(len(values), split_ratios)
        mode_name = str(mode)
        if mode_name == BasicTSMode.EVAL:
            mode_name = BasicTSMode.TEST.value
        if mode_name not in {"train", "val", "test"}:
            raise ValueError(f"unsupported forecasting mode: {mode}")
        split_slice = getattr(bounds, mode_name)

        target_values = np.asarray(values[split_slice, :, target_index], dtype=np.float32)
        target_observed = np.asarray(observed[split_slice, :, target_index], dtype=bool)
        self._observed = target_observed
        self._data = np.where(target_observed, target_values, null_value).astype(np.float32)
        self._timestamps = self._encode_timestamps(timestamps[split_slice])

        minimum_length = input_len + output_len
        if len(self._data) < minimum_length:
            raise ValueError(
                f"{mode_name} split length {len(self._data)} cannot form one window "
                f"requiring {minimum_length} points"
            )

    @staticmethod
    def _encode_timestamps(timestamps: np.ndarray) -> np.ndarray:
        index = pd.DatetimeIndex(timestamps)
        seconds = index.hour * 3600 + index.minute * 60 + index.second
        time_of_day = np.asarray(seconds / 86400.0, dtype=np.float32)
        day_of_week = np.asarray(index.dayofweek / 7.0, dtype=np.float32)
        return np.stack((time_of_day, day_of_week), axis=-1)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __len__(self) -> int:
        return len(self._data) - self.input_len - self.output_len + 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_start = index + self.input_len
        item: dict[str, Any] = {
            "inputs": self._data[index:target_start],
            "inputs_observed": self._observed[index:target_start],
            "targets": self._data[target_start : target_start + self.output_len],
            "targets_observed": self._observed[
                target_start : target_start + self.output_len
            ],
        }
        if self.use_timestamps:
            item["inputs_timestamps"] = self._timestamps[index:target_start]
            item["targets_timestamps"] = self._timestamps[
                target_start : target_start + self.output_len
            ]
        return item
