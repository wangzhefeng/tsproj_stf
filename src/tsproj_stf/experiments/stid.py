"""BasicTS STID 训练后端。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STID import STID, STIDConfig
from basicts.runners import BasicTSRunner
from basicts.runners.callback import EarlyStopping, GradientClipping
from easytorch.device import set_device_type

from tsproj_stf.data.adapters import (
    ProjectBasicTSScaler,
    ProjectForecastingDataset,
    ProjectForecastingTaskFlow,
)
from tsproj_stf.experiments.artifacts import ArtifactStore
from tsproj_stf.experiments.config import ExperimentConfig


def _num_nodes(data_path: Path) -> int:
    metadata = json.loads((data_path / "metadata.json").read_text(encoding="utf-8"))
    return len(metadata["node_ids"])


def build_stid_config(config: ExperimentConfig, store: ArtifactStore) -> BasicTSForecastingConfig:
    """把项目配置转换为固定语义的 BasicTS 配置。"""

    if not config.rescale:
        raise ValueError("STID experiments require rescale=true for original-scale metrics")
    params = config.model_params
    target_feature = str(params.get("target_feature", "speed"))
    null_value = float(params.get("null_value", 0.0))
    data_path = Path(config.data_path)

    model_config = STIDConfig(
        input_len=config.input_len,
        output_len=config.output_len,
        num_features=_num_nodes(data_path),
        input_hidden_size=int(params.get("input_hidden_size", 32)),
        intermediate_size=params.get("intermediate_size"),
        hidden_act=str(params.get("hidden_act", "relu")),
        num_layers=int(params.get("num_layers", 1)),
        if_spatial=True,
        spatial_hidden_size=int(params.get("spatial_hidden_size", 32)),
        if_time_in_day=True,
        if_day_in_week=True,
        num_time_in_day=int(params.get("num_time_in_day", 288)),
        num_day_in_week=7,
        tid_hidden_size=int(params.get("tid_hidden_size", 32)),
        diw_hidden_size=int(params.get("diw_hidden_size", 32)),
    )
    dataset_params = {
        "data_file_path": str(data_path),
        "dataset_name": config.dataset,
        "input_len": config.input_len,
        "output_len": config.output_len,
        "split_ratios": config.split_ratios,
        "target_feature": target_feature,
        "null_value": null_value,
        "use_timestamps": True,
        "memmap": False,
    }
    return BasicTSForecastingConfig(
        model=STID,
        model_config=model_config,
        dataset_name=config.dataset,
        dataset_type=ProjectForecastingDataset,
        dataset_params=dataset_params,
        scaler=ProjectBasicTSScaler,
        data_file_path=str(data_path),
        split_ratios=config.split_ratios,
        target_feature=target_feature,
        gpus=None,
        num_epochs=int(params.get("epochs", 100)),
        num_steps=None,
        batch_size=int(params.get("batch_size", 64)),
        input_len=config.input_len,
        output_len=config.output_len,
        use_timestamps=True,
        loss="MAE",
        metrics=["MAE", "RMSE", "MAPE", "WAPE"],
        target_metric="MAE",
        best_metric="min",
        callbacks=[
            EarlyStopping(patience=int(params.get("patience", 10))),
            GradientClipping(max_norm=float(params.get("max_grad_norm", 5.0))),
        ],
        optimizer_params={
            "lr": float(params.get("learning_rate", 0.001)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
        },
        lr_scheduler=torch.optim.lr_scheduler.MultiStepLR,
        lr_scheduler_params={
            "milestones": list(params.get("lr_milestones", [1, 50, 80])),
            "gamma": float(params.get("lr_gamma", 0.5)),
        },
        seed=config.seed,
        null_val=null_value,
        null_to_num=0.0,
        norm_each_channel=True,
        rescale=True,
        taskflow=ProjectForecastingTaskFlow(),
        eval_horizons=list(config.horizons),
        eval_after_train=True,
        save_results=True,
        deterministic=True,
        cudnn_benchmark=False,
        cudnn_determinstic=True,
        ckpt_save_dir=str(store.run_dir / "checkpoint"),
        train_data_num_workers=0,
        val_data_num_workers=0,
        test_data_num_workers=0,
    )


def run_stid_backend(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练最佳 STID checkpoint，并返回原量纲测试预测、目标和项目 mask。"""

    basic_config = build_stid_config(config, store)
    basic_config.save()
    set_device_type("cpu")
    runner = BasicTSRunner(basic_config)
    runner.init_logger(logger_name="tsproj-stf-STID", log_file_name="run")
    runner.train()

    for attribute in ("_prediction_memmap", "_targets_memmap", "_inputs_memmap"):
        memmap = getattr(runner, attribute, None)
        if memmap is not None:
            memmap.flush()

    test_dataset = ProjectForecastingDataset(
        data_file_path=config.data_path,
        dataset_name=config.dataset,
        input_len=config.input_len,
        output_len=config.output_len,
        mode="test",
        split_ratios=config.split_ratios,
        target_feature=str(config.model_params.get("target_feature", "speed")),
        null_value=float(config.model_params.get("null_value", 0.0)),
        use_timestamps=True,
    )
    sample_count = len(test_dataset)
    num_nodes = test_dataset.data.shape[1]
    result_dir = Path(runner.ckpt_save_dir) / "test_results"
    shape = (sample_count, config.output_len, num_nodes)
    prediction = np.memmap(
        result_dir / "prediction.npy", dtype=np.float32, mode="r", shape=shape
    ).copy()
    targets = np.memmap(
        result_dir / "targets.npy", dtype=np.float32, mode="r", shape=shape
    ).copy()
    observed = np.stack(
        [test_dataset[index]["targets_observed"] for index in range(sample_count)]
    ).astype(bool)
    return prediction, targets, observed
