"""BasicTS Graph WaveNet 训练后端。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from basicts.configs import BasicTSForecastingConfig
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
from tsproj_stf.models.graph_wavenet.config import GraphWaveNetConfig
from tsproj_stf.models.graph_wavenet.model import GraphWaveNet


def _num_nodes(data_path: Path) -> int:
    values = np.load(data_path / "values.npy", mmap_mode="r")
    return int(values.shape[1])


def build_graph_wavenet_config(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> BasicTSForecastingConfig:
    """把项目实验配置翻译为 BasicTS Graph WaveNet 配置。"""

    if not config.rescale:
        raise ValueError("Graph WaveNet experiments require rescale=true")
    params = config.model_params
    data_path = Path(config.data_path)
    target_feature = str(params.get("target_feature", "speed"))
    null_value = float(params.get("null_value", 0.0))
    num_epochs = int(params.get("epochs", 100))
    model_config = GraphWaveNetConfig(
        input_len=config.input_len,
        output_len=config.output_len,
        num_nodes=_num_nodes(data_path),
        graph_mode=str(params["graph_mode"]),
        fixed_graph_path=str(data_path / "graphs.npz"),
        graph_name=str(params.get("graph_name", "physical")),
        residual_channels=int(params.get("residual_channels", 32)),
        dilation_channels=int(params.get("dilation_channels", 32)),
        skip_channels=int(params.get("skip_channels", 256)),
        end_channels=int(params.get("end_channels", 512)),
        kernel_size=int(params.get("kernel_size", 2)),
        dilations=tuple(int(value) for value in params.get("dilations", [1, 2, 4, 8])),
        diffusion_order=int(params.get("diffusion_order", 2)),
        adaptive_embedding_dim=int(params.get("adaptive_embedding_dim", 10)),
        dropout=float(params.get("dropout", 0.3)),
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
        model=GraphWaveNet,
        model_config=model_config,
        dataset_name=config.dataset,
        dataset_type=ProjectForecastingDataset,
        dataset_params=dataset_params,
        scaler=ProjectBasicTSScaler,
        data_file_path=str(data_path),
        split_ratios=config.split_ratios,
        target_feature=target_feature,
        gpus=None,
        num_epochs=num_epochs,
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
            EarlyStopping(patience=int(params.get("patience", 15))),
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
        test_interval=int(params.get("test_interval", num_epochs + 1)),
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


def run_graph_wavenet_backend(
    config: ExperimentConfig,
    store: ArtifactStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练最佳 Graph WaveNet checkpoint 并返回原量纲测试数组。"""

    basic_config = build_graph_wavenet_config(config, store)
    basic_config.save()
    set_device_type("cpu")
    runner = BasicTSRunner(basic_config)
    runner.init_logger(logger_name="tsproj-stf-GraphWaveNet", log_file_name="run")
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
    shape = (len(test_dataset), config.output_len, test_dataset.data.shape[1])
    result_dir = Path(runner.ckpt_save_dir) / "test_results"
    prediction = np.memmap(
        result_dir / "prediction.npy",
        dtype=np.float32,
        mode="r",
        shape=shape,
    ).copy()
    targets = np.memmap(
        result_dir / "targets.npy",
        dtype=np.float32,
        mode="r",
        shape=shape,
    ).copy()
    observed = np.stack(
        [test_dataset[index]["targets_observed"] for index in range(len(test_dataset))]
    ).astype(bool)
    return prediction, targets, observed
