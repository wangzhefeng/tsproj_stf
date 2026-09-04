#!/usr/bin/env python3
"""运行单个时空预测实验。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tsproj_stf.experiments.config import load_experiment_config
from tsproj_stf.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="仅覆盖训练模型的 epoch，适合 smoke")
    parser.add_argument("--name-suffix", default="")
    run_policy = parser.add_mutually_exclusive_group()
    run_policy.add_argument("--resume", action="store_true")
    run_policy.add_argument("--force-new-run", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    model_params = dict(config.model_params)
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        model_params["epochs"] = args.epochs
    config = replace(
        config,
        name=f"{config.name}{args.name_suffix}",
        seed=config.seed if args.seed is None else args.seed,
        model_params=model_params,
    )
    result = run_experiment(
        config,
        resume=args.resume,
        force_new_run=args.force_new_run,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "prediction_shape": list(result.prediction_shape),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
