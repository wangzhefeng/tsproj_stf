#!/usr/bin/env python3
"""准备项目支持的时空数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from tsproj_stf.data.datasets import (
    load_metr_la_raw,
    prepare_power_grid,
    save_metr_la_processed,
)
from tsproj_stf.data.download import download_file


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset config must be a mapping: {path}")
    return payload


def prepare_metr_la(config: dict[str, Any]) -> dict[str, Any]:
    """按配置下载、校验并转换 METR-LA。"""

    if config.get("dataset") != "METR-LA":
        raise ValueError(f"unsupported dataset: {config.get('dataset')}")
    raw_dir = Path(config["raw_dir"])
    sources = config["sources"]
    values_spec = sources["values"]
    graph_spec = sources["graph"]
    values_path = download_file(
        values_spec["url"],
        raw_dir / values_spec["filename"],
        checksum=values_spec["checksum"],
        algorithm=values_spec.get("algorithm", "sha256"),
    )
    graph_path = download_file(
        graph_spec["url"],
        raw_dir / graph_spec["filename"],
        checksum=graph_spec["checksum"],
        algorithm=graph_spec.get("algorithm", "sha256"),
    )
    data = load_metr_la_raw(
        values_path,
        graph_path,
        null_value=float(config.get("null_value", 0.0)),
    )
    return save_metr_la_processed(
        data,
        config["output_dir"],
        values_path,
        graph_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    if config.get("dataset") == "METR-LA":
        manifest = prepare_metr_la(config)
    elif config.get("dataset") == "POWER-GRID":
        manifest = prepare_power_grid(config)
    else:
        raise ValueError(f"unsupported dataset: {config.get('dataset')}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
