"""可复现实验产物。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike


class RunConflictError(RuntimeError):
    """已有 run ID 对应另一份 resolved config。"""


def fingerprint_bytes(content: bytes) -> str:
    """返回内容的 SHA-256 fingerprint。"""

    return hashlib.sha256(content).hexdigest()


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class ArtifactStore:
    """单个不可覆盖 run 的产物目录。"""

    run_dir: Path

    @classmethod
    def initialize(
        cls,
        root: str | Path,
        run_id: str,
        resolved_config: dict[str, Any],
        *,
        resume: bool = False,
        force_new_run: bool = False,
    ) -> ArtifactStore:
        if resume and force_new_run:
            raise ValueError("resume and force_new_run are mutually exclusive")
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError(f"invalid run_id: {run_id!r}")
        root_path = Path(root)
        run_dir = root_path / run_id
        if force_new_run:
            sequence = 2
            while run_dir.exists():
                run_dir = root_path / f"{run_id}-run{sequence}"
                sequence += 1
        config_path = run_dir / "resolved_config.yaml"
        if run_dir.exists():
            existing = (
                yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if config_path.exists()
                else None
            )
            if existing != resolved_config:
                raise RunConflictError(
                    f"run {run_id!r} already exists with a different resolved config"
                )
            if resume:
                run_log = run_dir / "run.log"
                if run_log.exists():
                    try:
                        status = json.loads(run_log.read_text(encoding="utf-8")).get(
                            "status"
                        )
                    except (json.JSONDecodeError, AttributeError) as error:
                        raise RunConflictError(
                            f"run {run_id!r} has an invalid completion marker"
                        ) from error
                    if status == "completed":
                        raise RunConflictError(
                            f"completed run {run_id!r} is immutable; use --force-new-run"
                        )
                return cls(run_dir=run_dir)
            raise FileExistsError(
                f"run {run_id!r} already exists; use --resume or --force-new-run"
            )

        if resume:
            raise FileNotFoundError(f"cannot resume missing run {run_id!r}")

        run_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_text(
            config_path,
            yaml.safe_dump(resolved_config, sort_keys=True, allow_unicode=True),
        )
        return cls(run_dir=run_dir)

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        _atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return path

    def write_text(self, filename: str, content: str) -> Path:
        path = self.run_dir / filename
        _atomic_write_text(path, content)
        return path

    def write_predictions(
        self,
        prediction: ArrayLike,
        targets: ArrayLike,
        observed: ArrayLike,
    ) -> Path:
        path = self.run_dir / "predictions.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.run_dir,
            prefix=".predictions.",
            suffix=".npz",
        )
        os.close(descriptor)
        try:
            np.savez_compressed(
                temporary_name,
                prediction=np.asarray(prediction),
                targets=np.asarray(targets),
                observed=np.asarray(observed, dtype=bool),
            )
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path
