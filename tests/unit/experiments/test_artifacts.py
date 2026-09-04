import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tsproj_stf.experiments.artifacts import (
    ArtifactStore,
    RunConflictError,
    ensure_clean_training_state,
    fingerprint_bytes,
)


def config(seed: int = 42) -> dict[str, object]:
    return {"name": "run", "seed": seed, "horizons": [3, 6, 12]}


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    assert fingerprint_bytes(b"same") == fingerprint_bytes(b"same")
    assert fingerprint_bytes(b"same") != fingerprint_bytes(b"different")


def test_initializes_run_with_resolved_config(tmp_path: Path) -> None:
    store = ArtifactStore.initialize(tmp_path, "run-42", config())

    saved = yaml.safe_load((store.run_dir / "resolved_config.yaml").read_text())
    assert saved == config()


def test_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    ArtifactStore.initialize(tmp_path, "run-42", config())

    with pytest.raises(FileExistsError, match="--resume.*--force-new-run"):
        ArtifactStore.initialize(tmp_path, "run-42", config())


def test_resumes_only_an_existing_run_with_the_same_config(tmp_path: Path) -> None:
    original = ArtifactStore.initialize(tmp_path, "run-42", config())

    resumed = ArtifactStore.initialize(tmp_path, "run-42", config(), resume=True)

    assert resumed.run_dir == original.run_dir
    with pytest.raises(RunConflictError, match="different resolved config"):
        ArtifactStore.initialize(tmp_path, "run-42", config(seed=43), resume=True)


def test_refuses_to_resume_a_completed_run(tmp_path: Path) -> None:
    store = ArtifactStore.initialize(tmp_path, "run-42", config())
    store.write_text("run.log", '{"status":"completed"}\n')

    with pytest.raises(RunConflictError, match="completed run"):
        ArtifactStore.initialize(tmp_path, "run-42", config(), resume=True)


def test_force_new_run_allocates_a_new_immutable_directory(tmp_path: Path) -> None:
    original = ArtifactStore.initialize(tmp_path, "run-42", config())

    second = ArtifactStore.initialize(tmp_path, "run-42", config(), force_new_run=True)
    third = ArtifactStore.initialize(tmp_path, "run-42", config(), force_new_run=True)

    assert original.run_dir.name == "run-42"
    assert second.run_dir.name == "run-42-run2"
    assert third.run_dir.name == "run-42-run3"


def test_resume_and_force_new_run_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ArtifactStore.initialize(
            tmp_path,
            "run-42",
            config(),
            resume=True,
            force_new_run=True,
        )


def test_reports_different_config_as_conflict(tmp_path: Path) -> None:
    ArtifactStore.initialize(tmp_path, "run-42", config())

    with pytest.raises(RunConflictError, match="different resolved config"):
        ArtifactStore.initialize(tmp_path, "run-42", config(seed=43))


def test_ensure_clean_training_state_rejects_stale_checkpoints(tmp_path: Path) -> None:
    store = ArtifactStore.initialize(tmp_path, "run-42", config())

    ensure_clean_training_state(store.run_dir)

    stale = store.run_dir / "checkpoint" / "abc123" / "STID_089.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"fake")
    with pytest.raises(RunConflictError, match="--force-new-run"):
        ensure_clean_training_state(store.run_dir)


def test_different_seed_run_ids_do_not_conflict(tmp_path: Path) -> None:
    first = ArtifactStore.initialize(tmp_path, "run-seed42", config(seed=42))
    second = ArtifactStore.initialize(tmp_path, "run-seed43", config(seed=43))

    assert first.run_dir != second.run_dir


def test_writes_json_and_prediction_arrays(tmp_path: Path) -> None:
    store = ArtifactStore.initialize(tmp_path, "run-42", config())
    store.write_json("metrics.json", {"overall": {"MAE": 1.25}})
    store.write_predictions(
        prediction=np.ones((1, 2, 3), dtype=np.float32),
        targets=np.zeros((1, 2, 3), dtype=np.float32),
        observed=np.ones((1, 2, 3), dtype=bool),
    )

    assert json.loads((store.run_dir / "metrics.json").read_text())["overall"]["MAE"] == 1.25
    with np.load(store.run_dir / "predictions.npz") as arrays:
        assert set(arrays.files) == {"prediction", "targets", "observed"}
        assert arrays["prediction"].shape == (1, 2, 3)
