from pathlib import Path

import pytest

from tsproj_stf.data.download import ChecksumMismatchError, download_file
from tsproj_stf.experiments.artifacts import fingerprint_bytes


def test_downloads_file_and_verifies_sha256(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"metr-la")
    destination = tmp_path / "downloads" / "target.bin"

    result = download_file(
        source.as_uri(),
        destination,
        checksum=fingerprint_bytes(b"metr-la"),
        algorithm="sha256",
    )

    assert result == destination
    assert destination.read_bytes() == b"metr-la"


def test_reuses_existing_file_only_after_checksum_verification(tmp_path: Path) -> None:
    destination = tmp_path / "target.bin"
    destination.write_bytes(b"valid")

    result = download_file(
        "https://invalid.example/not-requested",
        destination,
        checksum=fingerprint_bytes(b"valid"),
        algorithm="sha256",
    )

    assert result == destination


def test_checksum_mismatch_does_not_publish_download(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupt")
    destination = tmp_path / "target.bin"

    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        download_file(
            source.as_uri(),
            destination,
            checksum=fingerprint_bytes(b"expected"),
            algorithm="sha256",
        )

    assert not destination.exists()
