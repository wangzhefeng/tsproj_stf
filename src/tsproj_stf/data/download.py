"""带 checksum 门禁的数据下载。"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


class ChecksumMismatchError(ValueError):
    """下载内容与声明 checksum 不一致。"""


def _checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: str | Path,
    *,
    checksum: str,
    algorithm: str = "sha256",
) -> Path:
    """下载文件并仅在 checksum 通过后原子发布。"""

    target = Path(destination)
    expected = checksum.lower()
    if target.exists():
        actual = _checksum(target, algorithm)
        if actual != expected:
            raise ChecksumMismatchError(
                f"existing file checksum mismatch for {target}: expected {expected}, got {actual}"
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    digest = hashlib.new(algorithm)
    try:
        request = Request(url, headers={"User-Agent": "tsproj-stf/0.1"})
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ChecksumMismatchError(
                f"download checksum mismatch for {url}: expected {expected}, got {actual}"
            )
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return target
