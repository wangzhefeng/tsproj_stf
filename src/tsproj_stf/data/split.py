"""连续时间切分。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class SplitBounds:
    """Train、validation、test 的半开区间。"""

    train: slice
    val: slice
    test: slice

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": [self.train.start, self.train.stop],
            "val": [self.val.start, self.val.stop],
            "test": [self.test.start, self.test.stop],
        }


def chronological_split(
    length: int,
    ratios: tuple[float, float, float],
) -> SplitBounds:
    """按时间顺序切分完整序列。"""

    if length <= 0:
        raise ValueError("length must be positive")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("split ratios must be positive")
    if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios)}")

    train_stop = int(length * ratios[0])
    val_stop = train_stop + int(length * ratios[1])
    bounds = SplitBounds(
        train=slice(0, train_stop),
        val=slice(train_stop, val_stop),
        test=slice(val_stop, length),
    )
    if any(part.stop <= part.start for part in (bounds.train, bounds.val, bounds.test)):
        raise ValueError(f"empty split produced for length={length}, ratios={ratios}")
    return bounds
