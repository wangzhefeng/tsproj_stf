"""滑窗索引。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowIndex:
    """一个 one-shot 输入与目标窗口。"""

    input_slice: slice
    target_slice: slice


def make_window_indices(
    bounds: slice,
    input_len: int,
    output_len: int,
) -> tuple[WindowIndex, ...]:
    """在半开区间内生成不跨边界的连续滑窗。"""

    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if bounds.start is None or bounds.stop is None:
        raise ValueError("bounds must have explicit start and stop")
    if bounds.start < 0 or bounds.stop <= bounds.start:
        raise ValueError(f"invalid bounds: {bounds}")

    available = bounds.stop - bounds.start
    required = input_len + output_len
    if available < required:
        raise ValueError(
            f"split length {available} cannot form one window requiring {required} points"
        )

    return tuple(
        WindowIndex(
            input_slice=slice(start, start + input_len),
            target_slice=slice(start + input_len, start + required),
        )
        for start in range(bounds.start, bounds.stop - required + 1)
    )
