import numpy as np
import pytest

from tsproj_stf.data.validation import validate_regular_timestamps


def test_accepts_strictly_increasing_regular_timestamps() -> None:
    timestamps = np.array(
        ["2026-01-01T00:00", "2026-01-01T00:05", "2026-01-01T00:10"],
        dtype="datetime64[m]",
    )

    frequency = validate_regular_timestamps(timestamps)

    assert frequency == np.timedelta64(5, "m")


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([1, 1, 2]),
        np.array([2, 1, 3]),
        np.array([0, 1, 3]),
    ],
)
def test_rejects_non_monotonic_or_irregular_timestamps(timestamps: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strictly increasing and regularly spaced"):
        validate_regular_timestamps(timestamps)


def test_rejects_too_short_timestamp_axis() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validate_regular_timestamps(np.array([1]))
