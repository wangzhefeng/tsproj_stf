import pytest

from tsproj_stf.data.split import SplitBounds, chronological_split


def test_chronological_split_is_contiguous_and_complete() -> None:
    bounds = chronological_split(10, (0.7, 0.1, 0.2))

    assert bounds == SplitBounds(slice(0, 7), slice(7, 8), slice(8, 10))
    assert bounds.as_dict() == {
        "train": [0, 7],
        "val": [7, 8],
        "test": [8, 10],
    }


@pytest.mark.parametrize(
    ("length", "ratios", "message"),
    [
        (0, (0.7, 0.1, 0.2), "length must be positive"),
        (10, (0.7, 0.3, 0.1), "sum to 1"),
        (10, (0.8, 0.2, 0.0), "must be positive"),
        (2, (0.7, 0.1, 0.2), "empty split"),
    ],
)
def test_rejects_invalid_split(
    length: int,
    ratios: tuple[float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chronological_split(length, ratios)
