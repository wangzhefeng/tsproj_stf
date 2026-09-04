import pytest

from tsproj_stf.data.windows import WindowIndex, make_window_indices


def test_builds_all_windows_inside_bounds() -> None:
    windows = make_window_indices(slice(0, 30), input_len=12, output_len=12)

    assert len(windows) == 7
    assert windows[0] == WindowIndex(slice(0, 12), slice(12, 24))
    assert windows[-1] == WindowIndex(slice(6, 18), slice(18, 30))


def test_offset_split_never_crosses_its_boundaries() -> None:
    windows = make_window_indices(slice(30, 60), input_len=12, output_len=12)

    assert windows[0] == WindowIndex(slice(30, 42), slice(42, 54))
    assert windows[-1].target_slice.stop == 60
    assert all(window.input_slice.start >= 30 for window in windows)


def test_rejects_split_that_cannot_form_a_window() -> None:
    with pytest.raises(ValueError, match="cannot form one window"):
        make_window_indices(slice(0, 20), input_len=12, output_len=12)


@pytest.mark.parametrize(("input_len", "output_len"), [(0, 12), (12, 0), (-1, 12)])
def test_rejects_non_positive_window_lengths(input_len: int, output_len: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_window_indices(slice(0, 30), input_len=input_len, output_len=output_len)
