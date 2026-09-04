import numpy as np
import pytest

from tsproj_stf.data.scaling import ZScoreScaler


def test_fits_masked_statistics_along_time_axis() -> None:
    values = np.array([[1.0, 10.0], [3.0, 999.0], [5.0, 30.0]], dtype=np.float32)
    observed = np.array([[True, True], [True, False], [True, True]])

    scaler = ZScoreScaler.fit(values, observed)

    np.testing.assert_allclose(scaler.stats.mean, [[3.0, 20.0]])
    np.testing.assert_allclose(scaler.stats.std[0, 0], np.sqrt(8.0 / 3.0))
    np.testing.assert_allclose(scaler.stats.std[0, 1], 10.0)


def test_transform_and_inverse_transform_round_trip() -> None:
    train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    scaler = ZScoreScaler.fit(train, np.ones_like(train, dtype=bool))
    values = np.array([[7.0, 40.0]], dtype=np.float32)

    restored = scaler.inverse_transform(scaler.transform(values))

    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)


def test_constant_series_uses_unit_standard_deviation() -> None:
    values = np.ones((4, 2), dtype=np.float32)

    scaler = ZScoreScaler.fit(values, np.ones_like(values, dtype=bool))

    np.testing.assert_array_equal(scaler.stats.std, np.ones((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(scaler.transform(values), np.zeros_like(values))


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="observed shape"):
        ZScoreScaler.fit(np.ones((4, 2)), np.ones((4, 1), dtype=bool))


def test_rejects_fully_missing_series() -> None:
    values = np.ones((4, 2), dtype=np.float32)
    observed = np.array([[True, False]] * 4)

    with pytest.raises(ValueError, match="no observed training values"):
        ZScoreScaler.fit(values, observed)
