import numpy as np
import pytest

from tsproj_stf.baselines.persistence import persistence_forecast


def test_repeats_last_value_across_all_horizons() -> None:
    inputs = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])

    prediction = persistence_forecast(
        inputs,
        np.ones_like(inputs, dtype=bool),
        output_len=4,
    )

    assert prediction.shape == (1, 4, 2)
    np.testing.assert_array_equal(prediction[0], [[3.0, 30.0]] * 4)


def test_uses_last_observed_value_for_each_series() -> None:
    inputs = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    observed = np.array([[[True, True], [True, False], [False, True]]])

    prediction = persistence_forecast(inputs, observed, output_len=2)

    np.testing.assert_array_equal(prediction[0], [[2.0, 30.0], [2.0, 30.0]])


def test_rejects_series_without_observed_history() -> None:
    inputs = np.ones((1, 3, 2))
    observed = np.array([[[True, False], [True, False], [True, False]]])

    with pytest.raises(ValueError, match="no observed input history"):
        persistence_forecast(inputs, observed, output_len=2)


def test_uses_explicit_fallback_for_series_without_observed_history() -> None:
    inputs = np.ones((1, 3, 2))
    observed = np.array([[[True, False], [True, False], [True, False]]])

    prediction = persistence_forecast(inputs, observed, output_len=2, fallback_value=0.0)

    np.testing.assert_array_equal(prediction[0], [[1.0, 0.0], [1.0, 0.0]])


def test_rejects_non_positive_output_length() -> None:
    with pytest.raises(ValueError, match="output_len must be positive"):
        persistence_forecast(np.ones((1, 2, 1)), np.ones((1, 2, 1), dtype=bool), 0)
