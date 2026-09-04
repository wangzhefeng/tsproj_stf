import torch


def test_project_package_is_importable() -> None:
    import tsproj_stf

    assert tsproj_stf.__version__ == "0.1.0"


def test_basicts_stid_forward_shape_on_cpu() -> None:
    from basicts.models.STID import STID, STIDConfig

    config = STIDConfig(
        input_len=12,
        output_len=12,
        num_features=4,
        if_spatial=True,
        if_time_in_day=True,
        if_day_in_week=True,
        num_time_in_day=288,
    )
    model = STID(config).cpu()
    inputs = torch.randn(2, 12, 4)
    timestamps = torch.zeros(2, 12, 2)

    prediction = model(inputs=inputs, inputs_timestamps=timestamps)

    assert prediction.shape == (2, 12, 4)
    assert torch.isfinite(prediction).all()
