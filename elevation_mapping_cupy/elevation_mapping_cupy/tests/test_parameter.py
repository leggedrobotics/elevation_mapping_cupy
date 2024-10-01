import pytest
from elevation_mapping_cupy.parameter import Parameter


def test_parameter():
    param = Parameter(
        use_chainer=False,
        weight_file="../../../config/weights.dat",
        plugin_config_file="../../../config/plugin_config.yaml",
    )
    res = param.resolution
    param.set_value("resolution", 0.1)
    param.get_types()
    param.get_names()
    param.update()
    assert param.resolution == param.get_value("resolution")
    param.load_weights(param.weight_file)
