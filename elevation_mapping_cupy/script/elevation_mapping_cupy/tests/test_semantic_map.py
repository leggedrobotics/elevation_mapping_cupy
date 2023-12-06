import pytest
from elevation_mapping_cupy import semantic_map, parameter
import cupy as cp
import numpy as np


@pytest.fixture()
def semmap_ex(sem_lay, fusion_alg):
    p = parameter.Parameter(
        use_chainer=False,
        weight_file="../../../config/weights.dat",
        plugin_config_file="../../../config/plugin_config.yaml",
    )
    for subs, value in p.subscriber_cfg.items():
        value["channels"] = sem_lay
        value["fusion"] = fusion_alg
    p.update()
    e = semantic_map.SemanticMap(p)
    return e


@pytest.mark.parametrize(
    "sem_lay, fusion_alg,channels",
    [
        (["feat_0", "feat_1"], ["average", "average"], ["feat_0"]),
        (["feat_0", "feat_1"], ["average", "average"], []),
        (["feat_0", "feat_1", "rgb"], ["average", "average", "color"], ["rgb", "feat_0"],),
        (["feat_0", "feat_1", "rgb"], ["class_average", "average", "color"], ["rgb", "feat_0"],),
        (["feat_0", "feat_1", "rgb"], ["class_bayesian", "average", "color"], ["rgb", "feat_0"],),
        (["feat_0", "feat_1", "rgb"], ["class_bayesian", "average", "color"], ["rgb", "feat_0", "feat_1"],),
        (["feat_0", "feat_1", "rgb"], ["class_bayesian", "class_max", "color"], ["rgb", "feat_0", "feat_1"],),
        (["max1", "max2", "rgb"], ["class_max", "class_max", "color"], ["rgb", "max1", "max2"],),
    ],
)
def test_fusion_of_pcl(semmap_ex, channels):
    fusion = semmap_ex.get_fusion_of_pcl(channels=channels)
    assert len(fusion) <= len(channels)
    assert len(fusion) > 0 or len(channels) == 0
    assert all(isinstance(item, str) for item in fusion)


@pytest.mark.parametrize(
    "sem_lay, fusion_alg", [(["feat_0", "feat_1", "rgb"], ["average", "average", "color"]),],
)
@pytest.mark.parametrize("channels", [["rgb"], ["rgb", "feat_0"], []])
def test_indices_fusion(semmap_ex, channels, fusion_alg):
    pcl_indices, layer_indices = semmap_ex.get_indices_fusion(pcl_channels=channels, fusion_alg=fusion_alg[0])
