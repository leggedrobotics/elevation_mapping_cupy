import pytest
from elevation_mapping_cupy import parameter, elevation_mapping
import cupy as cp
import numpy as np


@pytest.fixture()
def elmap_ex(add_lay, fusion_alg):
    additional_layer = add_lay
    fusion_algorithms = fusion_alg
    p = parameter.Parameter(
        use_chainer=False,
        weight_file="../../../config/weights.dat",
        plugin_config_file="../../../config/plugin_config.yaml",
    )
    p.additional_layers = additional_layer
    p.fusion_algorithms = fusion_algorithms
    e = elevation_mapping.ElevationMap(p)
    return e


@pytest.mark.parametrize(
    "add_lay,fusion_alg",
    [
        (["feat_0", "feat_1", "rgb"], ["average", "average", "color"]),
        (["feat_0", "feat_1"], ["average", "average"]),
    ],
)
class TestElevationMap:
    def test_elmap_init(self, elmap_ex):
        assert len(elmap_ex.layer_names) == elmap_ex.elevation_map.shape[0]
        assert elmap_ex.color_map is None

    def test_elmap_input(self, elmap_ex):
        channels = ["x", "y", "z"] + elmap_ex.param.additional_layers
        points = cp.random.rand(100000, len(channels))
        R = cp.random.rand(3, 3)
        t = cp.random.rand(3)
        elmap_ex.input(points, channels, R, t, 0, 0)

    def test_elmap_update_normal(self, elmap_ex):
        elmap_ex.update_normal(elmap_ex.elevation_map[0])

    def test_elmap_move_to(self, elmap_ex):
        for i in range(20):
            pos = np.array([i * 0.01, i * 0.02, i * 0.01])
            R = cp.random.rand(3, 3)
            elmap_ex.move_to(pos, R)

    def test_get_map(self, elmap_ex):
        layers = [
            "elevation",
            "variance",
            "traversability",
            "min_filter",
            "smooth",
            "inpaint",
            "rgb",
        ]
        data = np.zeros((elmap_ex.cell_n - 2, elmap_ex.cell_n - 2), dtype=cp.float32)
        for layer in layers:
            elmap_ex.get_map_with_name_ref(layer, data)

# @pytest.mark.parametrize(
#     "add_lay, fusion_alg,channels",
#     [(["feat_0", "feat_1"], ["average", "average"],["feat_0"]),
#     (["feat_0", "feat_1"], ["average", "average"],[]),
#     (["feat_0", "feat_1", "rgb"], ["average", "average", "color"],["rgb", "feat_0"]),
#      ],
# )
# def test_fusion_of_pcl(elmap_ex, channels):
#     fusion = elmap_ex.get_fusion_of_pcl(channels=channels)
#     assert len(fusion) <= len(channels)
#     assert len(fusion)>0 or len(channels)==0
#     assert all(isinstance(item, str) for item in fusion)
#
#
# @pytest.mark.parametrize(
#     "add_lay, fusion_alg",
#     [(["feat_0", "feat_1", "rgb"], ["average", "average", "color"]),
#      ],
# )
# @pytest.mark.parametrize("channels", [["rgb"], ["rgb", "feat_0"], []])
# def test_indices_fusion(elmap_ex, channels,fusion_alg):
#     pcl_indices, layer_indices = elmap_ex.get_indices_fusion(pcl_channels=channels,fusion_alg=fusion_alg[0])
