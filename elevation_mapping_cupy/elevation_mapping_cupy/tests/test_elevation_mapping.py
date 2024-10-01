import pytest
from elevation_mapping_cupy import parameter, elevation_mapping
import cupy as cp
import numpy as np


def encode_max(maxim, index):
    maxim, index = cp.asarray(maxim, dtype=cp.float32), cp.asarray(index, dtype=cp.uint32)
    # fuse them
    maxim = maxim.astype(cp.float16)
    maxim = maxim.view(cp.uint16)
    maxim = maxim.astype(cp.uint32)
    index = index.astype(cp.uint32)
    mer = cp.array(cp.left_shift(index, 16) | maxim, dtype=cp.uint32)
    mer = mer.view(cp.float32)
    return mer


@pytest.fixture()
def elmap_ex(add_lay, fusion_alg):
    additional_layer = add_lay
    fusion_algorithms = fusion_alg
    p = parameter.Parameter(
        use_chainer=False,
        weight_file="../../../config/weights.dat",
        plugin_config_file="../../../config/plugin_config.yaml",
    )
    p.subscriber_cfg["front_cam"]["channels"] = additional_layer
    p.subscriber_cfg["front_cam"]["fusion"] = fusion_algorithms
    p.update()
    e = elevation_mapping.ElevationMap(p)
    return e


@pytest.mark.parametrize(
    "add_lay,fusion_alg",
    [
        (["feat_0", "feat_1", "rgb"], ["average", "average", "color"]),
        (["feat_0", "feat_1"], ["average", "average"]),
        (["feat_0", "feat_1"], ["class_average", "class_average"]),
        (["feat_0", "feat_1"], ["class_bayesian", "class_bayesian"]),
        (["feat_0", "feat_1"], ["class_bayesian", "class_max"]),
        (["feat_0", "feat_1"], ["bayesian_inference", "bayesian_inference"]),
    ],
)
class TestElevationMap:
    def test_init(self, elmap_ex):
        assert len(elmap_ex.layer_names) == elmap_ex.elevation_map.shape[0]
        # assert elmap_ex.color_map is None

    def test_input(self, elmap_ex):
        channels = ["x", "y", "z"] + elmap_ex.param.additional_layers
        if "class_max" in elmap_ex.param.fusion_algorithms:
            val = cp.random.rand(100000, len(channels), dtype=cp.float32).astype(cp.float16)
            ind = cp.random.randint(0, 2, (100000, len(channels)), dtype=cp.uint32).astype(cp.float32)
            points = encode_max(val, ind)
        else:
            points = cp.random.rand(100000, len(channels), dtype=elmap_ex.param.data_type)
        R = cp.random.rand(3, 3, dtype=elmap_ex.param.data_type)
        t = cp.random.rand(3, dtype=elmap_ex.param.data_type)
        elmap_ex.input_pointcloud(points, channels, R, t, 0, 0)

    def test_update_normal(self, elmap_ex):
        elmap_ex.update_normal(elmap_ex.elevation_map[0])

    def test_move_to(self, elmap_ex):
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

    def test_get_position(self, elmap_ex):
        pos = np.random.rand(1, 3)
        elmap_ex.get_position(pos)

    def test_clear(self, elmap_ex):
        elmap_ex.clear()

    def test_move(self, elmap_ex):
        delta_position = np.random.rand(3)
        elmap_ex.move(delta_position)

    def test_exists_layer(self, elmap_ex, add_lay):
        for layer in add_lay:
            assert elmap_ex.exists_layer(layer)

    def test_polygon_traversability(self, elmap_ex):
        polygon = cp.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
        result = np.array([0, 0, 0])
        number_polygons = elmap_ex.get_polygon_traversability(polygon, result)
        untraversable_polygon = np.zeros((number_polygons, 2))
        elmap_ex.get_untraversable_polygon(untraversable_polygon)

    def test_initialize_map(self, elmap_ex):
        methods = ["linear", "cubic", "nearest"]
        for method in methods:
            points = np.array([[-4.0, 0.0, 0.0], [-4.0, 8.0, 1.0], [4.0, 8.0, 0.0], [4.0, 0.0, 0.0]])
            elmap_ex.initialize_map(points, method)

    def test_plugins(self, elmap_ex):
        layers = elmap_ex.plugin_manager.layer_names
        data = np.zeros((200, 200), dtype=np.float32)
        for layer in layers:
            elmap_ex.get_map_with_name_ref(layer, data)
