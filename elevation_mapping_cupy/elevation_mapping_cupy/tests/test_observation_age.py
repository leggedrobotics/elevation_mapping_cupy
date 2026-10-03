"""Raw source-time evidence stays distinct from visibility and authored maps."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import ElevationMap, Parameter
from elevation_mapping_cupy.elevation_mapping import GridGeometry


@pytest.fixture
def terrain():
    root = Path(__file__).resolve().parents[2]
    parameter = Parameter(
        use_chainer=False,
        weight_file=str(root / "config/core/weights.dat"),
        plugin_config_file=str(root / "config/core/plugin_config.yaml"),
    )
    parameter.resolution = 0.2
    parameter.map_length = 4.0
    parameter.enable_visibility_cleanup = False
    parameter.enable_drift_compensation = False
    parameter.update()
    return ElevationMap(parameter)


def _observe(terrain, stamp, points=((1.0, 0.0, 0.0),)):
    terrain.input_pointcloud(
        np.asarray(points, dtype=np.float32),
        ["x", "y", "z"],
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        0.0,
        0.0,
        source_stamp_ns=stamp,
    )


def _geometry(terrain):
    length = (terrain.cell_n - 2) * terrain.resolution
    return GridGeometry(length, length, terrain.resolution, np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0]))


def test_cloud_elapsed_time_survives_timer_stall_and_only_raw_endpoints_reset(terrain):
    _observe(terrain, 1_000_000_000)
    original = terrain.new_map[2] > 0.0
    assert int(cp.count_nonzero(original)) == 1
    assert float(terrain.observation_age[original][0]) == 0.0
    # Valid authored/interpolated cells still have no raw source evidence.
    terrain.elevation_map[2, 3, 3] = 1.0
    terrain.update_time()
    _observe(terrain, 8_500_000_000, ((1.0, 0.8, 0.0), (np.nan, 0.0, 0.0)))
    assert float(terrain.observation_age[original][0]) == pytest.approx(7.5)
    assert float(terrain.elevation_map[4][original][0]) == pytest.approx(0.1)
    assert float(terrain.observation_age[terrain.new_map[2] > 0.0][0]) == 0.0
    assert bool(cp.isnan(terrain.observation_age[3, 3]))
    for _ in range(5):
        terrain.update_time()
    assert float(terrain.observation_age[original][0]) == pytest.approx(7.5)


@pytest.mark.parametrize("stamp", [None, 0, -1, 2_000_000_000, 1_000_000_000])
def test_missing_zero_duplicate_or_backward_cloud_cannot_grant_evidence(terrain, stamp):
    _observe(terrain, 2_000_000_000)
    _observe(terrain, stamp)
    assert bool(cp.all(cp.isnan(terrain.observation_age)))
    assert terrain._observation_stamp_ns == 2_000_000_000
    _observe(terrain, 3_000_000_000)
    assert int(cp.count_nonzero(cp.isfinite(terrain.observation_age))) == 1


def test_shift_clear_restore_and_patch_invalidate_evidence(terrain):
    terrain.observation_age.fill(2.0)
    terrain.shift_map_xy(cp.asarray([1, -2]))
    assert bool(cp.all(cp.isnan(terrain.observation_age[:, 0])))
    assert bool(cp.all(cp.isnan(terrain.observation_age[-2:, :])))
    assert float(terrain.observation_age[2, 2]) == 2.0
    terrain.clear()
    assert bool(cp.all(cp.isnan(terrain.observation_age)))
    terrain.observation_age.fill(1.0)
    size = terrain.cell_n - 2
    terrain.set_full_map(
        {}, {"elevation": np.ones((size, size)), "observation_age": np.zeros((size, size))}, _geometry(terrain)
    )
    assert bool(cp.all(cp.isnan(terrain.observation_age)))
    terrain.observation_age.fill(1.0)
    mask = np.full((size, size), np.nan)
    mask[: size // 2] = 1.0
    terrain.apply_masked_replace({"elevation": np.ones((size, size))}, mask, _geometry(terrain))
    assert int(cp.count_nonzero(cp.isnan(terrain.observation_age))) == (size // 2) * size
    with pytest.raises(ValueError, match="read-only"):
        terrain.apply_masked_replace({"observation_age": np.zeros((size, size))}, None, _geometry(terrain))


def test_published_float32_age_is_conservative_and_get_layer_is_read_only(terrain):
    age = float(np.float32(0.1)) + 1e-10
    terrain.observation_age.fill(age)
    published = terrain.export_layers(["observation_age"])["observation_age"]
    assert published.dtype == np.float32 and np.all(published.astype(np.float64) >= age)
    terrain.get_layer("observation_age").fill(0.0)
    assert bool(cp.all(terrain.observation_age == age))
    assert "observation_age" in terrain.list_layers()


def test_cloud_timestamp_commits_only_after_fusion_and_images_do_not_retimestamp():
    from elevation_mapping_cupy.elevation_mapping_node import ElevationMappingNode
    from elevation_mapping_cupy.tests.test_pointcloud_parser import _padded_cloud

    old_stamp = SimpleNamespace(sec=1, nanosec=0)
    node = SimpleNamespace(
        _last_t=old_stamp,
        param=SimpleNamespace(subscriber_cfg={"lidar": {}}),
        map_frame="map",
        _map=SimpleNamespace(input_pointcloud=Mock()),
        _pointcloud_process_counter=0,
    )
    cloud = _padded_cloud()
    cloud.header.frame_id = "map"
    cloud.header.stamp.sec = 5
    ElevationMappingNode.pointcloud_callback(node, cloud, "lidar")
    assert node._map.input_pointcloud.call_args.kwargs["source_stamp_ns"] == 5_000_000_000
    assert node._last_t == cloud.header.stamp
    node._map.input_pointcloud.side_effect = RuntimeError("fusion failed")
    node._last_t = old_stamp
    with pytest.raises(RuntimeError, match="fusion failed"):
        ElevationMappingNode.pointcloud_callback(node, cloud, "lidar")
    assert node._last_t is old_stamp
    node.cv_bridge = SimpleNamespace(imgmsg_to_cv2=lambda *args, **kwargs: np.zeros((2, 2)))
    node.resolve_image_channels = lambda key: []
    camera = SimpleNamespace(header=SimpleNamespace(frame_id="map", stamp=SimpleNamespace(sec=100, nanosec=0)))
    camera_info = SimpleNamespace(k=np.eye(3).ravel(), d=np.zeros(5))
    ElevationMappingNode.image_callback(node, camera, camera_info, "camera")
    assert node._last_t is old_stamp
