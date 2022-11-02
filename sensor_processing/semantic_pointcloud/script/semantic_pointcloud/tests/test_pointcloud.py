import pytest
from ..pointcloud_node import PointcloudNode


@pytest.fixture()
def pointcloud_ex(cam_name):
    node = PointcloudNode(cam_name)
    return node


@pytest.mark.parametrize(
    "cam_name",
    [
        "front_cam",
    ],
)
def test_initialize_semantics(pointcloud_ex):
    pointcloud_ex.initialize_semantics()
