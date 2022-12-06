import pytest
from ..pointcloud_node import PointcloudNode
import cupy as cp


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


@pytest.mark.parametrize(
    "cam_name",
    [
        "front_cam",
    ],
)
def test_pub_seg(pointcloud_ex):
    # TODO we have to encode the max class other wise it is going to make problem
    amount = len(
        pointcloud_ex.segmentation_channels.keys()
    ) + pointcloud_ex.param.fusion.count("max_class")+1
    prob = cp.random.rand(amount, 360, 640)
    pointcloud_ex.publish_segmentation_image(prob)
