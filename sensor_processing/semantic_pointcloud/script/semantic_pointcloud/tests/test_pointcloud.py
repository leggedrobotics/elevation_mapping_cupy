import pytest
from ..pointcloud_node import PointcloudNode
import cupy as cp
from ..utils import encode_max

"""This test file only works if ros is installed
"""


@pytest.fixture()
def pointcloud_ex(cam_name):
    node = PointcloudNode(cam_name)
    node.initialize_semantics()
    return node


@pytest.mark.parametrize(
    "cam_name",
    [
        "front_cam",
    ],
)
def test_initialize(pointcloud_ex):
    # todo here we can add more test
    pointcloud_ex.initialize_semantics()
    pointcloud_ex.register_sub_pub()


@pytest.mark.parametrize(
    "cam_name",
    [
        ("front_cam"),
    ],
)
def test_pub_seg(pointcloud_ex):
    # TODO we have to encode the max class other wise it is going to make problem
    if pointcloud_ex.param.semantic_segmentation:
        seg_chan = len(pointcloud_ex.segmentation_channels.keys())
        amount = seg_chan + pointcloud_ex.param.fusion.count("max_class")
        if "class_max" in pointcloud_ex.param.fusion:
            val = cp.random.rand(amount, 360, 640, dtype=cp.float32).astype(cp.float16)
            ind = cp.random.randint(0, 2, (amount, 360, 640), dtype=cp.uint32).astype(
                cp.float32
            )
            prob = encode_max(val, ind)
        else:
            prob = cp.random.rand(amount, 360, 640)
        pointcloud_ex.publish_segmentation_image(prob)
