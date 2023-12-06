import pytest
from ..pointcloud_node import PointcloudNode
import cupy as cp
from ..utils import encode_max

"""This test file only works if ros is installed and the ros master is running.
"""


@pytest.fixture()
def pointcloud_ex(cam_name, channels, fusion, semseg, segpub, showlbl):
    node = PointcloudNode(cam_name)
    node.param.channels = channels
    node.param.fusion = fusion
    node.param.semantic_segmentation = semseg
    node.param.show_label_legend = showlbl
    node.param.publish_segmentation_image = segpub
    node.__init__(cam_name)
    return node


@pytest.mark.parametrize(
    "cam_name,channels, fusion,semseg",
    [
        ("front_cam", ["feat_0", "feat_1"], ["average", "average"], False),
        ("front_cam", ["feat_0", "feat_1"], ["class_max", "average"], True),
        ("front_cam", ["feat_0", "feat_1"], ["class_bayesian", "average"], True),
    ],
)
@pytest.mark.parametrize(
    "segpub", [True, False],
)
@pytest.mark.parametrize(
    "showlbl", [True, False],
)
def test_initialize(pointcloud_ex):
    # todo here we can add more test
    # params: semseg, channels, class max, publish seg, showlabellegend
    pointcloud_ex.initialize_semantics()
    pointcloud_ex.register_sub_pub()


# if semseg then segpub and showlbl might be displayed
@pytest.mark.parametrize(
    "cam_name,channels, fusion,semseg,segpub,showlbl",
    [
        ("front_cam", ["feat_0", "feat_1"], ["average", "average"], False, False, False,),
        ("front_cam", ["feat_0", "feat_1"], ["class_max", "average"], True, True, True),
        ("front_cam", ["feat_0", "feat_1"], ["class_max", "average"], True, False, False,),
        ("front_cam", ["feat_0", "feat_1"], ["class_bayesian", "average"], True, True, True,),
        ("front_cam", ["feat_0", "feat_1"], ["class_bayesian", "average"], True, False, False,),
    ],
)
def test_pcl_creation(pointcloud_ex, channels, fusion, semseg, segpub, showlbl):
    amount = 3
    if "class_max" in pointcloud_ex.param.fusion:
        val = cp.random.rand(360, 640, amount, dtype=cp.float32).astype(cp.float16)
        ind = cp.random.randint(0, 2, (360, 640, amount), dtype=cp.uint32).astype(cp.float32)
        img = encode_max(val, ind)
    else:
        img = (cp.random.rand(360, 640, amount) * 255).astype(cp.int32)
    pointcloud_ex.P = cp.random.rand(3, 4)
    depth = cp.random.rand(360, 640) * 8
    pointcloud_ex.create_pcl_from_image(img, depth, None)
