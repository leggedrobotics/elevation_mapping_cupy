"""The image callback must hand ElevationMap.input_image the map-to-camera transform."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

tf2_ros = pytest.importorskip("tf2_ros")
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from tf_transformations import quaternion_from_matrix


def _camera_pose():
    # Optical frame (z forward, x right, y down) looking along map +y, 0.5 m above (2, 1).
    R_map_cam = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    p_map_cam = np.array([2.0, 1.0, 0.5])
    return R_map_cam, p_map_cam


def _tf_buffer(R_map_cam, p_map_cam):
    T = np.eye(4)
    T[:3, :3] = R_map_cam
    q = quaternion_from_matrix(T)
    msg = TransformStamped()
    msg.header.frame_id = "map"
    msg.child_frame_id = "camera"
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = p_map_cam
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
    buffer = tf2_ros.Buffer()
    buffer.set_transform_static(msg, "test")
    return buffer


def test_image_callback_passes_map_to_camera_transform():
    from elevation_mapping_cupy.elevation_mapping_node import ElevationMappingNode

    R_map_cam, p_map_cam = _camera_pose()
    node = SimpleNamespace(
        map_frame="map",
        _tf_buffer=_tf_buffer(R_map_cam, p_map_cam),
        cv_bridge=SimpleNamespace(imgmsg_to_cv2=lambda *args, **kwargs: np.zeros((4, 6), dtype=np.float32)),
        resolve_image_channels=lambda key: ["semantic"],
        _map=SimpleNamespace(input_image=Mock()),
        _image_process_counter=0,
    )
    node.safe_lookup_transform = partial(ElevationMappingNode.safe_lookup_transform, node)
    camera = SimpleNamespace(header=SimpleNamespace(frame_id="camera", stamp=Time().to_msg()))
    camera_info = SimpleNamespace(
        k=[1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0], d=[0.0] * 5, distortion_model="plumb_bob", height=4, width=6
    )

    ElevationMappingNode.image_callback(node, camera, camera_info, "camera")

    R, t = node._map.input_image.call_args.args[2:4]
    # input_image projects map points with K @ (R @ p + t) and places the camera at -R.T @ t.
    point_ahead = p_map_cam + R_map_cam[:, 2]
    np.testing.assert_allclose(R @ point_ahead + t, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(-R.T @ t, p_map_cam, atol=1e-6)
