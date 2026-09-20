#!/usr/bin/env python3
import math
import message_filters
import numpy as np
import os
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from elevation_map_msgs.msg import ChannelInfo
from elevation_map_msgs.srv import Initialize
import ros2_numpy as rnp
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf_transformations import quaternion_matrix
import tf2_ros
import tf2_py as tf2
from rclpy.duration import Duration
from rclpy.serialization import serialize_message, deserialize_message
from grid_map_msgs.msg import GridMap
from grid_map_msgs.srv import SetGridMap, ProcessFile
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from std_srvs.srv import Trigger
import rosbag2_py
from elevation_mapping_cupy import ElevationMap, Parameter
from elevation_mapping_cupy.elevation_mapping import GridGeometry
from elevation_mapping_cupy.gridmap_utils import encode_layer_to_multiarray, decode_multiarray_to_rows_cols

PDC_DATATYPE = {
    "1": np.int8,
    "2": np.uint8,
    "3": np.int16,
    "4": np.uint16,
    "5": np.int32,
    "6": np.uint32,
    "7": np.float32,
    "8": np.float64,
}

def _pointcloud2_xyz_f32(msg: PointCloud2) -> np.ndarray:
    """
    Convert a PointCloud2 into an (N,3) float32 numpy array for fields (x,y,z).

    Supported (fail-loudly):
      - little-endian clouds
      - fields x,y,z present and FLOAT32

    This intentionally does not support arbitrary field layouts or RGB/semantic channels.
    """
    if msg.is_bigendian:
        raise ValueError("PointCloud2 big-endian is not supported.")

    want = {"x", "y", "z"}
    fields = {f.name: f for f in msg.fields}
    missing = want.difference(fields.keys())
    if missing:
        raise ValueError(f"PointCloud2 is missing required fields: {sorted(missing)}")

    for name in ("x", "y", "z"):
        f = fields[name]
        if f.datatype != PointField.FLOAT32 or f.count != 1:
            raise ValueError(
                f"PointCloud2 field '{name}' must be FLOAT32 count=1, got datatype={f.datatype} count={f.count}"
            )

    dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (np.float32, np.float32, np.float32),
            "offsets": (fields["x"].offset, fields["y"].offset, fields["z"].offset),
            "itemsize": msg.point_step,
        }
    )
    arr = np.frombuffer(msg.data, dtype=dtype)
    pts = np.stack((arr["x"], arr["y"], arr["z"]), axis=-1).astype(np.float32, copy=False)

    if not msg.is_dense:
        good = np.isfinite(pts).all(axis=1)
        pts = pts[good]
    return pts

class ElevationMappingNode(Node):
    def __init__(self):
        super().__init__(
            'elevation_mapping_node',
            automatically_declare_parameters_from_overrides=True,
            allow_undeclared_parameters=False
        )

        self.root = get_package_share_directory("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        # Initialize parameters with some defaults
        self.param = Parameter(
            use_chainer=False,
            weight_file=weight_file,
            plugin_config_file=plugin_config_file
        )

        # Read ROS parameters (including YAML)
        self.initialize_ros()
        self.set_param_values_from_ros()

        # Overwrite subscriber_cfg from loaded YAML
        self.param.subscriber_cfg = self.my_subscribers

        self.initialize_elevation_mapping()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()
        self.register_services()
        self._last_t = None

    def initialize_elevation_mapping(self) -> None:
        self.param.update()
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        self._map_data = np.zeros(
            (self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32
        )
        self.get_logger().info(f"Initialized map with length: {self._map.map_length}, resolution: {self._map.resolution}, cells: {self._map.cell_n}")

        self._map_q = None
        self._map_t = None

    def initialize_ros(self) -> None:
        self._tf_buffer = tf2_ros.Buffer()
        # The TF listener gets its own node and executor thread (as the ROS 1 tf2 listener had).
        # With the default listener the /tf callbacks run on this node's single-threaded executor,
        # which is saturated by the GPU point-cloud processing: the buffer then falls seconds to
        # minutes behind, pose_update() silently recentres the map on a stale pose (map frozen at
        # the start position / trailing the robot), and the un-consumed reliable subscription
        # backs up every /tf publisher on the robot. A separate node is required because rclpy
        # allows a node in only one executor (spin_thread=True on the same node is a no-op once
        # main() adds it to its own executor).
        # /tf is subscribed best-effort so this node can never stall the /tf writers again; tf2
        # interpolates over dropped samples. /tf_static keeps the transient-local default.
        # use_global_arguments=False: the launch file's `-r __node:=...` remap would otherwise rename
        # this node to the main node's name (duplicate node names, duplicated parameter services).
        self._tf_node = rclpy.create_node(
            self.get_name() + '_tf_listener',
            namespace=self.get_namespace(),
            use_global_arguments=False,
            start_parameter_services=False,
        )
        tf_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._listener = tf2_ros.TransformListener(
            self._tf_buffer, self._tf_node, spin_thread=True, qos=tf_qos
        )
        self.get_ros_params()

    def get_ros_params(self) -> None:
        self.use_chainer = self.get_parameter('use_chainer').get_parameter_value().bool_value
        self.initialize_frame_id = self.get_parameter(
            'initialize_frame_id'
        ).get_parameter_value().string_array_value
        self.initialize_tf_offset = self.get_parameter('initialize_tf_offset').get_parameter_value().double_array_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.corrected_map_frame = self.get_parameter('corrected_map_frame').get_parameter_value().string_value
        self.initialize_method = self.get_parameter('initialize_method').get_parameter_value().string_value
        self.position_lowpass_alpha = self.get_parameter('position_lowpass_alpha').get_parameter_value().double_value
        self.orientation_lowpass_alpha = self.get_parameter('orientation_lowpass_alpha').get_parameter_value().double_value
        self.recordable_fps = self.get_parameter('recordable_fps').get_parameter_value().double_value
        self.update_variance_fps = self.get_parameter('update_variance_fps').get_parameter_value().double_value
        self.time_interval = self.get_parameter('time_interval').get_parameter_value().double_value
        self.update_pose_fps = self.get_parameter('update_pose_fps').get_parameter_value().double_value
        if not self.has_parameter('tf_stale_warn_s'):
            self.declare_parameter('tf_stale_warn_s', 0.5)
        self.tf_stale_warn_s = self.get_parameter('tf_stale_warn_s').get_parameter_value().double_value
        if not self.has_parameter('cupy_memory_pool_trim_interval_s'):
            self.declare_parameter('cupy_memory_pool_trim_interval_s', 5.0)
        self.cupy_memory_pool_trim_interval_s = float(
            self.get_parameter('cupy_memory_pool_trim_interval_s').value
        )
        self.initialize_tf_grid_size = self.get_parameter('initialize_tf_grid_size').get_parameter_value().double_value
        self.map_acquire_fps = self.get_parameter('map_acquire_fps').get_parameter_value().double_value
        self.publish_statistics_fps = self.get_parameter('publish_statistics_fps').get_parameter_value().double_value
        self.enable_pointcloud_publishing = self.get_parameter('enable_pointcloud_publishing').get_parameter_value().bool_value
        self.enable_normal_arrow_publishing = self.get_parameter('enable_normal_arrow_publishing').get_parameter_value().bool_value
        self.enable_drift_corrected_TF_publishing = self.get_parameter('enable_drift_corrected_TF_publishing').get_parameter_value().bool_value
        self.use_initializer_at_start = self.get_parameter('use_initializer_at_start').get_parameter_value().bool_value
        if not self.has_parameter('always_clear_with_initializer'):
            self.declare_parameter('always_clear_with_initializer', False)
        self.always_clear_with_initializer = self.get_parameter('always_clear_with_initializer').get_parameter_value().bool_value
        subscribers_params = self.get_parameters_by_prefix('subscribers')
        self.my_subscribers = {}
        for param_name, param_value in subscribers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                sub_key, sub_param = parts[:2]
                if sub_key not in self.my_subscribers:
                    self.my_subscribers[sub_key] = {}
                self.my_subscribers[sub_key][sub_param] = param_value.value
        publishers_params = self.get_parameters_by_prefix('publishers')
        self.my_publishers = {}
        for param_name, param_value in publishers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                pub_key, pub_param = parts[:2]
                if pub_key not in self.my_publishers:
                    self.my_publishers[pub_key] = {}
                self.my_publishers[pub_key][pub_param] = param_value.value


    def set_param_values_from_ros(self):
        # Assign to self.param so it won't use defaults. This is research code: crash loudly if
        # a required parameter is missing or mistyped.
        self.param.use_chainer = self.use_chainer
        if self.has_parameter("plugin_config_file"):
            plugin_config_file = self.get_parameter("plugin_config_file").get_parameter_value().string_value
            assert plugin_config_file
            self.param.plugin_config_file = plugin_config_file
        if self.has_parameter("weight_file"):
            weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
            assert weight_file
            self.param.weight_file = weight_file
        self.param.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.param.map_length = self.get_parameter('map_length').get_parameter_value().double_value
        self.param.sensor_noise_factor = self.get_parameter('sensor_noise_factor').get_parameter_value().double_value
        self.param.mahalanobis_thresh = self.get_parameter('mahalanobis_thresh').get_parameter_value().double_value
        self.param.outlier_variance = self.get_parameter('outlier_variance').get_parameter_value().double_value
        self.param.drift_compensation_variance_inlier = self.get_parameter(
            'drift_compensation_variance_inlier'
        ).get_parameter_value().double_value
        self.param.checker_layer = self.get_parameter('checker_layer').get_parameter_value().string_value
        self.param.max_drift = self.get_parameter('max_drift').get_parameter_value().double_value
        self.param.drift_compensation_alpha = self.get_parameter(
            'drift_compensation_alpha'
        ).get_parameter_value().double_value
        self.param.time_variance = self.get_parameter('time_variance').get_parameter_value().double_value
        self.param.max_variance = self.get_parameter('max_variance').get_parameter_value().double_value
        self.param.initial_variance = self.get_parameter('initial_variance').get_parameter_value().double_value
        self.param.initialized_variance = self.get_parameter(
            'initialized_variance'
        ).get_parameter_value().double_value
        self.param.traversability_inlier = self.get_parameter(
            'traversability_inlier'
        ).get_parameter_value().double_value
        self.param.dilation_size = self.get_parameter('dilation_size').get_parameter_value().integer_value
        self.param.dilation_size_initialize = self.get_parameter(
            'dilation_size_initialize'
        ).get_parameter_value().integer_value
        self.param.wall_num_thresh = self.get_parameter('wall_num_thresh').get_parameter_value().integer_value
        self.param.min_height_drift_cnt = self.get_parameter(
            'min_height_drift_cnt'
        ).get_parameter_value().integer_value
        self.param.position_noise_thresh = self.get_parameter(
            'position_noise_thresh'
        ).get_parameter_value().double_value
        self.param.orientation_noise_thresh = self.get_parameter(
            'orientation_noise_thresh'
        ).get_parameter_value().double_value
        self.param.min_valid_distance = self.get_parameter(
            'min_valid_distance'
        ).get_parameter_value().double_value
        self.param.max_height_range = self.get_parameter(
            'max_height_range'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_a = self.get_parameter(
            'ramped_height_range_a'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_b = self.get_parameter(
            'ramped_height_range_b'
        ).get_parameter_value().double_value
        self.param.ramped_height_range_c = self.get_parameter(
            'ramped_height_range_c'
        ).get_parameter_value().double_value
        self.param.max_ray_length = self.get_parameter('max_ray_length').get_parameter_value().double_value
        self.param.cleanup_step = self.get_parameter('cleanup_step').get_parameter_value().double_value
        self.param.cleanup_cos_thresh = self.get_parameter(
            'cleanup_cos_thresh'
        ).get_parameter_value().double_value
        self.param.safe_thresh = self.get_parameter('safe_thresh').get_parameter_value().double_value
        self.param.safe_min_thresh = self.get_parameter('safe_min_thresh').get_parameter_value().double_value
        self.param.max_unsafe_n = self.get_parameter('max_unsafe_n').get_parameter_value().integer_value
        self.param.overlap_clear_range_xy = self.get_parameter(
            'overlap_clear_range_xy'
        ).get_parameter_value().double_value
        self.param.overlap_clear_range_z = self.get_parameter(
            'overlap_clear_range_z'
        ).get_parameter_value().double_value
        self.param.enable_edge_sharpen = self.get_parameter(
            'enable_edge_sharpen'
        ).get_parameter_value().bool_value
        self.param.enable_visibility_cleanup = self.get_parameter(
            'enable_visibility_cleanup'
        ).get_parameter_value().bool_value
        self.param.enable_drift_compensation = self.get_parameter(
            'enable_drift_compensation'
        ).get_parameter_value().bool_value
        self.param.enable_overlap_clearance = self.get_parameter(
            'enable_overlap_clearance'
        ).get_parameter_value().bool_value
        self.param.use_only_above_for_upper_bound = self.get_parameter(
            'use_only_above_for_upper_bound'
        ).get_parameter_value().bool_value

        mask_param = self.get_parameter('masked_replace_service_mask_layer_name').get_parameter_value().string_value
        topic_param = self.get_parameter('save_map_default_topic').get_parameter_value().string_value
        storage_param = self.get_parameter('save_map_storage_id').get_parameter_value().string_value
        service_ns_param = self.get_parameter('service_namespace').get_parameter_value().string_value

        if not mask_param:
            raise ValueError("masked_replace_service_mask_layer_name must be a non-empty string")
        if not topic_param:
            raise ValueError("save_map_default_topic must be a non-empty string")
        if not storage_param:
            raise ValueError("save_map_storage_id must be a non-empty string")
        if not service_ns_param:
            raise ValueError("service_namespace must be a non-empty string")

        self.masked_replace_mask_layer_name = mask_param
        self.save_map_default_topic = topic_param
        self.save_map_storage_id = storage_param
        self.service_namespace = self._normalize_namespace(service_ns_param)

    def register_subscribers(self) -> None:
        self._pointcloud_subs = {}
        self._image_syncs = {}
        self._image_filter_subs = {}
        self._channel_info_subs = {}
        self._image_channels = {}

        if any(config.get("data_type") == "image" for config in self.my_subscribers.values()):
            self.cv_bridge = CvBridge()

        for key, config in self.my_subscribers.items():
            data_type = config.get("data_type")
            if data_type == "image":
                topic_name = config.get("topic_name")
                camera_info_topic_name = config.get(
                    "camera_info_topic_name",
                    config.get("topic_name_camera_info"),
                )
                if not topic_name:
                    raise ValueError(f"Image subscriber '{key}' is missing required key 'topic_name'.")
                if not camera_info_topic_name:
                    raise ValueError(
                        f"Image subscriber '{key}' is missing required key 'camera_info_topic_name'."
                    )

                camera_sub = message_filters.Subscriber(self, Image, topic_name)
                camera_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic_name)
                image_sync = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub],
                    queue_size=10,
                    slop=0.5,
                )
                image_sync.registerCallback(partial(self.image_callback, sub_key=key))
                self._image_filter_subs[key] = [camera_sub, camera_info_sub]
                self._image_syncs[key] = image_sync

                channel_info_topic_name = config.get("channel_info_topic_name")
                if channel_info_topic_name:
                    self._channel_info_subs[key] = self.create_subscription(
                        ChannelInfo,
                        channel_info_topic_name,
                        partial(self.channel_info_callback, sub_key=key),
                        10,
                    )
                continue

            if data_type != "pointcloud":
                raise ValueError(
                    f"Unsupported subscriber data_type='{data_type}' for '{key}'. "
                    "Supported: pointcloud and image."
                )

            topic_name = config.get("topic_name")
            if not topic_name:
                raise ValueError(f"Subscriber '{key}' is missing required key 'topic_name'.")

            # Use sensor data QoS (BEST_EFFORT) for point clouds
            qos_profile = QoSPresetProfiles.get_from_short_key("sensor_data")
            self._pointcloud_subs[key] = self.create_subscription(
                PointCloud2,
                topic_name,
                partial(self.pointcloud_callback, sub_key=key),
                qos_profile,
            )

    def channel_info_callback(self, msg: ChannelInfo, sub_key: str) -> None:
        self._image_channels[sub_key] = list(msg.channels)

    def resolve_image_channels(self, sub_key: str) -> List[str]:
        configured_channels = self.param.subscriber_cfg[sub_key].get("channels", [])
        if configured_channels:
            return configured_channels

        live_channels = self._image_channels.get(sub_key, [])
        if live_channels:
            return live_channels

        self.get_logger().warning(
            (
                f"Image subscriber '{sub_key}' has no resolved channels yet. "
                "Configure 'channels' or wait for ChannelInfo."
            ),
            throttle_duration_sec=5.0,
        )
        return []

    def register_publishers(self) -> None:
        self._publishers_dict = {}
        self._publishers_timers = []

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.get_name()}/{pub_key}"
            publisher = self.create_publisher(GridMap, topic_name, 10)
            self._publishers_dict[pub_key] = publisher

            fps = pub_config.get("fps", 1.0)
            timer = self.create_timer(
                1.0 / fps,
                partial(self.publish_map, key=pub_key)
            )
            self._publishers_timers.append(timer)

    def register_timers(self) -> None:
        self.time_pose_update = self.create_timer(
            1.0 / self.update_pose_fps,
            self.pose_update
        )
        self.timer_variance = self.create_timer(
            1.0 / self.update_variance_fps,
            self.update_variance
        )
        self.timer_time = self.create_timer(
            self.time_interval,
            self.update_time
        )
        self.timer_cupy_memory_pool = None
        if self.cupy_memory_pool_trim_interval_s > 0.0:
            self.timer_cupy_memory_pool = self.create_timer(
                self.cupy_memory_pool_trim_interval_s,
                self.trim_cupy_memory_pool
            )

    def register_services(self) -> None:
        service_masked = self._resolve_service_name('masked_replace')
        service_save = self._resolve_service_name('save_map')
        service_load = self._resolve_service_name('load_map')
        service_clear = self._resolve_service_name('clear_map')
        service_initialize = self._resolve_service_name('initialize')
        service_clear_with_initializer = self._resolve_service_name('clear_map_with_initializer')

        self._srv_masked_replace = self.create_service(
            SetGridMap,
            service_masked,
            self.handle_masked_replace
        )
        self._srv_save_map = self.create_service(
            ProcessFile,
            service_save,
            self.handle_save_map
        )
        self._srv_load_map = self.create_service(
            ProcessFile,
            service_load,
            self.handle_load_map
        )
        self._srv_clear_map = self.create_service(
            Trigger,
            service_clear,
            self.handle_clear_map
        )
        self._srv_initialize = self.create_service(
            Initialize,
            service_initialize,
            self.handle_initialize
        )
        self._srv_clear_map_with_initializer = self.create_service(
            Trigger,
            service_clear_with_initializer,
            self.handle_clear_map_with_initializer
        )

    def publish_map(self, key: str) -> None:
        if self._map_q is None:
            return
        center = self._get_map_center()
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self._last_t if self._last_t is not None else self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        actual_map_length = (self._map.cell_n - 2) * self._map.resolution
        gm.info.length_x = actual_map_length
        gm.info.length_y = actual_map_length
        if self._map_t is not None:
            gm.info.pose.position.x = self._map_t.x
            gm.info.pose.position.y = self._map_t.y
            # grid_map_ros (and our RViz usage) treats GridMap as a horizontal 2.5D surface and ignores pose.z and
            # pose.orientation. Foxglove's GridMap renderer *does* apply them, which can make the map appear tilted
            # and shifted in Z when we embed the robot pose here. Keep pose.x/y as the map center in `map_frame`,
            # but publish a neutral pose for visualization sanity.
            gm.info.pose.position.z = 0.0
        else:
            gm.info.pose.position.x = float(center[0])
            gm.info.pose.position.y = float(center[1])
            gm.info.pose.position.z = 0.0

        gm.info.pose.orientation.x = 0.0
        gm.info.pose.orientation.y = 0.0
        gm.info.pose.orientation.z = 0.0
        gm.info.pose.orientation.w = 1.0
        gm.layers = []
        gm.basic_layers = self.my_publishers[key]["basic_layers"]

        for layer in self.my_publishers[key].get("layers", []):
            gm.layers.append(layer)
            self._map.get_map_with_name_ref(layer, self._map_data)
            # After fixing CUDA kernels and removing flips in elevation_mapping.py, no flip needed here
            map_data_for_gridmap = self._map_data
            gm.data.append(self._numpy_to_multiarray(map_data_for_gridmap, layout="gridmap_column"))

        gm.outer_start_index = 0
        gm.inner_start_index = 0
        self._publishers_dict[key].publish(gm)

    def handle_masked_replace(self, request, response):
        try:
            layer_arrays, geometry = self._grid_map_to_numpy(request.map)
            mask = layer_arrays.pop(self.masked_replace_mask_layer_name, None)
            if not layer_arrays:
                raise ValueError("Provide at least one data layer to update.")
            self._map.apply_masked_replace(layer_arrays, mask, geometry)
            self._republish_all_once()
            self.get_logger().info(f"masked_replace updated {len(layer_arrays)} layer(s).")
        except Exception as exc:
            self.get_logger().error(f"masked_replace failed: {exc}")
        return response

    def handle_save_map(self, request, response):
        try:
            fused_path, raw_path = self._prepare_bag_paths(request.file_path)
            topic_base = request.topic_name or self.save_map_default_topic
            fused_topic = self._resolve_topic_name(topic_base)
            raw_topic = self._resolve_topic_name(f"{topic_base}_raw")

            fused_layer_names = self._collect_fused_layer_names()
            raw_layer_names = self._map.list_layers()
            self.get_logger().info(
                f"Saving map: fused layers={fused_layer_names}, raw layers={raw_layer_names}"
            )

            fused_layers = self._map.export_layers(fused_layer_names)
            raw_layers = self._map.export_layers(raw_layer_names)
            self.get_logger().info(
                f"Exported raw layer keys: {list(raw_layers.keys())}"
            )
            if "elevation" in fused_layers:
                n_finite = int(np.isfinite(fused_layers["elevation"]).sum())
                self.get_logger().info(f"save_map: fused 'elevation' finite cells={n_finite}")
            if "is_valid" in raw_layers:
                n_valid = int((raw_layers["is_valid"] > 0.5).sum())
                self.get_logger().info(f"save_map: raw 'is_valid' valid cells={n_valid}")

            gm_fused = self._build_grid_map_message(
                fused_layer_names,
                fused_layers,
                self._collect_basic_layers(),
            )
            gm_raw = self._build_grid_map_message(
                raw_layer_names,
                raw_layers,
                ['elevation'],
            )
            self.get_logger().info(
                f"Built fused msg layers={gm_fused.layers}, raw msg layers={gm_raw.layers}"
            )

            self._write_grid_map_bag(fused_path, fused_topic, gm_fused)
            self._write_grid_map_bag(raw_path, raw_topic, gm_raw)

            response.success = True
        except Exception as exc:
            self.get_logger().error(f"save_map failed: {exc}")
            response.success = False
        return response

    def handle_load_map(self, request, response):
        try:
            fused_path = Path(request.file_path).expanduser().resolve()
            raw_path = Path(f"{fused_path}_raw")
            if not fused_path.exists():
                raise FileNotFoundError(f"Fused map bag '{fused_path}' does not exist.")
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw map bag '{raw_path}' does not exist.")

            topic_base = request.topic_name or self.save_map_default_topic
            fused_topic = self._resolve_topic_name(topic_base)
            raw_topic = self._resolve_topic_name(f"{topic_base}_raw")

            fused_msg = self._read_latest_grid_map(fused_path, fused_topic)
            raw_msg = self._read_latest_grid_map(raw_path, raw_topic)

            fused_layers, _ = self._grid_map_to_numpy(fused_msg)
            raw_layers, geometry = self._grid_map_to_numpy(raw_msg)

            self._map.set_full_map(fused_layers, raw_layers, geometry)

            pose_position = raw_msg.info.pose.position
            pose_orientation = raw_msg.info.pose.orientation
            self._map_t = Vector3(x=pose_position.x, y=pose_position.y, z=pose_position.z)
            self._map_q = Quaternion(
                x=pose_orientation.x,
                y=pose_orientation.y,
                z=pose_orientation.z,
                w=pose_orientation.w,
            )
            self._last_t = self.get_clock().now().to_msg()
            self._republish_all_once()
            # Quick sanity: the restored elevation should contain at least some finite values.
            tmp = np.zeros((self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32)
            self._map.get_map_with_name_ref("elevation", tmp)
            n_finite = int(np.isfinite(tmp).sum())
            self.get_logger().info(f"load_map: restored 'elevation' finite cells={n_finite}")

            response.success = True
        except Exception as exc:
            self.get_logger().error(f"load_map failed: {exc}")
            response.success = False
        return response

    def handle_clear_map(self, request, response):
        del request
        try:
            self._map.clear()
            if self.always_clear_with_initializer:
                self._initialize_with_tf()
            self._last_t = self.get_clock().now().to_msg()
            self._republish_all_once()
            response.success = True
            response.message = "Elevation map cleared."
            self.get_logger().info("clear_map: reset elevation map to empty state.")
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"clear_map failed: {exc}")
        return response

    def handle_clear_map_with_initializer(self, request, response):
        del request
        try:
            self._map.clear()
            initialized = self._initialize_with_tf()
            self._last_t = self.get_clock().now().to_msg()
            self._republish_all_once()
            response.success = initialized
            response.message = "Elevation map cleared with TF initializer." if initialized else "TF initializer failed."
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"clear_map_with_initializer failed: {exc}")
        return response

    def handle_initialize(self, request, response):
        try:
            if request.type != Initialize.Request.POINTS:
                raise ValueError(f"Unsupported initialize request type: {request.type}")
            points = []
            for point in request.points:
                point_frame_id = point.header.frame_id
                p = np.array([point.point.x, point.point.y, point.point.z], dtype=np.float32)
                if point_frame_id and point_frame_id != self.map_frame:
                    transform = self.safe_lookup_transform(self.map_frame, point_frame_id, point.header.stamp)
                    if transform is None:
                        response.success = False
                        return response
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
                    p = R @ p + np.array([t.x, t.y, t.z], dtype=np.float32)
                points.append(p)

            method = self._initialize_method_from_request(request.method)
            self.get_logger().info(f"Initializing map with {len(points)} point(s) using {method}.")
            self._map.initialize_map(np.asarray(points, dtype=np.float32), method)
            self._last_t = self.get_clock().now().to_msg()
            self._republish_all_once()
            response.success = True
        except Exception as exc:
            response.success = False
            self.get_logger().error(f"initialize failed: {exc}")
        return response

    def _grid_map_to_numpy(self, grid_map_msg: GridMap):
        if len(grid_map_msg.layers) != len(grid_map_msg.data):
            raise ValueError("Mismatch between GridMap layers and data arrays.")

        arrays: Dict[str, np.ndarray] = {}
        for name, array_msg in zip(grid_map_msg.layers, grid_map_msg.data):
            arrays[name] = decode_multiarray_to_rows_cols(name, array_msg)

        center = np.array(
            [
                grid_map_msg.info.pose.position.x,
                grid_map_msg.info.pose.position.y,
                grid_map_msg.info.pose.position.z,
            ],
            dtype=np.float32,
        )
        orientation = np.array(
            [
                grid_map_msg.info.pose.orientation.x,
                grid_map_msg.info.pose.orientation.y,
                grid_map_msg.info.pose.orientation.z,
                grid_map_msg.info.pose.orientation.w,
            ],
            dtype=np.float32,
        )

        geometry = GridGeometry(
            length_x=grid_map_msg.info.length_x,
            length_y=grid_map_msg.info.length_y,
            resolution=grid_map_msg.info.resolution,
            center=center,
            orientation=orientation,
        )
        return arrays, geometry

    def _extract_layout_shape(self, array_msg: Float32MultiArray) -> tuple:
        if array_msg.layout.dim:
            cols = array_msg.layout.dim[0].size or 1
            rows = array_msg.layout.dim[1].size if len(array_msg.layout.dim) > 1 else (
                len(array_msg.data) // cols if cols else len(array_msg.data)
            )
        else:
            cols = int(math.sqrt(len(array_msg.data)))
            rows = cols
        return cols, rows

    def _collect_fused_layer_names(self) -> List[str]:
        fused: List[str] = []
        for config in self.my_publishers.values():
            fused.extend(config.get('layers', []))
        if not fused:
            fused = ['elevation']
        ordered: List[str] = []
        for name in fused:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _collect_basic_layers(self) -> List[str]:
        basics: List[str] = []
        for config in self.my_publishers.values():
            basics.extend(config.get('basic_layers', []))
        if not basics:
            basics = ['elevation']
        ordered: List[str] = []
        for name in basics:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _build_grid_map_message(
        self,
        layer_names: List[str],
        layer_data: Dict[str, np.ndarray],
        basic_layers: List[str],
    ) -> GridMap:
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self._last_t if self._last_t is not None else self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        actual_map_length = (self._map.cell_n - 2) * self._map.resolution
        gm.info.length_x = actual_map_length
        gm.info.length_y = actual_map_length

        center = self._get_map_center()
        gm.info.pose.position.x = float(center[0])
        gm.info.pose.position.y = float(center[1])
        gm.info.pose.position.z = float(center[2])
        if self._map_q is not None:
            gm.info.pose.orientation.x = self._map_q.x
            gm.info.pose.orientation.y = self._map_q.y
            gm.info.pose.orientation.z = self._map_q.z
            gm.info.pose.orientation.w = self._map_q.w
        else:
            gm.info.pose.orientation.w = 1.0

        gm.layers = []
        gm.basic_layers = basic_layers
        for name in layer_names:
            data = layer_data.get(name)
            if data is None:
                continue
            gm.layers.append(name)
            gm.data.append(self._numpy_to_multiarray(data))
        gm.outer_start_index = 0
        gm.inner_start_index = 0
        return gm

    def _numpy_to_multiarray(self, data: np.ndarray, layout: str = "gridmap_column") -> Float32MultiArray:
        return encode_layer_to_multiarray(data, layout=layout)

    def _resolve_service_name(self, suffix: str) -> str:
        base = self.service_namespace
        if not base:
            base = f"/{self.get_name()}"
        return f"{base}/{suffix}".replace('//', '/')

    def _resolve_topic_name(self, topic: str) -> str:
        topic = topic.strip('/') or self.save_map_default_topic
        base = self.service_namespace
        if not base:
            base = f"/{self.get_name()}"
        return f"{base}/{topic}".replace('//', '/')

    def _prepare_bag_paths(self, file_path: str):
        if not file_path:
            raise ValueError("file_path must be provided.")
        fused_path = Path(file_path).expanduser().resolve()
        raw_path = Path(f"{fused_path}_raw")
        if fused_path.exists():
            raise FileExistsError(f"Bag path '{fused_path}' already exists.")
        if raw_path.exists():
            raise FileExistsError(f"Bag path '{raw_path}' already exists.")
        fused_path.parent.mkdir(parents=True, exist_ok=True)
        return fused_path, raw_path

    def _make_topic_metadata(self, topic: str) -> rosbag2_py.TopicMetadata:
        msg_type = "grid_map_msgs/msg/GridMap"
        serialization_format = "cdr"
        return rosbag2_py.TopicMetadata(0, topic, msg_type, serialization_format)

    def _write_grid_map_bag(self, path: Path, topic: str, grid_map_msg: GridMap) -> None:
        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id=self.save_map_storage_id)
        converter_options = rosbag2_py.ConverterOptions('', '')
        writer.open(storage_options, converter_options)
        topic_metadata = self._make_topic_metadata(topic)
        writer.create_topic(topic_metadata)
        writer.write(topic, serialize_message(grid_map_msg), self.get_clock().now().nanoseconds)

    def _read_latest_grid_map(self, path: Path, topic: str) -> GridMap:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id=self.save_map_storage_id)
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)
        latest = None
        while reader.has_next():
            current_topic, data, _ = reader.read_next()
            if current_topic != topic:
                continue
            msg = deserialize_message(data, GridMap)
            latest = msg
        if latest is None:
            raise ValueError(f"No messages for topic '{topic}' in bag '{path}'.")
        return latest

    def _get_map_center(self) -> np.ndarray:
        center = np.zeros((1, 3), dtype=np.float32)
        self._map.get_center_position(center)
        return center[0]

    def _republish_all_once(self) -> None:
        if self._map_q is None:
            return
        for key in self._publishers_dict.keys():
            self.publish_map(key)

    def _normalize_namespace(self, value: str) -> str:
        value = value.strip() if value else ''
        if not value:
            return ''
        if not value.startswith('/'):
            value = f'/{value}'
        return value.rstrip('/')

    def _initialize_method_from_request(self, method: int) -> str:
        if method == Initialize.Request.NEAREST:
            return "nearest"
        if method == Initialize.Request.LINEAR:
            return "linear"
        if method == Initialize.Request.CUBIC:
            return "cubic"
        raise ValueError(f"Unsupported initialize method: {method}")

    def _initialize_with_tf(self) -> bool:
        points = []
        last_point: Optional[np.ndarray] = None
        stamp = self.get_clock().now().to_msg()

        if len(self.initialize_tf_offset) < len(self.initialize_frame_id):
            self.get_logger().error(
                "initialize_tf_offset must have at least as many entries as initialize_frame_id."
            )
            return False

        for idx, frame_id in enumerate(self.initialize_frame_id):
            transform = self.safe_lookup_transform(self.map_frame, frame_id, stamp)
            if transform is None:
                return False

            t = transform.transform.translation
            point = np.array([t.x, t.y, t.z + self.initialize_tf_offset[idx]], dtype=np.float32)
            points.append(point)
            last_point = point

        if points and len(points) < 3 and last_point is not None:
            s = float(self.initialize_tf_grid_size)
            points.extend(
                [
                    last_point + np.array([s, s, 0.0], dtype=np.float32),
                    last_point + np.array([-s, s, 0.0], dtype=np.float32),
                    last_point + np.array([s, -s, 0.0], dtype=np.float32),
                    last_point + np.array([-s, -s, 0.0], dtype=np.float32),
                ]
            )

        if not points:
            self.get_logger().error("No initialize_frame_id entries are configured.")
            return False

        self.get_logger().info(f"Initializing map with TF points using {self.initialize_method}.")
        self._map.initialize_map(np.asarray(points, dtype=np.float32), self.initialize_method)
        self._republish_all_once()
        return True

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time
            )
        except tf2_ros.ExtrapolationException:
            # Time is in the future/past, try with latest available
            try:
                latest = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time()
                )
                # The fallback is silent by design, but a stale "latest" transform means the map
                # is being recentred / ray-traced with an old robot pose. Make that visible.
                age_s = (self.get_clock().now() - rclpy.time.Time.from_msg(latest.header.stamp)).nanoseconds * 1e-9
                if age_s > self.tf_stale_warn_s:
                    self.get_logger().warning(
                        f"Latest TF '{source_frame}' -> '{target_frame}' is {age_s:.2f} s old "
                        f"(requested {rclpy.time.Time.from_msg(time).nanoseconds * 1e-9:.3f}); "
                        "map pose / ray origin are lagging.",
                        throttle_duration_sec=2.0,
                    )
                return latest
            # NOTE: The second lookup can also throw ExtrapolationException (e.g., TF buffer not populated yet,
            # or timestamps are discontinuous during sim resets). If we don't catch it here the whole node dies.
            except (
                tf2.LookupException,
                tf2.ConnectivityException,
                tf2.ExtrapolationException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.get_logger().warning(
                    f"Transform from '{source_frame}' to '{target_frame}' not available: {e}",
                    throttle_duration_sec=5.0
                )
                return None
        except tf2.LookupException as e:
            # Frame doesn't exist
            self.get_logger().warning(
                f"Frame '{target_frame}' or '{source_frame}' does not exist: {e}",
                throttle_duration_sec=5.0
            )
            return None
        except tf2.ConnectivityException as e:
            # No transform path between frames
            self.get_logger().warning(
                f"No transform path from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=5.0
            )
            return None
        except Exception as e:
            # Catch any other unexpected TF2 errors
            self.get_logger().warning(
                f"Unexpected TF2 error for transform from '{source_frame}' to '{target_frame}': {e}",
                throttle_duration_sec=5.0
            )
            return None

    def image_callback(self, camera_msg: Image, camera_info_msg: CameraInfo, sub_key: str) -> None:
        self._last_t = camera_msg.header.stamp

        frame_sensor_id = camera_msg.header.frame_id
        if not frame_sensor_id:
            raise ValueError("Image header.frame_id is empty.")

        semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")
        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, idx] for idx in range(semantic_img.shape[2])]
        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(camera_info_msg.d, dtype=np.float32).reshape(-1, 1)

        if frame_sensor_id == self.map_frame:
            t_np = np.zeros(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
        else:
            transform_camera_to_map = self.safe_lookup_transform(
                self.map_frame,
                frame_sensor_id,
                camera_msg.header.stamp,
            )
            if transform_camera_to_map is None:
                return
            t = transform_camera_to_map.transform.translation
            q = transform_camera_to_map.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

        channels = self.resolve_image_channels(sub_key)
        if not channels:
            return

        self._map.input_image(
            semantic_img,
            channels,
            R,
            t_np,
            K,
            D,
            camera_info_msg.distortion_model,
            camera_info_msg.height,
            camera_info_msg.width,
        )
        self._image_process_counter += 1

    def pointcloud_callback(self, msg: PointCloud2, sub_key: str) -> None:
        self._last_t = msg.header.stamp
        additional_channels = list(self.param.subscriber_cfg[sub_key].get("channels", []))
        channels = ["x", "y", "z"] + additional_channels

        if additional_channels:
            points = rnp.numpify(msg)
            if points is None:
                return

            if isinstance(points, dict):
                if not points:
                    return
                if "xyz" in points:
                    xyz_array = np.array(points["xyz"])
                    if xyz_array.ndim == 2 and xyz_array.shape[1] == 3:
                        pts = xyz_array
                    elif xyz_array.ndim == 1:
                        pts = xyz_array.reshape(-1, 3)
                    else:
                        pts = xyz_array[:, :3]
                elif all(name in points for name in ("x", "y", "z")):
                    pts = np.column_stack(
                        (
                            np.array(points["x"]).flatten(),
                            np.array(points["y"]).flatten(),
                            np.array(points["z"]).flatten(),
                        )
                    )
                else:
                    raise ValueError(
                        f"PointCloud2 dict for '{sub_key}' is missing xyz fields. "
                        f"Available: {list(points.keys())}"
                    )
                for channel in additional_channels:
                    if channel not in points:
                        raise ValueError(
                            f"PointCloud2 for '{sub_key}' is missing configured channel '{channel}'."
                        )
                    data = np.array(points[channel]).flatten()
                    if data.ndim == 1:
                        data = data[:, np.newaxis]
                    pts = np.hstack((pts, data))
            else:
                if points.size == 0:
                    return
                pts = rnp.point_cloud2.get_xyz_points(points)
                for channel in additional_channels:
                    if not hasattr(points, "dtype") or channel not in points.dtype.names:
                        raise ValueError(
                            f"PointCloud2 for '{sub_key}' is missing configured channel '{channel}'."
                        )
                    data = points[channel].flatten()
                    if data.ndim == 1:
                        data = data[:, np.newaxis]
                    pts = np.hstack((pts, data))
        else:
            pts = _pointcloud2_xyz_f32(msg)
        if pts.size == 0:
            return

        frame_sensor_id = msg.header.frame_id
        if not frame_sensor_id:
            raise ValueError("PointCloud2 header.frame_id is empty.")

        if frame_sensor_id == self.map_frame:
            t_np = np.zeros(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
        else:
            transform_sensor_to_map = self.safe_lookup_transform(
                self.map_frame,
                frame_sensor_id,
                msg.header.stamp,
            )
            if transform_sensor_to_map is None:
                # Transform not available yet.
                return
            t = transform_sensor_to_map.transform.translation
            q = transform_sensor_to_map.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

        sensor_t_np = t_np
        ray_source_frame = self.param.subscriber_cfg[sub_key].get("ray_source_frame")
        if ray_source_frame:
            if ray_source_frame == self.map_frame:
                sensor_t_np = np.zeros(3, dtype=np.float32)
            else:
                transform_ray_source_to_map = self.safe_lookup_transform(
                    self.map_frame,
                    ray_source_frame,
                    msg.header.stamp,
                )
                if transform_ray_source_to_map is None:
                    self.get_logger().error(
                        (
                            f"[{sub_key}] ray_source_frame '{ray_source_frame}' is not available. "
                            "Skipping cloud instead of falling back to the point cloud frame."
                        ),
                        throttle_duration_sec=2.0,
                    )
                    return
                ray_t = transform_ray_source_to_map.transform.translation
                sensor_t_np = np.array([ray_t.x, ray_t.y, ray_t.z], dtype=np.float32)
            self.get_logger().info(
                (
                    f"[{sub_key}] point frame: {frame_sensor_id}, cleanup ray source frame: "
                    f"{ray_source_frame}, ray origin in {self.map_frame} = "
                    f"[{sensor_t_np[0]:.3f}, {sensor_t_np[1]:.3f}, {sensor_t_np[2]:.3f}]"
                ),
                throttle_duration_sec=5.0,
            )

        self._map.input_pointcloud(pts, channels, R, t_np, 0, 0, sensor_t_np)
        self._pointcloud_process_counter += 1

    def pose_update(self) -> None:
        stamp = self._last_t if self._last_t is not None else self.get_clock().now().to_msg()
        transform = self.safe_lookup_transform(
            self.map_frame,
            self.base_frame,
            stamp
        )
        if transform is None:
            # Transform not available, skip pose update
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        self._map.move_to(trans, rot)
        self._map_t = t
        self._map_q = q
        if self.use_initializer_at_start:
            self.get_logger().info("Clearing map with initializer.", throttle_duration_sec=5.0)
            if self._initialize_with_tf():
                self.use_initializer_at_start = False

    def update_variance(self) -> None:
        self._map.update_variance()

    def update_time(self) -> None:
        self._map.update_time()

    def trim_cupy_memory_pool(self) -> None:
        self._map.trim_memory_pool()

    def destroy_node(self) -> None:
        # Stop the dedicated TF listener thread/node first.
        try:
            if getattr(self, '_listener', None) is not None and getattr(self._listener, 'executor', None) is not None:
                self._listener.executor.shutdown()
            if getattr(self, '_tf_node', None) is not None:
                self._tf_node.destroy_node()
        except Exception as exc:  # shutdown must never raise
            self.get_logger().debug(f"TF listener shutdown: {exc}")
        super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationMappingNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # launch_testing / signal handlers can already have shut down the context.
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
