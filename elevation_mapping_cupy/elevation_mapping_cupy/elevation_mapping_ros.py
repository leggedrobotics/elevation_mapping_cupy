#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from functools import partial
import threading
import time  # Import time module for benchmarking

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory
import ros2_numpy as rnp
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from tf_transformations import quaternion_matrix
import tf2_ros

import message_filters
from cv_bridge import CvBridge
from rclpy.duration import Duration
# Message imports
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from geometry_msgs.msg import TransformStamped

# Custom module imports
from elevation_mapping_cupy import ElevationMap, Parameter

import concurrent.futures

# Define point cloud data types
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


class ElevationMappingNode(Node):
    def __init__(self):
        # Initialize the node without passing callback_group to super()
        super().__init__(
            'elevation_mapping_node',
            automatically_declare_parameters_from_overrides=True
        )

        # Create a ReentrantCallbackGroup for subscriptions and timers
        self.callback_group = ReentrantCallbackGroup()

        # Get package share directory
        self.root = get_package_share_directory("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        # Initialize Parameters
        self.param = Parameter(
            use_chainer=False,
            weight_file=weight_file,
            plugin_config_file=plugin_config_file
        )

        self.node_name = "elevation_mapping"

        # Initialize ROS components
        self.initialize_ros()

        # Assign subscriber configuration to parameters
        self.param.subscriber_cfg = self.my_subscribers

        # Initialize Elevation Mapping
        self.initialize_elevation_mapping()

        # Register ROS subscribers, publishers, and timers
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()

        # Initialize last timestamp
        self._last_t = None

        # Initialize transform storage with thread safety
        self._transform_lock = threading.Lock()
        self._current_transform = None

        # Initialize ThreadPoolExecutor for pointcloud processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.get_logger().info("ThreadPoolExecutor initialized with 2 workers.")

    def initialize_elevation_mapping(self):
        self.param.update()
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        self._map_data = np.zeros(
            (self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32
        )
        self._map_q = None
        self._map_t = None

    def initialize_ros(self):
        # Initialize TF buffer and listener with increased cache time
        self._tf_buffer = tf2_ros.Buffer()
        # Initialize TransformListener
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Retrieve ROS parameters
        self.get_ros_params()

    def get_ros_params(self):
        # Get parameters with default values
        self.use_chainer = self.get_parameter('use_chainer').get_parameter_value().bool_value
        self.weight_file = self.get_parameter('weight_file').get_parameter_value().string_value
        self.plugin_config_file = self.get_parameter('plugin_config_file').get_parameter_value().string_value
        self.initialize_frame_id = self.get_parameter('initialize_frame_id').get_parameter_value().string_value
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
        self.initialize_tf_grid_size = self.get_parameter('initialize_tf_grid_size').get_parameter_value().double_value
        self.map_acquire_fps = self.get_parameter('map_acquire_fps').get_parameter_value().double_value
        self.publish_statistics_fps = self.get_parameter('publish_statistics_fps').get_parameter_value().double_value
        self.enable_pointcloud_publishing = self.get_parameter('enable_pointcloud_publishing').get_parameter_value().bool_value
        self.enable_normal_arrow_publishing = self.get_parameter('enable_normal_arrow_publishing').get_parameter_value().bool_value
        self.enable_drift_corrected_TF_publishing = self.get_parameter('enable_drift_corrected_TF_publishing').get_parameter_value().bool_value
        self.use_initializer_at_start = self.get_parameter('use_initializer_at_start').get_parameter_value().bool_value

        # Log parameters
        self.get_logger().info("\n=== Loaded Parameters ===")
        self.get_logger().info(f"use_chainer: {self.use_chainer}")
        self.get_logger().info(f"weight_file: {self.weight_file}")
        self.get_logger().info(f"plugin_config_file: {self.plugin_config_file}")
        self.get_logger().info(f"initialize_frame_id: {self.initialize_frame_id}")
        self.get_logger().info(f"initialize_tf_offset: {self.initialize_tf_offset}")
        self.get_logger().info(f"map_frame: {self.map_frame}")
        self.get_logger().info(f"base_frame: {self.base_frame}")
        self.get_logger().info(f"corrected_map_frame: {self.corrected_map_frame}")
        self.get_logger().info(f"initialize_method: {self.initialize_method}")
        self.get_logger().info(f"position_lowpass_alpha: {self.position_lowpass_alpha}")
        self.get_logger().info(f"orientation_lowpass_alpha: {self.orientation_lowpass_alpha}")
        self.get_logger().info(f"recordable_fps: {self.recordable_fps}")
        self.get_logger().info(f"update_variance_fps: {self.update_variance_fps}")
        self.get_logger().info(f"time_interval: {self.time_interval}")
        self.get_logger().info(f"update_pose_fps: {self.update_pose_fps}")
        self.get_logger().info(f"initialize_tf_grid_size: {self.initialize_tf_grid_size}")
        self.get_logger().info(f"map_acquire_fps: {self.map_acquire_fps}")
        self.get_logger().info(f"publish_statistics_fps: {self.publish_statistics_fps}")
        self.get_logger().info(f"enable_pointcloud_publishing: {self.enable_pointcloud_publishing}")
        self.get_logger().info(f"enable_normal_arrow_publishing: {self.enable_normal_arrow_publishing}")
        self.get_logger().info(f"enable_drift_corrected_TF_publishing: {self.enable_drift_corrected_TF_publishing}")
        self.get_logger().info(f"use_initializer_at_start: {self.use_initializer_at_start}")
        self.get_logger().info("=======================\n")

        # Retrieve subscribers using prefix
        subscribers_params = self.get_parameters_by_prefix('subscribers')
        self.get_logger().info(f"Subscribers params: {subscribers_params}")
        self.my_subscribers = {}
        for param_name, param_value in subscribers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                sub_key, sub_param = parts[:2]
                if sub_key not in self.my_subscribers:
                    self.my_subscribers[sub_key] = {}
                self.my_subscribers[sub_key][sub_param] = param_value.value
            else:
                self.get_logger().warn(f"Unexpected subscriber parameter format: {param_name}")

        # Retrieve publishers using prefix
        publishers_params = self.get_parameters_by_prefix('publishers')
        self.get_logger().info(f"Publishers params: {publishers_params}")
        self.my_publishers = {}
        for param_name, param_value in publishers_params.items():
            parts = param_name.split('.')
            if len(parts) >= 2:
                pub_key, pub_param = parts[:2]
                if pub_key not in self.my_publishers:
                    self.my_publishers[pub_key] = {}
                self.my_publishers[pub_key][pub_param] = param_value.value
            else:
                self.get_logger().warn(f"Unexpected publisher parameter format: {param_name}")

        # Log loaded subscribers and publishers for debugging
        self.get_logger().info(f"Loaded Subscribers: {self.my_subscribers}")
        self.get_logger().info(f"Loaded Publishers: {self.my_publishers}")

        # If you want to list all parameters for debugging
        params = self.get_parameters_by_prefix('')
        for param_name, param_value in params.items():
            self.get_logger().info(f"Parameter '{param_name}': {param_value.value}")

    def register_subscribers(self):
        # Initialize CvBridge if needed
        for config in self.my_subscribers.values():
            if config.get("data_type") == "image":
                self.cv_bridge = CvBridge()
                break

        # Dictionaries to hold subscribers
        pointcloud_subs = {}
        image_subs = {}

        for key, config in self.my_subscribers.items():
            data_type = config.get("data_type")
            if data_type == "image":
                topic_name_camera = config.get("topic_name_camera", "/camera/image")
                topic_name_camera_info = config.get("topic_name_camera_info", "/camera/camera_info")

                # Initialize message_filters Subscribers without passing callback_group
                camera_sub = message_filters.Subscriber(
                    self,
                    Image,
                    topic_name_camera
                )
                camera_info_sub = message_filters.Subscriber(
                    self,
                    CameraInfo,
                    topic_name_camera_info
                )

                # Synchronize image and camera info
                image_sync = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub], queue_size=10, slop=0.5
                )
                image_sync.registerCallback(partial(self.image_callback, sub_key=key))
                image_subs[key] = image_sync

            elif data_type == "pointcloud":
                topic_name = config.get("topic_name", "/pointcloud")
                qos_profile = rclpy.qos.QoSProfile(
                    depth=10,
                    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST
                )

                # Create subscription with ReentrantCallbackGroup
                subscription = self.create_subscription(
                    PointCloud2,
                    topic_name,
                    partial(self.pointcloud_callback, sub_key=key),
                    qos_profile,
                    callback_group=self.callback_group  # Assign ReentrantCallbackGroup
                )
                pointcloud_subs[key] = subscription

            else:
                self.get_logger().warn(f"Unknown data_type '{data_type}' for subscriber '{key}'")

        self.get_logger().info(
            f"Registered {len(pointcloud_subs)} pointcloud subscribers and {len(image_subs)} image subscribers."
        )

    def register_publishers(self):
        self._publishers_dict = {}
        self._publishers_timers = []

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.node_name}/{pub_key}"
            publisher = self.create_publisher(GridMap, topic_name, 10)
            self._publishers_dict[pub_key] = publisher

            fps = pub_config.get("fps", 1.0)
            timer = self.create_timer(
                1.0 / fps,
                partial(self.publish_map, key=pub_key),
                callback_group=self.callback_group  # Assign ReentrantCallbackGroup
            )
            self._publishers_timers.append(timer)

            self.get_logger().info(f"Publisher '{pub_key}' registered on topic '{topic_name}' with {fps} Hz.")

    def register_timers(self):
        # Remove separate transform_updater and update_pose timers
        # Register the combined timer instead
        self.time_pose_update = self.create_timer(
            0.1,
            self.pose_update,
            callback_group=self.callback_group  # Assign ReentrantCallbackGroup
        )

        # Register additional timers with the ReentrantCallbackGroup
        self.timer_variance = self.create_timer(
            1.0 / self.update_variance_fps,
            self.update_variance,
            callback_group=self.callback_group  # Assign ReentrantCallbackGroup
        )
        self.timer_time = self.create_timer(
            self.time_interval,
            self.update_time,
            callback_group=self.callback_group  # Assign ReentrantCallbackGroup
        )

        self.get_logger().info("Combined transform and pose updater timer, variance, and time timers registered.")

    def publish_map(self, key):
        self.get_logger().info("publish_map entered")
        if self._map_q is None:
            self.get_logger().info("No map pose available for publishing.")
            return
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        gm.info.length_x = self._map.map_length
        gm.info.length_y = self._map.map_length
        gm.info.pose.position.x = self._map_t.x
        gm.info.pose.position.y = self._map_t.y
        gm.info.pose.position.z = 0.0
        gm.info.pose.orientation.w = 1.0  # self._map_q.w
        gm.info.pose.orientation.x = 0.0  # self._map_q.x
        gm.info.pose.orientation.y = 0.0  # self._map_q.y
        gm.info.pose.orientation.z = 0.0  # self._map_q.z
        gm.layers = []
        gm.basic_layers = self.my_publishers[key]["basic_layers"]
        for layer in self.my_publishers[key].get("layers", []):
            gm.layers.append(layer)
            self._map.get_map_with_name_ref(layer, self._map_data)
            
            arr = Float32MultiArray()
            arr.layout = MAL()
            N = self._map_data.shape[0]
            arr.layout.dim.append(MAD(label="column_index", size=N, stride=int(N * N)))
            arr.layout.dim.append(MAD(label="row_index", size=N, stride=N))
            
            # Convert to a Python list to satisfy the message requirements
            arr.data = self._map_data.T.flatten().tolist()
            
            gm.data.append(arr)
        gm.outer_start_index = 0
        gm.inner_start_index = 0

        self._publishers_dict[key].publish(gm)


    def image_callback(self, camera_msg, camera_info_msg, sub_key):
        # Get pose of image
        ti = camera_msg.header.stamp
        self._last_t = ti

        # Retrieve the latest transform
        transform = self.get_current_transform()
        if transform is None:
            self.get_logger().warn("No available transform for image_callback.")
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        t_np = np.array([t.x, t.y, t.z])
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return

        # Ensure image is grayscale or split channels
        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, k] for k in range(semantic_img.shape[2])]
        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(camera_info_msg.d, dtype=np.float32).reshape(-1, 1)

        if not np.all(D == 0.0):
            self.get_logger().warn("Camera distortion coefficients are not zero. Undistortion not implemented.")

        # Process image
        self._map.input_image(
            sub_key, semantic_img, R, t_np, K, D, camera_info_msg.height, camera_info_msg.width
        )
        self._image_process_counter += 1
        self.get_logger().debug(f"Images processed: {self._image_process_counter}")


    def pointcloud_callback(self, msg, sub_key):
        start_time = time.time()
        self._last_t = msg.header.stamp
        self._pointcloud_stamp = msg.header.stamp
        channels = ["x", "y", "z"] + self.param.subscriber_cfg[sub_key].get("channels", [])

        try:
            points = rnp.numpify(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to numpify pointcloud: {e}")
            return

        if points['xyz'].size == 0:
            self.get_logger().warn("Received empty point cloud.")
            return

        frame_sensor_id = msg.header.frame_id
        try:
            # Await the transform lookup
            transform_sensor_to_odom = self._tf_buffer.lookup_transform(
                self.map_frame,
                frame_sensor_id,
                self._pointcloud_stamp
            )
            
            # Process the transform directly here instead of using callback
            t = transform_sensor_to_odom.transform.translation
            q = transform_sensor_to_odom.transform.rotation
            t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
            
            # Submit the processing to the executor
            self.thread_pool.submit(self.process_pointcloud, points['xyz'], channels, R, t_np, sub_key)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in pointcloud_callback: {str(e)}")

    def process_pointcloud(self, xyz, channels, R, t_np, sub_key):
        try:
            callback_start = time.time()
            self._map.input_pointcloud(xyz, channels, R, t_np, 0, 0)
            self._pointcloud_process_counter += 1
            callback_end = time.time()
            self.get_logger().info(f"[process_pointcloud] input_pointcloud execution time: {callback_end - callback_start:.4f} seconds")
        except Exception as e:
            self.get_logger().error(f"Error processing pointcloud in separate thread: {str(e)}")

    def pose_update(self):
        if self._last_t is None:
            self.get_logger().debug("No timestamp available for pose update.")
            return

        try:
            transform = self._tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                self._last_t
            )

            # Update the current transform with thread safety
            with self._transform_lock:
                self._current_transform = transform

            # Update the pose
            t = transform.transform.translation
            q = transform.transform.rotation
            trans = np.array([t.x, t.y, t.z], dtype=np.float32)
            rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

            self._map.move_to(trans, rot)
            self._map_t = t
            self._map_q = q

            self.get_logger().debug("Pose updated.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform lookup failed in pose_update: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in pose_update: {str(e)}")

    def update_variance(self):
        self._map.update_variance()
        self.get_logger().debug("Variance updated.")

    def update_time(self):
        self._map.update_time()
        self.get_logger().debug("Time updated.")


    def get_current_transform(self):
        """Thread-safe method to retrieve the latest transform."""
        with self._transform_lock:
            return self._current_transform

    def print_tf_frames(self):
        """Print all available transforms in the TF buffer with timestamp"""
        try:
            # Get current timestamp
            current_time = self.get_clock().now()

            # Try to get the specific transform from odom to base_footprint
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.map_frame,
                    'base_footprint',
                    rclpy.time.Time(),  # get the latest transform
                    rclpy.duration.Duration(seconds=2.0)
                )
                self.get_logger().info(f"\nOdom to base_footprint transform timestamp: "
                                       f"{transform.header.stamp.sec}.{transform.header.stamp.nanosec}")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"Could not lookup odom to base_footprint transform: {e}")

            # Get all frame IDs
            frames = self._tf_buffer.all_frames_as_string()
            # Uncomment the line below if you want to log all available TF frames
            # self.get_logger().info(f"Available TF frames:\n{frames}")
        except Exception as e:
            self.get_logger().error(f"Failed to get TF frames: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down ThreadPoolExecutor.")
        self.thread_pool.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ElevationMappingNode()

    # Initialize a MultiThreadedExecutor instead of SingleThreadedExecutor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # Use the executor to spin
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ElevationMappingNode.")
    finally:
        # Shutdown the executor and destroy the node
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
