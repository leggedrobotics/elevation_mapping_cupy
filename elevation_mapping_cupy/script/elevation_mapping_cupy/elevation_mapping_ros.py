from elevation_mapping_cupy import ElevationMap
from elevation_mapping_cupy import Parameter

# General
import os
import numpy as np

# ROS
import rospy
import ros_numpy
from tf.transformations import quaternion_matrix
import tf2_ros
import rospkg
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD

import numpy as np
from functools import partial

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


class ElevationMapWrapper:
    def __init__(self):
        rospack = rospkg.RosPack()
        self.root = rospack.get_path("elevation_mapping_cupy")
        weight_file = os.path.join(self.root, "config/core/weights.dat")
        plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")
        self.param = Parameter(use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file)

        self.node_name = "elevation_mapping"

        # ROS
        self.initalize_ros()
        self.param.subscriber_cfg = self.subscribers

        self.initalize_elevation_mapping()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()

        self._last_t = None

    def initalize_elevation_mapping(self):
        self.param.update()
        # TODO add statistics across processed topics
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        self._map_data = np.zeros((self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32)
        self._map_q = None
        self._map_t = None

    def initalize_ros(self):
        rospy.init_node(self.node_name, anonymous=False)
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer)
        self.get_ros_params()

    def register_subscribers(self):
        # check if CV bridge is needed
        for config in self.param.subscriber_cfg.values():
            if config["data_type"] == "image":
                self.cv_bridge = CvBridge()
                break

        pointcloud_subs = {}
        image_subs = {}
        for key, config in self.subscribers.items():
            if config["data_type"] == "image":
                camera_sub = message_filters.Subscriber(config["topic_name_camera"], Image)
                camera_info_sub = message_filters.Subscriber(config["topic_name_camera_info"], CameraInfo)
                image_subs[key] = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub], queue_size=10, slop=0.5
                )
                image_subs[key].registerCallback(self.image_callback, key)

            elif config["data_type"] == "pointcloud":
                pointcloud_subs[key] = rospy.Subscriber(
                    config["topic_name"], PointCloud2, self.pointcloud_callback, key
                )

    def register_publishers(self):
        self._publishers = {}
        self._publishers_timers = []
        for k, v in self.publishers.items():
            self._publishers[k] = rospy.Publisher(f"/{self.node_name}/{k}", GridMap, queue_size=10)
            # partial(.) allows to pass a default argument to a callback
            self._publishers_timers.append(rospy.Timer(rospy.Duration(1 / v["fps"]), partial(self.publish_map, k)))

    def publish_map(self, k, t):
        print("publish_map")
        if self._map_q is None:
            return
        gm = GridMap()
        gm.info.header.frame_id = self.map_frame
        gm.info.header.stamp = rospy.Time.now()
        gm.info.header.seq = 0
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
        gm.basic_layers = self.publishers[k]["basic_layers"]
        for i, layer in enumerate(self.publishers[k]["layers"]):
            gm.layers.append(layer)
            try:
                self._map.get_map_with_name_ref(layer, self._map_data)
                data = self._map_data.copy()
                arr = Float32MultiArray()
                arr.layout = MAL()
                N = self._map_data.shape[0]
                arr.layout.dim.append(MAD(label="column_index", size=N, stride=int(N * N)))
                arr.layout.dim.append(MAD(label="row_index", size=N, stride=N))
                arr.data = tuple(np.ascontiguousarray(data.T).reshape(-1))
                gm.data.append(arr)
            except:
                if layer in gm.basic_layers:
                    print("Error: Missed Layer in basic layers")

        gm.outer_start_index = 0
        gm.inner_start_index = 0

        self._publishers[k].publish(gm)

    def register_timers(self):
        self.timer_variance = rospy.Timer(rospy.Duration(1 / self.update_variance_fps), self.update_variance)
        self.timer_pose = rospy.Timer(rospy.Duration(1 / self.update_pose_fps), self.update_pose)
        self.timer_time = rospy.Timer(rospy.Duration(self.time_interval), self.update_time)

    def image_callback(self, camera_msg, camera_info_msg, sub_key):
        # get pose of image
        ti = rospy.Time(secs=camera_msg.header.stamp.secs, nsecs=camera_msg.header.stamp.nsecs)
        self._last_t = ti
        try:
            transform = self._tf_buffer.lookup_transform(
                camera_msg.header.frame_id, self.map_frame, ti, rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Error: image_callback:", e)
            return

        t = transform.transform.translation
        t = np.array([t.x, t.y, t.z])
        q = transform.transform.rotation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")

        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, k] for k in range(3)]

        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.K, dtype=np.float32).reshape(3, 3)

        assert np.all(np.array(camera_info_msg.D) == 0.0), "Undistortion not implemented"
        D = np.array(camera_info_msg.D, dtype=np.float32).reshape(5, 1)
        
        # process pointcloud
        self._map.input_image(sub_key, semantic_img, R, t, K, D, camera_info_msg.height, camera_info_msg.width)
        self._image_process_counter += 1

    def pointcloud_callback(self, msg, sub_key):
        channels = ["x", "y", "z"] + self.param.subscriber_cfg[sub_key]["channels"]

        points = ros_numpy.numpify(msg)
        pts = np.empty((points.shape[0], 0))
        for ch in channels:
            p = points[ch]
            if len(p.shape) == 1:
                p = p[:, None]
            pts = np.append(pts, p, axis=1)

        # get pose of pointcloud
        ti = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs)
        self._last_t = ti
        try:
            transform = self._tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, ti, rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Error: pointcloud_callback: ", e)
            return

        t = transform.transform.translation
        t = np.array([t.x, t.y, t.z])
        q = transform.transform.rotation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        # process pointcloud
        self._map.input(pts, channels, R, t, 0, 0)
        self._pointcloud_process_counter += 1
        print("Pointclouds processed: ", self._pointcloud_process_counter)

    def update_pose(self, t):
        # get pose of base
        # t = rospy.Time.now()
        if self._last_t is None:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, self._last_t, rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Error: update_pose error: ", e)
            return
        t = transform.transform.translation
        trans = np.array([t.x, t.y, t.z])
        q = transform.transform.rotation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        self._map.move_to(trans, rot)

        self._map_t = t
        self._map_q = q

    def update_variance(self, t):
        self._map.update_variance()

    def update_time(self, t):
        self._map.update_time()

    def get_ros_params(self):
        # TODO fix this here when later launching with launch-file
        # This is currently {p} elevation_mapping")
        typ = "sim"
        para = os.path.join(self.root, f"config/{typ}_parameters.yaml")
        sens = os.path.join(self.root, f"config/{typ}_sensor_parameter.yaml")
        plugin = os.path.join(self.root, f"config/{typ}_plugin_config.yaml")

        os.system(f"rosparam delete /{self.node_name}")
        os.system(f"rosparam load {para} elevation_mapping")
        os.system(f"rosparam load {sens} elevation_mapping")
        os.system(f"rosparam load {plugin} elevation_mapping")

        self.subscribers = rospy.get_param("~subscribers")
        self.publishers = rospy.get_param("~publishers")
        self.initialize_frame_id = rospy.get_param("~initialize_frame_id", "base")
        self.initialize_tf_offset = rospy.get_param("~initialize_tf_offset", 0.0)
        self.pose_topic = rospy.get_param("~pose_topic", "pose")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.corrected_map_frame = rospy.get_param("~corrected_map_frame", "corrected_map")
        self.initialize_method = rospy.get_param("~initialize_method", "cubic")
        self.position_lowpass_alpha = rospy.get_param("~position_lowpass_alpha", 0.2)
        self.orientation_lowpass_alpha = rospy.get_param("~orientation_lowpass_alpha", 0.2)
        self.recordable_fps = rospy.get_param("~recordable_fps", 3.0)
        self.update_variance_fps = rospy.get_param("~update_variance_fps", 1.0)
        self.time_interval = rospy.get_param("~time_interval", 0.1)
        self.update_pose_fps = rospy.get_param("~update_pose_fps", 10.0)
        self.initialize_tf_grid_size = rospy.get_param("~initialize_tf_grid_size", 0.5)
        self.map_acquire_fps = rospy.get_param("~map_acquire_fps", 5.0)
        self.publish_statistics_fps = rospy.get_param("~publish_statistics_fps", 1.0)
        self.enable_pointcloud_publishing = rospy.get_param("~enable_pointcloud_publishing", False)
        self.enable_normal_arrow_publishing = rospy.get_param("~enable_normal_arrow_publishing", False)
        self.enable_drift_corrected_TF_publishing = rospy.get_param("~enable_drift_corrected_TF_publishing", False)
        self.use_initializer_at_start = rospy.get_param("~use_initializer_at_start", False)


if __name__ == "__main__":
    emw = ElevationMapWrapper()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            print("Error")
