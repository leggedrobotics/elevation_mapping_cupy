# # General
# import os
# import numpy as np
# import yaml

# from elevation_mapping_cupy import ElevationMap, Parameter



# # ROS2
# import rclpy
# from rclpy.node import Node
# from tf_transformations import quaternion_matrix
# import tf2_ros
# from ament_index_python.packages import get_package_share_directory
# from cv_bridge import CvBridge

# from sensor_msgs.msg import PointCloud2, Image, CameraInfo
# from grid_map_msgs.msg import GridMap
# from std_msgs.msg import Float32MultiArray, MultiArrayLayout as MAL, MultiArrayDimension as MAD

# from functools import partial
# import message_filters


# PDC_DATATYPE = {
#     "1": np.int8,
#     "2": np.uint8,
#     "3": np.int16,
#     "4": np.uint16,
#     "5": np.int32,
#     "6": np.uint32,
#     "7": np.float32,
#     "8": np.float64,
# }


# class ElevationMapWrapper(Node):
#     def __init__(self):
#         super().__init__('elevation_mapping')
#         self.node_name = "elevation_mapping"
#         self.root = get_package_share_directory("elevation_mapping_cupy")
#         weight_file = os.path.join(self.root, "config/core/weights.dat")
#         plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")
#         self.param = Parameter(use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file)
        
#         # ROS2
#         self.initialize_ros()
#         self.param.subscriber_cfg = self.subscribers

#         self.initialize_elevation_mapping()
#         # self.register_subscribers()
#         # self.register_publishers()
#         # self.register_timers()

#         self._last_t = None
        
     
       

#         # # ROS
#         # self.initalize_ros()
#         # self.param.subscriber_cfg = self.subscribers

        
#     def initialize_ros(self):        
#         self._tf_buffer = tf2_ros.Buffer()
#         self._listener = tf2_ros.TransformListener(self._tf_buffer, self)                
#         self.get_ros_params()


#     def initialize_elevation_mapping(self):
#         self.param.update()
#         # TODO add statistics across processed topics
#         self._pointcloud_process_counter = 0
#         self._image_process_counter = 0
#         self._map = ElevationMap(self.param)
#         self._map_data = np.zeros((self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32)
#         self._map_q = None
#         self._map_t = None

    


#     # def register_subscribers(self):
#     #     # check if CV bridge is needed
#     #     for config in self.param.subscriber_cfg.values():
#     #         if config["data_type"] == "image":
#     #             self.cv_bridge = CvBridge()
#     #             break

#     #     pointcloud_subs = {}
#     #     image_subs = {}
#     #     for key, config in self.subscribers.items():
#     #         if config["data_type"] == "image":
#     #             camera_sub = message_filters.Subscriber(config["topic_name_camera"], Image)
#     #             camera_info_sub = message_filters.Subscriber(config["topic_name_camera_info"], CameraInfo)
#     #             image_subs[key] = message_filters.ApproximateTimeSynchronizer(
#     #                 [camera_sub, camera_info_sub], queue_size=10, slop=0.5
#     #             )
#     #             image_subs[key].registerCallback(self.image_callback, key)

#     #         elif config["data_type"] == "pointcloud":
#     #             pointcloud_subs[key] = rospy.Subscriber(
#     #                 config["topic_name"], PointCloud2, self.pointcloud_callback, key
#     #             )

#     # def register_publishers(self):
#     #     self._publishers = {}
#     #     self._publishers_timers = []
#     #     for k, v in self.publishers.items():
#     #         self._publishers[k] = rospy.Publisher(f"/{self.node_name}/{k}", GridMap, queue_size=10)
#     #         # partial(.) allows to pass a default argument to a callback
#     #         self._publishers_timers.append(rospy.Timer(rospy.Duration(1 / v["fps"]), partial(self.publish_map, k)))

#     # def publish_map(self, k, t):
#     #     print("publish_map")
#     #     if self._map_q is None:
#     #         return
#     #     gm = GridMap()
#     #     gm.info.header.frame_id = self.map_frame
#     #     gm.info.header.stamp = rospy.Time.now()
#     #     gm.info.header.seq = 0
#     #     gm.info.resolution = self._map.resolution
#     #     gm.info.length_x = self._map.map_length
#     #     gm.info.length_y = self._map.map_length
#     #     gm.info.pose.position.x = self._map_t.x
#     #     gm.info.pose.position.y = self._map_t.y
#     #     gm.info.pose.position.z = 0.0
#     #     gm.info.pose.orientation.w = 1.0  # self._map_q.w
#     #     gm.info.pose.orientation.x = 0.0  # self._map_q.x
#     #     gm.info.pose.orientation.y = 0.0  # self._map_q.y
#     #     gm.info.pose.orientation.z = 0.0  # self._map_q.z
#     #     gm.layers = []
#     #     gm.basic_layers = self.publishers[k]["basic_layers"]
#     #     for i, layer in enumerate(self.publishers[k]["layers"]):
#     #         gm.layers.append(layer)
#     #         try:
#     #             self._map.get_map_with_name_ref(layer, self._map_data)
#     #             data = self._map_data.copy()
#     #             arr = Float32MultiArray()
#     #             arr.layout = MAL()
#     #             N = self._map_data.shape[0]
#     #             arr.layout.dim.append(MAD(label="column_index", size=N, stride=int(N * N)))
#     #             arr.layout.dim.append(MAD(label="row_index", size=N, stride=N))
#     #             arr.data = tuple(np.ascontiguousarray(data.T).reshape(-1))
#     #             gm.data.append(arr)
#     #         except:
#     #             if layer in gm.basic_layers:
#     #                 print("Error: Missed Layer in basic layers")

#     #     gm.outer_start_index = 0
#     #     gm.inner_start_index = 0

#     #     self._publishers[k].publish(gm)

#     # def register_timers(self):
#     #     self.timer_variance = rospy.Timer(rospy.Duration(1 / self.update_variance_fps), self.update_variance)
#     #     self.timer_pose = rospy.Timer(rospy.Duration(1 / self.update_pose_fps), self.update_pose)
#     #     self.timer_time = rospy.Timer(rospy.Duration(self.time_interval), self.update_time)

#     # def image_callback(self, camera_msg, camera_info_msg, sub_key):
#     #     # get pose of image
#     #     ti = rospy.Time(secs=camera_msg.header.stamp.secs, nsecs=camera_msg.header.stamp.nsecs)
#     #     self._last_t = ti
#     #     try:
#     #         transform = self._tf_buffer.lookup_transform(
#     #             camera_msg.header.frame_id, self.map_frame, ti, rospy.Duration(1.0)
#     #         )
#     #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#     #         print("Error: image_callback:", e)
#     #         return

#     #     t = transform.transform.translation
#     #     t = np.array([t.x, t.y, t.z])
#     #     q = transform.transform.rotation
#     #     R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

#     #     semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")

#     #     if len(semantic_img.shape) != 2:
#     #         semantic_img = [semantic_img[:, :, k] for k in range(3)]

#     #     else:
#     #         semantic_img = [semantic_img]

#     #     K = np.array(camera_info_msg.K, dtype=np.float32).reshape(3, 3)

#     #     assert np.all(np.array(camera_info_msg.D) == 0.0), "Undistortion not implemented"
#     #     D = np.array(camera_info_msg.D, dtype=np.float32).reshape(5, 1)
        
#     #     # process pointcloud
#     #     self._map.input_image(sub_key, semantic_img, R, t, K, D, camera_info_msg.height, camera_info_msg.width)
#     #     self._image_process_counter += 1

#     # def pointcloud_callback(self, msg, sub_key):
#     #     channels = ["x", "y", "z"] + self.param.subscriber_cfg[sub_key]["channels"]

#     #     points = ros_numpy.numpify(msg)
#     #     pts = np.empty((points.shape[0], 0))
#     #     for ch in channels:
#     #         p = points[ch]
#     #         if len(p.shape) == 1:
#     #             p = p[:, None]
#     #         pts = np.append(pts, p, axis=1)

#     #     # get pose of pointcloud
#     #     ti = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs)
#     #     self._last_t = ti
#     #     try:
#     #         transform = self._tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, ti, rospy.Duration(1.0))
#     #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#     #         print("Error: pointcloud_callback: ", e)
#     #         return

#     #     t = transform.transform.translation
#     #     t = np.array([t.x, t.y, t.z])
#     #     q = transform.transform.rotation
#     #     R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

#     #     # process pointcloud
#     #     self._map.input(pts, channels, R, t, 0, 0)
#     #     self._pointcloud_process_counter += 1
#     #     print("Pointclouds processed: ", self._pointcloud_process_counter)

#     # def update_pose(self, t):
#     #     # get pose of base
#     #     # t = rospy.Time.now()
#     #     if self._last_t is None:
#     #         return
#     #     try:
#     #         transform = self._tf_buffer.lookup_transform(
#     #             self.map_frame, self.base_frame, self._last_t, rospy.Duration(1.0)
#     #         )
#     #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#     #         print("Error: update_pose error: ", e)
#     #         return
#     #     t = transform.transform.translation
#     #     trans = np.array([t.x, t.y, t.z])
#     #     q = transform.transform.rotation
#     #     rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
#     #     self._map.move_to(trans, rot)

#     #     self._map_t = t
#     #     self._map_q = q

#     # def update_variance(self, t):
#     #     self._map.update_variance()

#     # def update_time(self, t):
#     #     self._map.update_time()
        

                   
#     def declare_parameters_from_yaml(self, yaml_file):
#         def declare_nested_parameters(parent_name, param_dict):
#             for sub_name, sub_value in param_dict.items():
#                 full_param_name = f"{parent_name}.{sub_name}"
            
#                 if isinstance(sub_value, dict):
#                     declare_nested_parameters(full_param_name, sub_value)
#                 else:
#                     if "elevation_mapping.ros__parameters" in full_param_name:
#                         full_param_name = full_param_name.replace("elevation_mapping.ros__parameters.", "")
#                     self.declare_parameter(full_param_name, sub_value)
                    
#         with open(yaml_file, 'r') as file:
#             params = yaml.safe_load(file)

#             # Set parameters in the node
#             for param_name, param_value in params.items():
#                 if isinstance(param_value, dict):
#                     # For nested dictionaries, set each nested parameter individually
#                     declare_nested_parameters(param_name, param_value)
#                 else:
#                     # For top-level parameters
#                     if "elevation_mapping.ros__parameters" in param_name:
#                         param_name = param_name.replace("elevation_mapping.ros__parameters.", "")
#                     self.declare_parameter(param_name, param_value)
                    
                    
     

#     def get_ros_params(self):
#         # TODO fix this here when later launching with launch-file
#         # This is currently {p} elevation_mapping")        
#         param_file = os.path.join(self.root, f"config/core/core_param.yaml")
#         self.declare_parameters_from_yaml(param_file)

#         # os.system(f"rosparam delete /{self.node_name}")
#         # os.system(f"rosparam load {para} elevation_mapping")        
#         # self.subscribers = self.get_parameter("subscribers").get_parameter_value().string_value
#         # self.publishers = self.get_parameter("publishers").get_parameter_value().string_value
#         self.initialize_frame_id = self.get_parameter("initialize_frame_id").get_parameter_value().string_array_value
#         self.initialize_tf_offset = self.get_parameter("initialize_tf_offset").get_parameter_value().double_value
#         self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
#         self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
#         self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
#         self.corrected_map_frame = self.get_parameter("corrected_map_frame").get_parameter_value().string_value
#         self.initialize_method = self.get_parameter("initialize_method").get_parameter_value().string_value
#         self.position_lowpass_alpha = self.get_parameter("position_lowpass_alpha").get_parameter_value().double_value
#         self.orientation_lowpass_alpha = self.get_parameter("orientation_lowpass_alpha").get_parameter_value().double_value        
#         self.update_variance_fps = self.get_parameter("update_variance_fps").get_parameter_value().double_value
#         self.time_interval = self.get_parameter("time_interval").get_parameter_value().double_value
#         self.update_pose_fps = self.get_parameter("update_pose_fps").get_parameter_value().double_value
#         self.initialize_tf_grid_size = self.get_parameter("initialize_tf_grid_size").get_parameter_value().double_value
#         self.map_acquire_fps = self.get_parameter("map_acquire_fps").get_parameter_value().double_value
#         self.publish_statistics_fps = self.get_parameter("publish_statistics_fps").get_parameter_value().double_value
#         self.enable_pointcloud_publishing = self.get_parameter("enable_pointcloud_publishing").get_parameter_value().bool_value
#         self.enable_normal_arrow_publishing = self.get_parameter("enable_normal_arrow_publishing").get_parameter_value().bool_value
#         self.enable_drift_corrected_TF_publishing = self.get_parameter("enable_drift_corrected_TF_publishing").get_parameter_value().bool_value
#         self.use_initializer_at_start = self.get_parameter("use_initializer_at_start").get_parameter_value().bool_value


# def main(args=None):
#     rclpy.init(args=args)
#     node = ElevationMapWrapper()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()