from elevation_mapping_cupy import ElevationMap
from elevation_mapping_cupy import Parameter

# General
import os 
import numpy as np

# ROS
import rospy
import ros_numpy
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2
import tf2_ros
import rospkg

class ElevationMapWrapper():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.root = rospack.get_path("elevation_mapping_cupy")
        self.initalize_elevation_mapping()
        
        # ROS
        self.initalize_ros()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()

    def initalize_elevation_mapping(self):
        weight_file = os.path.join(self.root, "config/weights.dat") 
        plugin_config_file = os.path.join(self.root, "config/plugin_config.yaml")
        param = Parameter( use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file )
        self._pointcloud_process_counter = 0
        self._map = ElevationMap(param)
        self._map_data = np.zeros((self._map.cell_n - 2, self._map.cell_n - 2), dtype=np.float32)
        
    def initalize_ros(self):
        rospy.init_node("elevation_mapping", anonymous=False)
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer)
        self.get_ros_params()
        
    def register_subscribers(self):
        pointcloud_subs = {}
        for config in self.subscribers.values(): 
            self.subscribers
            rospy.Subscriber(config["topic_name"], PointCloud2, self.pointcloud_callback, (config["channels"], config["fusion"]))
            
    def register_publishers(self):
        # TODO publishers
        print(self.publishers)
        
    def register_timers(self):
        self.timer_variance = rospy.Timer(rospy.Duration(1/self.update_variance_fps), self.update_variance)
        self.timer_pose = rospy.Timer(rospy.Duration(1/self.update_pose_fps), self.update_pose)
        self.timer_time = rospy.Timer(rospy.Duration(self.time_interval), self.update_time)
        
    def pointcloud_callback(self, data, config):
        # convert pcd into numpy array
        points = ros_numpy.numpify(data)
        out = np.copy( np.frombuffer(points.tobytes(), np.dtype(np.float32) ))
        out = out.reshape(out.shape[0],-1)[:,:3]
        points = points.astype(np.float32)
        
        # get pose of pointcloud
        t = rospy.Time(secs=data.header.stamp.secs, nsecs=data.header.stamp.nsecs)
        try:
            transform = self._tf_buffer.lookup_transform(self.map_frame, data.header.frame_id, t, rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return

        t = transform.transform.translation
        t = np.array( [t.x, t.y, t.z] )
        q = transform.transform.rotation
        R = quaternion_matrix([q.w, q.x, q.y, q.z])
            
        # process pointcloud
        self._map.input(points, 0, R, t, 0, 0)
        self._map.update_normal(self._map.elevation_map[0])
        self._pointcloud_process_counter += 1
        
    def update_pose(self, t):
        # get pose of base
        t = rospy.Time.now()
        try:
            transform = self._tf_buffer.lookup_transform(self.map_frame, self.base_frame, t, rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return        
        t = transform.transform.translation
        trans = np.array( [t.x, t.y, t.z] )
        q = transform.transform.rotation
        rot = quaternion_matrix([q.w, q.x, q.y, q.z])
        self._map.move_to(trans, rot)
        
    def update_variance(self, t):
        self._map.update_variance()
    
    def update_time(self, t):
        self._map.update_time()
    
    def get_ros_params(self):
        # TODO fix this here when later launching with launch-file
        # This is currently {p} elevation_mapping")
        para = os.path.join(self.root, "config/parameters.yaml")
        sens = os.path.join(self.root, "config/sensor_parameter.yaml")
        os.system(f"rosparam load {para} elevation_mapping")
        os.system(f"rosparam load {sens} elevation_mapping")
        
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
    
if __name__ == '__main__':
    emw = ElevationMapWrapper()
    
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            print("Error")