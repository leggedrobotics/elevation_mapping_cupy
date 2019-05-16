#!/usr/bin/env python

# Simple talker demo that published std_msgs/Strings messages
# to the 'chatter' topic

import rospy
import rospkg
import numpy as np
# from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from elevation_mapping import ElevationMap, Parameter
import ros_numpy
import tf
import tf.transformations as tftf
import time
# import cupy as cp


class ElevationMappingNode:
    def __init__(self):
        # init ros
        rospy.init_node('elevation_mapping_cupy', anonymous=False)
        self.param = Parameter()
        self.load_rosparam()
        self.load_weights()
        self.elevation_map = ElevationMap(self.param)
        rospy.Subscriber(self.pointcloud_topic, PointCloud2,
                         self.point_callback, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseWithCovarianceStamped,
                         self.pose_callback, queue_size=1)

        self.map_publisher = rospy.Publisher('elevation_map',
                                             GridMap, queue_size=1)
        self.listener = tf.TransformListener()

        self.translation = np.zeros(3)
        self.R = np.zeros((3, 3))

        self.stamp = rospy.Time.now()

        # measurements
        self.time_sum = 0.0
        self.time_cnt = 0

        rospy.spin()

    def load_rosparam(self):
        self.param.use_cupy = rospy.get_param('~use_cupy', True)
        self.param.resolution = rospy.get_param('~resolution', 0.02)
        self.param.map_length = rospy.get_param('~map_length', 5.0)
        self.param.gather_mode = rospy.get_param('~gather_mode', 'max')
        self.param.sensor_noise_factor = rospy.get_param('~sensor_noise_factor', 0.05)
        self.param.mahalanobis_thresh = rospy.get_param('~mahalanobis_thresh', 2.0)
        self.param.outlier_variance = rospy.get_param('~outlier_variance', 0.01)
        self.param.time_variance = rospy.get_param('~time_variance', 0.01)
        self.param.initial_variance = rospy.get_param('~initial_variance', 10.0)
        self.pointcloud_topic = rospy.get_param('~pointcloud_topic', 'points')
        self.pose_topic = rospy.get_param('~pose_topic', 'pose')
        self.map_frame = rospy.get_param('~map_frame', 'map')

    def load_weights(self):
        filename = rospy.get_param('~weight_file', 'config/weights.yaml')
        rospack = rospkg.RosPack()
        root_path = rospack.get_path('elevation_mapping_cupy')
        filename = root_path + '/' + filename
        self.param.load_weights(filename)

    def point_callback(self, msg):
        print('recieved pointcloud')
        self.stamp = msg.header.stamp
        frame_id = msg.header.frame_id
        frame_id = 'ghost_desired/' + frame_id
        print('frame_id is ', frame_id)
        try:
            transform = self.listener.lookupTransform(self.map_frame,
                                                      frame_id,
                                                      msg.header.stamp)
            translation, quaternion = transform
        except:
            print('Could not get tf')
            return
        R = tftf.quaternion_matrix(quaternion)[0:3, 0:3]
        start = time.time()
        points = self.pointcloud2_to_array(msg)
        points = np.concatenate([points for i in range(4)])
        print('points convert', time.time() - start)
        start = time.time()
        translation = np.array(translation)
        position = np.array([translation[0], translation[1]])
        self.elevation_map.input(points, R, translation)
        total_time = time.time() - start
        print('number of points ', points.shape[0])
        print('total', total_time)
        self.time_sum += total_time
        self.time_cnt += 1
        print('average ', self.time_sum / self.time_cnt)

        start = time.time()
        self.publish_map()
        print('publish', time.time() - start)

    def pointcloud2_to_array(self, msg):
        pc = ros_numpy.numpify(msg).flatten()
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        points = points[~np.isnan(points).any(axis=1)]
        return points

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        position = np.array([position.x, position.y])
        self.elevation_map.move_to(position)

    def publish_map(self):
        start = time.time()
        msg = self.get_gridmap_msg()
        print('gridmap msg convertion ', time.time() - start)
        start = time.time()
        self.map_publisher.publish(msg)
        print('gridmap msg publish ', time.time() - start)

    def get_gridmap_msg(self):
        msg = GridMap()
        msg.info.header.stamp = self.stamp
        msg.info.header.frame_id = self.map_frame
        msg.info.resolution = self.elevation_map.resolution
        msg.info.length_x = self.elevation_map.map_length
        msg.info.length_y = self.elevation_map.map_length
        msg.info.pose.position.x = self.elevation_map.center[0]
        msg.info.pose.position.y = self.elevation_map.center[1]
        msg.info.pose.position.z = 0
        msg.info.pose.orientation.w = 1.0
        msg.layers = ['elevation', 'variance', 'traversability']
        map_array = self.elevation_map.get_maps()
        elevation_data = self.numpy_array_to_msg(map_array[0])
        variance_data = self.numpy_array_to_msg(map_array[1])
        traversability_data = self.numpy_array_to_msg(map_array[2])
        msg.data = [elevation_data, variance_data, traversability_data]
        return msg

    def numpy_array_to_msg(self, a):
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = 'column_index'
        dim.size = a.shape[0]
        dim.stride = a.shape[0]
        msg.layout.dim.append(dim)
        dim2 = MultiArrayDimension()
        dim2.label = 'row_index'
        dim2.size = a.shape[1]
        dim2.stride = a.shape[1]
        msg.layout.dim.append(dim2)
        msg.data = a.flatten()
        return msg


if __name__ == '__main__':
    try:
        node = ElevationMappingNode()
    except rospy.ROSInterruptException:
        pass
