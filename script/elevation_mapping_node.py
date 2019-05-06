#!/usr/bin/env python

# Simple talker demo that published std_msgs/Strings messages
# to the 'chatter' topic

import rospy
import numpy as np
# from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from elevation_mapping import ElevationMap
import ros_numpy
import tf
import tf.transformations as tftf
import time
# import cupy as cp


class ElevationMappingNode:
    def __init__(self, pointcloud_topic, pose_topic):
        # init ros
        self.elevation_map = ElevationMap()
        rospy.init_node('elevation_mapping_cupy', anonymous=False)
        rospy.Subscriber(pointcloud_topic, PointCloud2,
                         self.point_callback, queue_size=1)
        rospy.Subscriber(pose_topic, PoseWithCovarianceStamped,
                         self.pose_callback, queue_size=1)

        self.map_publisher = rospy.Publisher('elevation_map',
                                             GridMap, queue_size=1)
        self.listener = tf.TransformListener()

        self.translation = np.zeros(3)
        self.R = np.zeros((3, 3))

        self.stamp = rospy.Time.now()
        self.frame_id = 'map'

        # measurements
        self.time_sum = 0.0
        self.time_cnt = 0

        rospy.spin()


    def point_callback(self, msg):
        print('recieved pointcloud')
        frame_id = msg.header.frame_id
        frame_id = 'ghost_desired/' + frame_id
        print('frame_id is ', frame_id)
        # frame_id = 'ghost_desired/' + 'realsense_d435_front_camera_axis_aligned'
        try:
            (translation, quaternion) = self.listener.lookupTransform('odom',
                                                                      frame_id,
                                                                      msg.header.stamp)
        except:
            print('Could not get tf')
            self.publish_map()
            return
        R = tftf.quaternion_matrix(quaternion)[0:3, 0:3]
        self.stamp = msg.header.stamp
        pc = ros_numpy.numpify(msg)
        pc = pc.flatten()
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        points = points[~np.isnan(points).any(axis=1)]
        # print(points.shape)
        # n = 7 
        # points = np.concatenate([points for i in range(n)])
        points = points[0:5000]
        start = time.time()
        # print(points[100:200])
        translation = np.array(translation)
        position = np.array([translation[0], translation[1]])
        self.elevation_map.input(points, R, translation)
        self.elevation_map.move_to(position)
        # self.elevation_map.show()
        total_time = time.time() - start
        print(points.shape)
        print('total', total_time)
        self.time_sum += total_time
        self.time_cnt += 1
        print('average ', self.time_sum / self.time_cnt)
        self.publish_map()
        # print(points.shape)

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.translation = np.array([position.x, position.y, position.z])
        quaternion = (orientation.x, orientation.y,
                      orientation.z, orientation.w)
        self.R = tftf.quaternion_matrix(quaternion)[0:3, 0:3]
        # print('translation ', self.translation)
        # print('R ', self.R)

        position = np.array([position.x, position.y])
        # self.elevation_map.move_to(position)

    def publish_map(self):
        # start = time.time()
        msg = self.get_gridmap_msg()
        # print('get msg', time.time() - start)
        self.map_publisher.publish(msg)
        # print('publish', time.time() - start)

    def get_gridmap_msg(self):
        # print('get_gridmap_msg')
        start = time.time()
        msg = GridMap()
        msg.info.header.stamp = self.stamp
        msg.info.header.frame_id = self.frame_id
        msg.info.resolution = self.elevation_map.resolution
        msg.info.length_x = self.elevation_map.map_length
        msg.info.length_y = self.elevation_map.map_length
        msg.info.pose.position.x = self.elevation_map.center[0]
        msg.info.pose.position.y = self.elevation_map.center[1]
        msg.info.pose.position.z = 0
        msg.info.pose.orientation.w = 1.0
        msg.layers = ['elevation', 'variance']
        # print('info', time.time() - start)
        map_array = self.elevation_map.get_maps()
        # print('get_maps ', time.time() - start)
        # print(map_array)
        elevation_data = self.numpy_array_to_msg(map_array[0])
        variance_data = self.numpy_array_to_msg(map_array[1])
        # print('array_to_msg ', time.time() - start)
        msg.data = [elevation_data, variance_data]
        # msg.data.data = [list(elevation), list(variance)]
        # print(msg)
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
        # dim2.stride = dim.stride * a.shape[1]
        dim2.stride = a.shape[1]
        msg.layout.dim.append(dim2)
        msg.data = a.flatten()
        return msg


if __name__ == '__main__':
    try:
        pointcloud_topic = '/realsense_d435_front/depth/color/points'
        pose_topic = '/state_estimator/pose_in_odom'
        node = ElevationMappingNode(pointcloud_topic, pose_topic)
    except rospy.ROSInterruptException:
        pass
